# case_agent.py
# Retrieval + Answer generation + Self-evaluation for casefiles DBs.

import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment before using case_agent.py")

CASE_SUMMARY_DB = "./chroma_case_summary"
CASE_RAW_DB = "./chroma_case_raw"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# System prompt for answer generation (customize further as needed)
SYSTEM_PROMPT = (
    "You are LegalBot (Casefiles). Use the provided context (summaries and raw text) strictly to answer questions "
    "about South African law, focusing on eviction, family law, custody, and related areas. "
    "If the context doesn't contain enough to answer, say: "
    "'I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources.'"
)

def _get_dbs():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    summary_db = Chroma(persist_directory=CASE_SUMMARY_DB, embedding_function=embeddings)
    raw_db = Chroma(persist_directory=CASE_RAW_DB, embedding_function=embeddings)
    return summary_db, raw_db

def _retrieve(question: str, k_summary: int = 3, k_raw: int = 3):
    summary_db, raw_db = _get_dbs()
    summaries = summary_db.similarity_search(question, k=k_summary) if summary_db else []
    raw_chunks = raw_db.similarity_search(question, k=k_raw) if raw_db else []
    # results are list of Document objects (langchain). We'll return text and metadata.
    return summaries, raw_chunks

def _build_context_text(summaries, raw_chunks, max_chars=6000) -> str:
    parts = []
    total = 0
    # prefer summaries first for concise context
    for doc in summaries:
        txt = getattr(doc, "page_content", str(doc))
        if total + len(txt) > max_chars:
            break
        parts.append(f"[SUMMARY] {txt}")
        total += len(txt)
    for doc in raw_chunks:
        txt = getattr(doc, "page_content", str(doc))
        if total + len(txt) > max_chars:
            break
        # include source metadata if present
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "")
        parts.append(f"[RAW - {src}] {txt}")
        total += len(txt)
    return "\n\n".join(parts).strip()

def _call_llm(prompt_text: str) -> str:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)
    try:
        # prefer predict if available
        out = llm.predict(prompt_text)
    except Exception:
        out_obj = llm([{"role": "user", "content": prompt_text}])
        out = getattr(out_obj, "content", str(out_obj))
    return out

def _evaluator_check(question: str, draft_answer: str, sources: List[str]) -> Tuple[bool, str]:
    """
    Run a quick overlap heuristic then ask an LLM evaluator. Returns (is_valid, explanation).
    """
    # quick heuristic: check if draft_answer mentions important terms present in sources
    joined_sources = "\n\n".join(sources)[:8000]
    eval_prompt = (
        "You are an evaluator. Determine whether the ANSWER is supported by the SOURCE TEXTS and whether it answers "
        "the QUESTION correctly and in accordance with South African law. Reply with EXACTLY one line starting with "
        "'VALID' or 'INVALID', then a short reason (one or two sentences).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{draft_answer}\n\n"
        f"SOURCES:\n{joined_sources}\n\n"
        "If the answer is speculative or not supported, reply 'INVALID'."
    )
    verdict = _call_llm(eval_prompt)
    # parse first token
    v = verdict.strip().splitlines()[0].strip().upper()
    is_valid = v.startswith("VALID")
    explanation = "\n".join(verdict.strip().splitlines()[1:]).strip()
    return is_valid, explanation or verdict.strip()

def get_casefile_response(question: str, chat_history: List[dict] = None) -> str:
    """
    Main entrypoint: retrieve from case_summary and raw DBs, draft an answer, run evaluator.
    Returns the final answer string (or 'no knowledge' message).
    """
    summaries, raw_chunks = _retrieve(question, k_summary=3, k_raw=4)

    if not summaries and not raw_chunks:
        return "I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources."

    # build context from summaries first, then raw chunks
    context = _build_context_text(summaries, raw_chunks, max_chars=6000)

    prompt_text = (
        SYSTEM_PROMPT + "\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer concisely, and if you cite sources, include the source filenames at the end."
    )

    draft = _call_llm(prompt_text)

    # gather source texts for evaluation: use page_content and metadata source
    source_texts = []
    src_names = []
    for d in (summaries + raw_chunks):
        source_texts.append(getattr(d, "page_content", str(d)))
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source")
        if src and src not in src_names:
            src_names.append(src)

    is_valid, explanation = _evaluator_check(question, draft, source_texts)

    if is_valid:
        if src_names:
            return draft.strip() + "\n\n**Sources:** " + ", ".join(src_names[:6])
        return draft.strip()
    else:
        return "I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources."

# Example quick test
if __name__ == "__main__":
    q = "What does South African law require for a lawful eviction?"
    print(get_casefile_response(q, chat_history=[]))
