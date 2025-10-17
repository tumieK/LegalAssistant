import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment before using case_agent.py")

CASE_SUMMARY_DB = "./chroma_case_summary"
CASE_RAW_DB = "./chroma_case_raw"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

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
    return summaries, raw_chunks

def _build_context_text(summaries, raw_chunks, max_chars=6000) -> str:
    parts = []
    total = 0
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
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "")
        parts.append(f"[RAW - {src}] {txt}")
        total += len(txt)
    return "\n\n".join(parts).strip()

def _call_llm(prompt_text: str) -> str:
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)
    try:
        out = llm.invoke(prompt_text)  
        return out.content.strip()
    except Exception:
        out_obj = llm([{"role": "user", "content": prompt_text}])
        if hasattr(out_obj, "content"):
            return out_obj.content.strip()
        return str(out_obj)


def _evaluator_check(question: str, draft_answer: str, sources: List[str]) -> Tuple[bool, str]:
    """
    Run a quick overlap heuristic then ask an LLM evaluator. Returns (is_valid, explanation).
    """
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

    context = _build_context_text(summaries, raw_chunks, max_chars=6000)

    prompt_text = (
        SYSTEM_PROMPT + "\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer concisely, and if you cite sources, include the source filenames at the end."
    )

    draft = _call_llm(prompt_text)

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

if __name__ == "__main__":
    q = "What does South African law require for a lawful eviction?"
    print(get_casefile_response(q, chat_history=[]))
