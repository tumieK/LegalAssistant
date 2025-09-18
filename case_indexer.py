# case_indexer.py
# Index case files (PDF/TXT) from ./casefiles into two Chroma DBs:
#  - ./chroma_case_summary  (one summary doc per case)
#  - ./chroma_case_raw      (raw chunks from the case)

import os
import logging
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader

# Use same LangChain imports as your existing pipeline for compatibility
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment before running case_indexer.py")

# Config - keep in sync with your main pipeline
CASEFILES_DIR = "./casefiles"
CASE_SUMMARY_DB = "./chroma_case_summary"
CASE_RAW_DB = "./chroma_case_raw"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts).strip()

def chunk_text_simple(text: str, max_chars: int = 2000) -> List[str]:
    if not text:
        return []
    parts = []
    cur = ""
    for paragraph in text.split("\n"):
        if len(cur) + len(paragraph) + 1 <= max_chars:
            cur += paragraph + "\n"
        else:
            if cur.strip():
                parts.append(cur.strip())
            cur = paragraph + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts

# def summarize_text(text: str, llm: ChatOpenAI) -> str:
#     # keep summary prompt concise; you may customize
#     prompt = (
#         "Summarize the legal case below in 180-300 words. Include: court (if present), year (if present), "
#         "key legal issue(s), short statement of facts, decision/outcome, and 3-5 short definitions of key terms.\n\n"
#         f"{text[:6000]}"  # limit token usage by passing the start of the doc; for long docs you can chunk pre-summarize
#     )
#     # ChatOpenAI as used in your environment supports .generate/call; use simple .predict or .__call__
#     try:
#         resp = llm.invoke(prompt)
#     except Exception:
#         # fallback to calling as function; compatibility across langchain versions
#         obj = llm([{"role": "user", "content": prompt}])
#         resp = getattr(obj, "content", str(obj))
#     return resp.strip()

def summarize_text(text: str, llm: ChatOpenAI) -> str:
    prompt = (
        "Summarize the legal case below in 180-300 words. Include: court (if present), year (if present), "
        "key legal issue(s), short statement of facts, decision/outcome, and 3-5 short definitions of key terms.\n\n"
        f"{text[:6000]}"
    )
    try:
        resp = llm.invoke(prompt)   # returns AIMessage
        return resp.content.strip()  # use .content, not .strip() directly
    except Exception as e:
        return f"[Summary generation failed: {e}]"


def index_casefiles(case_dir: str = CASEFILES_DIR,
                    summary_dir: str = CASE_SUMMARY_DB,
                    raw_dir: str = CASE_RAW_DB):
    os.makedirs(case_dir, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    summary_db = Chroma(persist_directory=summary_dir, embedding_function=embeddings)
    raw_db = Chroma(persist_directory=raw_dir, embedding_function=embeddings)

    # LLM for summarization
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)

    for fname in sorted(os.listdir(case_dir)):
        path = os.path.join(case_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                raw_text = extract_text_from_pdf(path)
            elif fname.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            else:
                # skip unsupported file types
                continue

            if not raw_text.strip():
                logging.warning(f"{fname} is empty, skipping")
                continue

            # 1) add raw chunks to raw_db
            chunks = chunk_text_simple(raw_text, max_chars=2000)
            metadatas = [{"source": fname, "type": "raw", "chunk_index": i} for i in range(len(chunks))]
            raw_db.add_texts(texts=chunks, metadatas=metadatas)

            # 2) create summary and add to summary_db (one summary doc per case)
            summary = summarize_text(raw_text, llm)
            summary_db.add_texts(texts=[summary], metadatas=[{"source": fname, "type": "summary"}])

            logging.info(f"Indexed {fname}: raw chunks={len(chunks)} summary_length={len(summary)}")
        except Exception as e:
            logging.exception(f"Failed to index {fname}: {e}")

    # persist
    print("Indexing complete. Raw DB:", raw_dir, "Summary DB:", summary_dir)

if __name__ == "__main__":
    index_casefiles()
