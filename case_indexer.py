import os
import io
import logging
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your .env file before running case_indexer.py")

CASEFILES_DIR = "./casefiles"
CASE_SUMMARY_DB = "./chroma_case_summary"
CASE_RAW_DB = "./chroma_case_raw"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO)


def extract_text_from_pdf(path: str) -> str:
    """Try normal PDF text extraction first, then OCR if needed."""
    try:
        reader = PdfReader(path)
        text_parts = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                text_parts.append(txt)

        if text_parts:  
            return "\n".join(text_parts).strip()


        logging.info(f"OCR processing {path} (no text layer found)")
        ocr_text = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    ocr_text.append(text)
        return "\n".join(ocr_text).strip()

    except Exception as e:
        logging.error(f"Failed to extract text from {path}: {e}")
        return ""


def chunk_text_simple(text: str, max_chars: int = 2000) -> List[str]:
    """Chunk text into smaller parts for embedding."""
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


def summarize_text(text: str, llm: ChatOpenAI) -> str:
    """Generate concise legal case summary."""
    prompt = (
        "Summarize the legal case below in 180-300 words. Include: court (if present), year (if present), "
        "key legal issue(s), short statement of facts, decision/outcome, and 3-5 short definitions of key terms.\n\n"
        f"{text[:6000]}"
    )
    try:
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        logging.error(f"Summary generation failed: {e}")
        return f"[Summary generation failed: {e}]"



def index_casefiles(case_dir: str = CASEFILES_DIR,
                    summary_dir: str = CASE_SUMMARY_DB,
                    raw_dir: str = CASE_RAW_DB):
    os.makedirs(case_dir, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    summary_db = Chroma(persist_directory=summary_dir, embedding_function=embeddings)
    raw_db = Chroma(persist_directory=raw_dir, embedding_function=embeddings)

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
                logging.warning(f"Unsupported file type: {fname}")
                continue

            if not raw_text.strip():
                logging.warning(f"{fname} is empty even after OCR, skipping")
                continue

            chunks = chunk_text_simple(raw_text, max_chars=2000)
            metadatas = [{"source": fname, "type": "raw", "chunk_index": i} for i in range(len(chunks))]
            raw_db.add_texts(texts=chunks, metadatas=metadatas)


            summary = summarize_text(raw_text, llm)
            summary_db.add_texts(texts=[summary], metadatas=[{"source": fname, "type": "summary"}])

            logging.info(f"Indexed {fname}: raw chunks={len(chunks)} summary_length={len(summary)}")

        except Exception as e:
            logging.exception(f"Failed to index {fname}: {e}")

    print(f"\n✅ Indexing complete.\nRaw DB: {raw_dir}\nSummary DB: {summary_dir}")



if __name__ == "__main__":
    index_casefiles()
