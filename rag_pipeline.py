import os
import json
import logging
from typing import List, Tuple
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import sqlite3  # built-in

# ---- LangChain + Chroma + OpenAI Imports ---- #
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import EnsembleRetriever

# ---------- Config / Environment ---------- #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment.")

HISTORY_FILE = "chat_history.json"
DOCS_DIR = "./LegalDocs"
CHROMA_DIR = "./chroma_db"  # default doc index (used for Streamlit uploads)
CASE_SUMMARY_DB = "./chroma_case_summary"  # prebuilt summary DB
CASE_RAW_DB = "./chroma_case_raw"          # prebuilt full-text DB

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 4

# ---------- Save / Load Chat History ---------- #
def save_chat_history(history: List[dict]):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ---------- PDF Text Extraction ---------- #
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

# ---------- Simple Text Chunking ---------- #
def chunk_text_simple(text: str, max_chars: int = 2500) -> List[str]:
    if not text:
        return []
    parts, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 <= max_chars:
            cur += para + "\n"
        else:
            if cur.strip():
                parts.append(cur.strip())
            cur = para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts

# ---------- History Conversion ---------- #
def convert_history_to_tuples(history: List[dict]) -> List[Tuple[str, str]]:
    pairs, pending_user = [], None
    for msg in history:
        role, content = msg.get("role", ""), msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user:
            pairs.append((pending_user, content))
            pending_user = None
    return pairs

# ---------- Vector DB and RAG Chain ---------- #
@st.cache_resource(show_spinner=False)
def get_vectordb_and_chain():
    """Combine both summary + raw Chroma DBs for hybrid retrieval."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

    # Load available databases
    retrievers = []
    weights = []

    if os.path.exists(CASE_SUMMARY_DB):
        summary_db = Chroma(persist_directory=CASE_SUMMARY_DB, embedding_function=embeddings)
        retrievers.append(summary_db.as_retriever(search_kwargs={"k": TOP_K}))
        weights.append(0.7)
        print("✅ Loaded summary database")

    if os.path.exists(CASE_RAW_DB):
        raw_db = Chroma(persist_directory=CASE_RAW_DB, embedding_function=embeddings)
        retrievers.append(raw_db.as_retriever(search_kwargs={"k": TOP_K}))
        weights.append(0.3)
        print("✅ Loaded raw database")

    # fallback to default upload DB if nothing else exists
    if not retrievers:
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retrievers.append(vectordb.as_retriever(search_kwargs={"k": TOP_K}))
        weights.append(1.0)
        print("⚠️ Using default document DB only (no summary/raw DBs found)")

    # Combine multiple retrievers
    if len(retrievers) > 1:
        combined_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
    else:
        combined_retriever = retrievers[0]

    # LLM setup
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=CHAT_MODEL,
        temperature=0.0
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=combined_retriever,
        return_source_documents=True,
        chain_type="stuff",
    )

    return combined_retriever, qa_chain

retriever, qa_chain = get_vectordb_and_chain()

# ---------- Index Documents (for Streamlit uploads) ---------- #
def index_documents_in_folder(folder: str = DOCS_DIR):
    os.makedirs(folder, exist_ok=True)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            continue
        try:
            if fname.lower().endswith(".pdf"):
                raw = extract_text_from_pdf(path)
            elif fname.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
            else:
                continue

            chunks = chunk_text_simple(raw, max_chars=2000)
            if not chunks:
                continue

            embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
            vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

            metadatas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
            vectordb.add_texts(texts=chunks, metadatas=metadatas)
            vectordb.persist()

        except Exception as e:
            logging.exception(f"Indexing failed for {fname}: {e}")

# ---------- Hybrid RAG Response ---------- #
def get_hybrid_response(question: str, full_history: List[dict], model_name: str = CHAT_MODEL) -> str:
    try:
        lc_history = convert_history_to_tuples(full_history)

        # --- Step 1: Try RAG ---
        resp = qa_chain.invoke({"question": question, "chat_history": lc_history})
        answer = (resp.get("answer") or "").strip()
        source_docs = resp.get("source_documents", [])

        if answer and len(answer.split()) > 6:
            sources = []
            for d in source_docs:
                src = None
                if hasattr(d, "metadata"):
                    src = d.metadata.get("source")
                elif isinstance(d, dict):
                    src = d.get("metadata", {}).get("source")
                if src and src not in sources:
                    sources.append(src)

            if sources:
                answer += "\n\n**Sources:** " + ", ".join(sources[:6])
            else:
                answer += "\n\n**Source:** Based on LegalBot’s trained understanding of South African law."
            return answer

        # --- Step 2: Fallback direct model ---
        chat_llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model_name,
            temperature=0.0
        )
        system_prompt = (
            "You are LegalBot, an expert AI assistant specialized in South African law. "
            "Focus on eviction, rent disputes, family law, divorce, child custody, and child support. "
            "Be concise, accurate, and professional. If you lack info, respond: "
            "'I'm still learning and do not have sufficient data regarding your question. "
            "Please consult official legal resources.'"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for m in full_history[-6:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})
        llm_response = chat_llm.invoke(messages)
        fallback_answer = getattr(llm_response, "content", str(llm_response))
        fallback_answer += "\n\n**Source:** Based on LegalBot’s general training."
        return fallback_answer

    except Exception as e:
        logging.exception(f"Hybrid response failed: {e}")
        return "⚠️ Internal error while processing your request."

# ---------- Streamlit UI ---------- #
def run_chatbot():
    st.set_page_config(page_title="Legal AI Chatbot (RAG)", layout="wide")
    st.title("⚖️ Legal AI Chatbot (Dual-RAG)")
    st.caption("Retrieves from both case summaries and full case texts (South Africa).")

    os.makedirs(DOCS_DIR, exist_ok=True)

    with st.sidebar:
        if st.button("🔁 Re-index uploaded documents"):
            with st.spinner("Indexing documents into Chroma..."):
                index_documents_in_folder(DOCS_DIR)
            st.success("Indexing complete.")
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.experimental_rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
        if not isinstance(st.session_state.chat_history, list):
            st.session_state.chat_history = []

    uploaded = st.file_uploader("Upload PDF or TXT to index into local RAG DB", type=["pdf", "txt"])
    if uploaded is not None:
        dest = os.path.join(DOCS_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}. Indexing...")
        index_documents_in_folder(DOCS_DIR)
        st.success("✅ Document indexed.")

    for m in st.session_state.chat_history:
        with st.chat_message(m.get("role", "user")):
            st.markdown(m.get("content", ""))

    user_q = st.chat_input("💬 Ask your legal question...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        reply = get_hybrid_response(user_q, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
        save_chat_history(st.session_state.chat_history)
