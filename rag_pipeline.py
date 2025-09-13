# rag_pipeline.py
# ------------------------------------------------------
# Run with: streamlit run rag_pipeline.py
# Requirements:
# pip install streamlit langchain openai chromadb pypdf python-dotenv
# ------------------------------------------------------

import os
import json
import logging
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

#------Imports LangChain + Chroma + OpenAI ----- #
#----------------------------------------------- #
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# ---------- Config/ Environment Variables ---------- #
#----------------------------------------------------#

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment.")

HISTORY_FILE = "chat_history.json"
DOCS_DIR = "./LegalDocs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "text-embedding-3-small"  
CHAT_MODEL = "gpt-4o-mini"            
TOP_K = 4

# ---------- Utilities (Save and Load Chat History) ---------- #
# ------------------------------------------------------------#

def save_chat_history(history: List[dict]):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ------------ PDF Text Extraction ------------ #
#---------------------------------------------- #

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for p in reader.pages:
        text_parts.append(p.extract_text() or "")
    return "\n".join(text_parts).strip()

#------------------- Text Chunking ------------ #
#----------------------------------------------- #

def chunk_text_simple(text: str, max_chars: int = 2500) -> List[str]:
    """Very basic chunker by characters (preserves sentences roughly)."""
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

# ------------------------ History Conversion --------------- #
# ----------------------------------------------------------- #
def convert_history_to_tuples(history: List[dict]) -> List[Tuple[str, str]]:
    """
    Convert [{'role':'user','content':...}, {'role':'assistant','content':...}, ...]
    into [('user message 1','assistant reply 1'), ...]
    """
    pairs = []
    pending_user = None
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None
    return pairs

# ---------- Vector DB and RAG chain setup ---------- #
# --------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_vectordb_and_chain(persist_dir: str = CHROMA_DIR):
    # embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

    # Chroma vector store (persisted)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # Chat LLM (OpenAI Chat model)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)

    # ConversationalRetrievalChain (RAG)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )

    return vectordb, qa_chain

vectordb, qa_chain = get_vectordb_and_chain()

# ---------- Index documents helper ---------- #
# -------------------------------------------- #
def index_documents_in_folder(folder: str = DOCS_DIR):
    """Index all PDFs/TXT under DOCS_DIR into Chroma (adds texts in chunks)."""
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

            # add texts with source metadata; Chroma will deduplicate by ids internally
            metadatas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
            vectordb.add_texts(texts=chunks, metadatas=metadatas)
            vectordb.persist()
        except Exception as e:
            logging.exception(f"Indexing failed for {fname}: {e}")

# ---------- Hybrid response (RAG first, fallback to direct chat) ---------- #
# --------------------------------------------------------------------------- #
def get_hybrid_response(question: str, full_history: List[dict]) -> str:
    """
    Attempt to answer using the RAG chain. If the RAG answer is empty or too short,
    do a direct ChatOpenAI call using recent chat history as context.
    """
    try:
        lc_history = convert_history_to_tuples(full_history)

        # LangChain RAG expects chat_history as list of tuples (user, assistant)
        # Call the chain:
        resp = qa_chain({"question": question, "chat_history": lc_history})
        answer = (resp.get("answer") or "").strip()
        source_docs = resp.get("source_documents", [])

        # If RAG returned a meaningful answer, include sources and return
        if answer and len(answer.split()) > 6:
            if source_docs:
                sources = []
                for d in source_docs:
                    src = None
                    # metadata may differ depending on how it was added
                    if hasattr(d, "metadata"):
                        src = d.metadata.get("source")
                    elif isinstance(d, dict):
                        src = d.get("metadata", {}).get("source")
                    if src:
                        if src not in sources:
                            sources.append(src)
                if sources:
                    answer += "\n\n**Sources:** " + ", ".join(sources[:6])
            return answer

        # Fallback: call ChatOpenAI directly with brief system prompt + last 6 user/assistant turns
        chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)
        system_prompt = (
            "You are LegalBot, an expert AI assistant specialized in South African law. "
            "Your focus areas are eviction, rent disputes, substandard housing, family law, divorce, child custody, "
             "and child support. "
            "Provide answers that are concise, accurate, professional, and written in plain, understandable language. "
            "Do NOT reference the source documents or say 'in this document'. "
            "If you do not have enough information to answer the question reliably, respond with: "
            "'I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources.' "
            "Always base your answers on South African law. "
            "Avoid uncertain language such as 'might', 'maybe', 'could', or 'would'. "
            "Do not provide legal advice outside the specified areas."
        )

        # Build messages for fallback (OpenAI-style list of dicts)
        messages = [{"role": "system", "content": system_prompt}]
        # include last up to 6 turns from session history
        recent = full_history[-6:]
        for m in recent:
            messages.append({"role": m["role"], "content": m["content"]})
        # include current question
        messages.append({"role": "user", "content": question})

        # Use the underlying ChatOpenAI client to get the answer (ChatOpenAI is callable and returns a LangChain Message)
        # But easiest: call ChatOpenAI directly via its 'generate' or call interface.
        llm_response = chat_llm(messages)  # returns an object with .content
        # llm_response here is an AI message object (LangChain ChatMessage)
        fallback_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        return fallback_answer

    except Exception as e:
        logging.exception(f"Hybrid response failed: {e}")
        return "⚠️ Sorry — there was an internal error while processing your request."

# ---------- Streamlit UI ------------------------------------------------#
# ------------------------------------------------------------------------ #
def run_chatbot():
    st.set_page_config(page_title="Legal AI Chatbot (RAG)", layout="wide")
    st.title("⚖️ Legal AI Chatbot (RAG)")
    st.caption("Ask about eviction, rent disputes, family law and child custody (South Africa).")

    # ensure docs directory exists
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Sidebar controls
    with st.sidebar:
        if st.button("🔁 Re-index documents"):
            with st.spinner("Indexing documents into Chroma..."):
                index_documents_in_folder(DOCS_DIR)
            st.success("Indexing complete.")
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            save_chat_history([])
            st.experimental_rerun()

    # session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
        # ensure format is list of dicts {role, content}
        if not isinstance(st.session_state.chat_history, list):
            st.session_state.chat_history = []

    # File uploader to ingest docs
    uploaded = st.file_uploader("Upload PDF or TXT to index into the RAG DB", type=["pdf", "txt"])
    if uploaded is not None:
        dest = os.path.join(DOCS_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name} to {DOCS_DIR}. Indexing...")
        index_documents_in_folder(DOCS_DIR)
        st.success("Indexing finished.")

    # Display chat history
    for m in st.session_state.chat_history:
        role = m.get("role", "user")
        with st.chat_message(role):
            st.markdown(m.get("content", ""))

    # Chat input
    user_q = st.chat_input("💬 Ask your legal question...")
    if user_q:
        # append user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # generate hybrid response
        reply = get_hybrid_response(user_q, st.session_state.chat_history)

        # append and display
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        save_chat_history(st.session_state.chat_history)
