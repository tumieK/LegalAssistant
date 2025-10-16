# # rag_pipeline.py
# # ------------------------------------------------------
# # Run with: streamlit run rag_pipeline.py
# # Requirements:
# # pip install streamlit langchain openai chromadb pypdf python-dotenv
# # ------------------------------------------------------

# import os
# import json
# import logging
# from typing import List, Tuple, Optional

# import streamlit as st
# from dotenv import load_dotenv
# from pypdf import PdfReader

# # Use built-in sqlite3 for Python >= 3.11
# import sqlite3

# #------Imports LangChain + Chroma + OpenAI ----- #
# #----------------------------------------------- #
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain.chains import ConversationalRetrievalChain

# # ---------- Config/ Environment Variables ---------- #
# #----------------------------------------------------#

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise RuntimeError("Set OPENAI_API_KEY in the environment.")

# HISTORY_FILE = "chat_history.json"
# DOCS_DIR = "./LegalDocs"
# CHROMA_DIR = "./chroma_db"
# EMBED_MODEL = "text-embedding-3-small"  
# CHAT_MODEL = "gpt-4o-mini"            
# TOP_K = 4

# # ---------- Utilities (Save and Load Chat History) ---------- #
# # ------------------------------------------------------------#

# def save_chat_history(history: List[dict]):
#     with open(HISTORY_FILE, "w", encoding="utf-8") as f:
#         json.dump(history, f, ensure_ascii=False, indent=2)

# def load_chat_history() -> List[dict]:
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return []

# # ------------ PDF Text Extraction ------------ #
# #---------------------------------------------- #

# def extract_text_from_pdf(path: str) -> str:
#     reader = PdfReader(path)
#     text_parts = []
#     for p in reader.pages:
#         text_parts.append(p.extract_text() or "")
#     return "\n".join(text_parts).strip()

# #------------------- Text Chunking ------------ #
# #----------------------------------------------- #

# def chunk_text_simple(text: str, max_chars: int = 2500) -> List[str]:
#     """Very basic chunker by characters (preserves sentences roughly)."""
#     if not text:
#         return []
#     parts = []
#     cur = ""
#     for paragraph in text.split("\n"):
#         if len(cur) + len(paragraph) + 1 <= max_chars:
#             cur += paragraph + "\n"
#         else:
#             if cur.strip():
#                 parts.append(cur.strip())
#             cur = paragraph + "\n"
#     if cur.strip():
#         parts.append(cur.strip())
#     return parts

# # ------------------------ History Conversion --------------- #
# # ----------------------------------------------------------- #
# def convert_history_to_tuples(history: List[dict]) -> List[Tuple[str, str]]:
#     """
#     Convert [{'role':'user','content':...}, {'role':'assistant','content':...}, ...]
#     into [('user message 1','assistant reply 1'), ...]
#     """
#     pairs = []
#     pending_user = None
#     for msg in history:
#         role = msg.get("role", "")
#         content = msg.get("content", "")
#         if role == "user":
#             pending_user = content
#         elif role == "assistant" and pending_user is not None:
#             pairs.append((pending_user, content))
#             pending_user = None
#     return pairs

# # ---------- Vector DB and RAG chain setup ---------- #
# # --------------------------------------------------- #

# @st.cache_resource(show_spinner=False)
# def get_vectordb_and_chain(persist_dir: str = CHROMA_DIR):
#     # embeddings using OpenAI
#     embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

#     # Chroma vector store (persisted)
#     vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

#     # Retriever
#     retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

#     # Chat LLM (OpenAI Chat model)
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)

#     # ConversationalRetrievalChain (RAG)
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff",
#     )

#     return vectordb, qa_chain

# vectordb, qa_chain = get_vectordb_and_chain()

# # ---------- Index documents helper ---------- #
# # -------------------------------------------- #
# def index_documents_in_folder(folder: str = DOCS_DIR):
#     """Index all PDFs/TXT under DOCS_DIR into Chroma (adds texts in chunks)."""
#     os.makedirs(folder, exist_ok=True)

#     for fname in os.listdir(folder):
#         path = os.path.join(folder, fname)
#         if not os.path.isfile(path):
#             continue
#         try:
#             if fname.lower().endswith(".pdf"):
#                 raw = extract_text_from_pdf(path)
#             elif fname.lower().endswith(".txt"):
#                 with open(path, "r", encoding="utf-8", errors="ignore") as f:
#                     raw = f.read()
#             else:
#                 continue

#             chunks = chunk_text_simple(raw, max_chars=2000)
#             if not chunks:
#                 continue

#             # add texts with source metadata; Chroma will deduplicate by ids internally
#             metadatas = [{"source": fname, "chunk_index": i} for i in range(len(chunks))]
#             vectordb.add_texts(texts=chunks, metadatas=metadatas)
#             vectordb.persist()
#         except Exception as e:
#             logging.exception(f"Indexing failed for {fname}: {e}")

# # ---------- Hybrid response (RAG first, fallback to direct chat) ---------- #
# # --------------------------------------------------------------------------- #
# # def get_hybrid_response(question: str, full_history: List[dict], model_name: str = "gpt-4o-mini") -> str:
    
# #     """
# #     Attempt to answer using the RAG chain. If the RAG answer is empty or too short,
# #     do a direct ChatOpenAI call using recent chat history as context.
# #     """
# #     try:
# #         lc_history = convert_history_to_tuples(full_history)

# #         # LangChain RAG expects chat_history as list of tuples (user, assistant)
# #         # Call the chain:
# #         resp = qa_chain.invoke({"question": question, "chat_history": lc_history})
# #         answer = (resp.get("answer") or "").strip()
# #         source_docs = resp.get("source_documents", [])

# #         # If RAG returned a meaningful answer, include sources and return
# #         if answer and len(answer.split()) > 6:
# #             if source_docs:
# #                 sources = []
# #                 for d in source_docs:
# #                     src = None
# #                     # metadata may differ depending on how it was added
# #                     if hasattr(d, "metadata"):
# #                         src = d.metadata.get("source")
# #                     elif isinstance(d, dict):
# #                         src = d.get("metadata", {}).get("source")
# #                     if src:
# #                         if src not in sources:
# #                             sources.append(src)
# #                 if sources:
# #                     answer += "\n\n**Sources:** " + ", ".join(sources[:6])
# #             return answer

# #         # Fallback: call ChatOpenAI directly with brief system prompt + last 6 user/assistant turns
# #         chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=CHAT_MODEL, temperature=0.0)
# #         system_prompt = (
# #             "You are LegalBot, an expert AI assistant specialized in South African law. "
# #             "Your focus areas are eviction, rent disputes, substandard housing, family law, divorce, child custody, "
# #              "and child support. "
# #             "Provide answers that are concise, accurate, professional, and written in plain, understandable language. "
# #             "Do NOT reference the source documents or say 'in this document'. "
# #             "If you do not have enough information to answer the question reliably, respond with: "
# #             "'I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources.' "
# #             "Always base your answers on South African law. "
# #             "Avoid uncertain language such as 'might', 'maybe', 'could', or 'would'. "
# #             "Do not provide legal advice outside the specified areas."
# #         )

# #         # Build messages for fallback (OpenAI-style list of dicts)
# #         messages = [{"role": "system", "content": system_prompt}]
# #         # include last up to 6 turns from session history
# #         recent = full_history[-6:]
# #         for m in recent:
# #             messages.append({"role": m["role"], "content": m["content"]})
# #         # include current question
# #         messages.append({"role": "user", "content": question})

# #         # Use the underlying ChatOpenAI client to get the answer (ChatOpenAI is callable and returns a LangChain Message)
# #         # But easiest: call ChatOpenAI directly via its 'generate' or call interface.
# #         llm_response = chat_llm.invoke(messages)  # returns an object with .content
# #         # llm_response here is an AI message object (LangChain ChatMessage)
# #         fallback_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
# #         return fallback_answer

# #     except Exception as e:
# #         logging.exception(f"Hybrid response failed: {e}")
# #         return "⚠️ Sorry — there was an internal error while processing your request."


# # def get_hybrid_response(question: str, full_history: List[dict], model_name: str = "gpt-4o-mini") -> str:
# #     """
# #     Attempt to answer using the RAG chain first. If the RAG answer is empty or too short,
# #     fallback to a direct ChatOpenAI call using recent chat history as context.
# #     Always append a Sources section:
# #       - If documents are found, list their filenames.
# #       - If no documents, note that the answer is based on AI training knowledge.
# #     """
# #     try:
# #         lc_history = convert_history_to_tuples(full_history)

# #         # Call the RAG chain
# #         resp = qa_chain.invoke({"question": question, "chat_history": lc_history})
# #         answer = (resp.get("answer") or "").strip()
# #         source_docs = resp.get("source_documents", [])

# #         # --- Handle RAG answer ---
# #         if answer and len(answer.split()) > 6:
# #             sources = []
# #             if source_docs:
# #                 for d in source_docs:
# #                     src = None
# #                     if hasattr(d, "metadata"):
# #                         src = d.metadata.get("source")
# #                     elif isinstance(d, dict):
# #                         src = d.get("metadata", {}).get("source")
# #                     if src and src not in sources:
# #                         sources.append(src)
# #             if sources:
# #                 answer += "\n\n**Sources:** " + ", ".join(sources[:6])
# #             else:
# #                 answer += "\n\n**Source:** Based on knowledge learned during AI training"
# #             return answer

# #         # --- Fallback: direct ChatOpenAI ---
# #         chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, temperature=0.0)
# #         system_prompt = (
# #             "You are LegalBot, an expert AI assistant specialized in South African law. "
# #             "Your focus areas are eviction, rent disputes, substandard housing, family law, divorce, child custody, "
# #             "and child support. "
# #             "Provide concise, accurate, professional answers in plain, understandable language. "
# #             "If you do not have enough information to answer the question reliably, respond with: "
# #             "'I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources.' "
# #             "Always base your answers on South African law. "
# #             "Avoid uncertain language such as 'might', 'maybe', 'could', or 'would'. "
# #             "Do not provide legal advice outside the specified areas."
# #         )

# #         # Build messages
# #         messages = [{"role": "system", "content": system_prompt}]
# #         recent = full_history[-6:]  # include last up to 6 messages
# #         for m in recent:
# #             messages.append({"role": m["role"], "content": m["content"]})
# #         messages.append({"role": "user", "content": question})

# #         llm_response = chat_llm.invoke(messages)  # returns object with .content
# #         fallback_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

# #         # Append fallback source
# #         fallback_answer += "\n\n**Source:** Based on knowledge learned during AI training"

# #         return fallback_answer

# #     except Exception as e:
# #         logging.exception(f"Hybrid response failed: {e}")
# #         return "⚠️ Sorry — there was an internal error while processing your request."

# def get_hybrid_response(question: str, full_history: List[dict], model_name: str = "gpt-4o-mini") -> str:
#     """
#     Attempt to answer using the RAG chain. If the RAG answer is empty or too short,
#     do a direct ChatOpenAI call using recent chat history as context.
#     Always append a Sources section — either from retrieved docs or model training.
#     """
#     try:
#         lc_history = convert_history_to_tuples(full_history)

#         # 1️⃣ Try the RAG pipeline first
#         resp = qa_chain.invoke({"question": question, "chat_history": lc_history})
#         answer = (resp.get("answer") or "").strip()
#         source_docs = resp.get("source_documents", [])

#         # 2️⃣ If RAG produced a meaningful answer
#         if answer and len(answer.split()) > 6:
#             sources = []
#             for d in source_docs:
#                 src = None
#                 if hasattr(d, "metadata"):
#                     src = d.metadata.get("source")
#                 elif isinstance(d, dict):
#                     src = d.get("metadata", {}).get("source")
#                 if src and src not in sources:
#                     sources.append(src)

#             if sources:
#                 answer += "\n\n**Sources:** " + ", ".join(sources[:6])
#             else:
#                 answer += "\n\n**Source:** Based on LegalBot’s trained understanding of South African law."

#             return answer

#         # 3️⃣ Fallback: direct model (no docs)
#         chat_llm = ChatOpenAI(
#             openai_api_key=OPENAI_API_KEY,
#             model_name=CHAT_MODEL,
#             temperature=0.0
#         )

#         system_prompt = (
#             "You are LegalBot, an expert AI assistant specialized in South African law. "
#             "Your focus areas are eviction, rent disputes, substandard housing, family law, divorce, child custody, "
#             "and child support. "
#             "Provide answers that are concise, accurate, professional, and written in plain, understandable language. "
#             "If you do not have enough information, respond with: "
#             "'I'm still learning and do not have sufficient data regarding your question. Please consult official legal resources.' "
#             "Always base your answers on South African law and never reference source documents."
#         )

#         messages = [{"role": "system", "content": system_prompt}]
#         recent = full_history[-6:]
#         for m in recent:
#             messages.append({"role": m["role"], "content": m["content"]})
#         messages.append({"role": "user", "content": question})

#         llm_response = chat_llm.invoke(messages)
#         fallback_answer = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

#         # Always add a fallback source note
#         fallback_answer += "\n\n**Source:** Based on LegalBot’s training and general understanding of South African law."

#         return fallback_answer

#     except Exception as e:
#         logging.exception(f"Hybrid response failed: {e}")
#         return "⚠️ Sorry — there was an internal error while processing your request."


# # ---------- Streamlit UI ------------------------------------------------#
# # ------------------------------------------------------------------------ #
# def run_chatbot():
#     st.set_page_config(page_title="Legal AI Chatbot (RAG)", layout="wide")
#     st.title("⚖️ Legal AI Chatbot (RAG)")
#     st.caption("Ask about eviction, rent disputes, family law and child custody (South Africa).")

#     # ensure docs directory exists
#     os.makedirs(DOCS_DIR, exist_ok=True)

#     # Sidebar controls
#     with st.sidebar:
#         if st.button("🔁 Re-index documents"):
#             with st.spinner("Indexing documents into Chroma..."):
#                 index_documents_in_folder(DOCS_DIR)
#             st.success("Indexing complete.")
#         if st.button("🗑️ Clear chat history"):
#             st.session_state.chat_history = []
#             save_chat_history([])
#             st.experimental_rerun()

#     # session state init
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = load_chat_history()
#         # ensure format is list of dicts {role, content}
#         if not isinstance(st.session_state.chat_history, list):
#             st.session_state.chat_history = []

#     # File uploader to ingest docs
#     uploaded = st.file_uploader("Upload PDF or TXT to index into the RAG DB", type=["pdf", "txt"])
#     if uploaded is not None:
#         dest = os.path.join(DOCS_DIR, uploaded.name)
#         with open(dest, "wb") as f:
#             f.write(uploaded.getbuffer())
#         st.success(f"Saved {uploaded.name} to {DOCS_DIR}. Indexing...")
#         index_documents_in_folder(DOCS_DIR)
#         st.success("Indexing finished.")

#     # Display chat history
#     for m in st.session_state.chat_history:
#         role = m.get("role", "user")
#         with st.chat_message(role):
#             st.markdown(m.get("content", ""))

#     # Chat input
#     user_q = st.chat_input("💬 Ask your legal question...")
#     if user_q:
#         # append user message
#         st.session_state.chat_history.append({"role": "user", "content": user_q})
#         with st.chat_message("user"):
#             st.markdown(user_q)

#         # generate hybrid response
#         reply = get_hybrid_response(user_q, st.session_state.chat_history)

#         # append and display
#         st.session_state.chat_history.append({"role": "assistant", "content": reply})
#         with st.chat_message("assistant"):
#             st.markdown(reply)

#         save_chat_history(st.session_state.chat_history)

# rag_pipeline.py
# ------------------------------------------------------
# Run with: streamlit run rag_pipeline.py
# ------------------------------------------------------

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
