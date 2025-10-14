import os #file/folder operations
import re #regular expressions
import json #read/write JSON files
import logging #logging errors
import numpy as np #Math for embeddings similarity
from PyPDF2 import PdfReader #Extract text from PDF files
import streamlit as st #Web app framework
import openai #Calls OpenAI API for embedding and chat responses

# API Key
openai.api_key = OPENAI_API_KEY

#chroma persistance= None

HISTORY_FILE = "chat_history.json"

# Save / Load Chat History
def save_chat_history(chat_history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# === Embedding Functions === //check again
def get_embedding(text, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=engine)
    return response["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

#Text chucking and PDF extraction
def chunk_text(text, max_tokens=300):
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk.split()) + len(sentence.split()) <= max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Load Legal Docs into Memory 
def load_documents(folder_path):
    document_chunks = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return document_chunks

    for file_name in os.listdir(folder_path):
        try:
            full_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".txt"):
                with open(full_path, "r", encoding="utf-8") as file:
                    content = file.read()
            elif file_name.endswith(".pdf"):
                content = extract_text_from_pdf(full_path)
            else:
                continue

            chunks = chunk_text(content)
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                document_chunks.append({
                    "content": chunk,
                    "embedding": embedding,
                    "source": f"{file_name}_chunk{i}"
                })
        except Exception as e:
            logging.error(f"Failed to load {file_name}: {e}")
    return document_chunks

# Generating a response from the bot
def get_bot_response(message, conversation_history, document_chunks):
    try:
        question_embedding = get_embedding(message)
        similarities = []
        for chunk in document_chunks:
            sim = cosine_similarity(question_embedding, chunk["embedding"])
            similarities.append({"score": sim, "content": chunk["content"]})

        top_chunks = sorted(similarities, key=lambda x: x["score"], reverse=True)[:3]
        context = "\n\n".join([chunk["content"] for chunk in top_chunks if chunk["score"] > 0.6])

        system_prompt = (
            "You are LegalBot, an AI assistant trained on South African law, "
            "specialising in eviction, rent disputes, substandard housing, family law, divorce, "
            "child custody, domestic violence, and child support. "
            "Provide clear, concise, professional answers based on the latest South African legal principles. "
            "Do NOT mention 'in this document'. If no relevant legal information is found, respond with: "
            "'I'm still learning and do not have enough data regarding your question. Please check official legal resources.' "
            "Only answer for South African law. Avoid words like might, maybe, could, or would."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4o", #check the model version so try to hard code 2025-04-14 and 2024-07-18
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation_history[-6:],
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"}
            ]
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        logging.error(f"Error getting response: {e}")
        return "⚠️ Sorry, I couldn't process your request."

# Streamlit UI 
st.set_page_config(page_title="Legal AI Chatbot", page_icon="⚖️", layout="wide")

st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    save_chat_history([])
    st.rerun()

st.markdown(
    """
    <style>
        .main { background-color: #f4f8fb; }
        .stChatMessage { font-size: 16px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("⚖️ Legal AI Chatbot")
st.caption("Ask legal questions about **eviction, custody, family law** in South Africa.")

# Session State Init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = load_documents("./LegalDocs")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload a legal document (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file:
    file_path = os.path.join("./LegalDocs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.document_chunks = load_documents("./LegalDocs")
    st.success(f"✅ {uploaded_file.name} uploaded and processed!")

# Display Chat History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Chat Input
user_input = st.chat_input("💬 Ask your legal question...")
if user_input and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_reply = get_bot_response(user_input, st.session_state.chat_history, st.session_state.document_chunks)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # Save after every message
    save_chat_history(st.session_state.chat_history)
