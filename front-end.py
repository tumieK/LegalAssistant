import streamlit as st  
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import torch

# === Hugging Face Settings ===
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
#MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_ID ="tinyllama/tinyllama-1.1b-chat-v1.0"
#MODEL_ID = "meta-llama/meta-llama-3-8b-instruct"
MODEL_ID ="gpt2"
# === Chat History Persistence ===
HISTORY_FILE = "legal_chat_history.json"

# === Load and Cache the LLM pipeline ===
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float32,
    ).to("cpu")
    #device=0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512,
        )

pipe = load_model()

# === Text Formatting ===
def format_text(text):
    if not text or not isinstance(text,str):
        return "No response"
    text = text.replace('**', '\n ')
    text = text.capitalize()
    text = re.sub(r'([.!?])\s*(\w)', lambda m: m.group(1) + m.group(2).upper(), text)
    return text

# === Chat History I/O ===
def load_chat_history():
    try:
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_chat_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# === LLM Response ===
def get_bot_response(message: str) -> str:
    # prompt = (
    #     "You are LegalBot, an AI legal assistant helping disadvantaged individuals understand eviction law, family law, "
    #     "and other legal topics in South Africa.\n"
    #     f"User: {message}\nLegalBot:"
    # )
    prompt = (
    "Answer the following question about South African law clearly and simply:\n"
    f"{message}"
    )

    try:
        response = pipe(prompt)
        if  not response or "generated_text" not in response[0]:
            return "Sorry,no response was generated."
        generated= response[0]["generated_text"]
        return generated[len(prompt):].strip() if generated else "Sorry, empty response."
    except Exception as e:
        return f"Sorry, something went wrong: {e}"

# === Streamlit UI ===
st.set_page_config(page_title="Legal Assistant Chatbot", page_icon="⚖️")
st.title("Legal Assistant Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
    st.session_state.chat_history.setdefault("Session 1", [])

# === Sidebar ===
with st.sidebar:
    st.header("Chat Sessions")
    session_names = list(st.session_state.chat_history.keys())
    selected_session = st.selectbox("Choose a session", session_names, index=session_names.index(st.session_state.current_session))

    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session

    if st.button("New Session"):
        new_name = f"Session {len(st.session_state.chat_history) + 1}"
        st.session_state.chat_history[new_name] = []
        st.session_state.current_session = new_name

# === Display Chat History ===
for msg in st.session_state.chat_history[st.session_state.current_session]:
    role = msg["role"]
    if role == "You":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**LegalBot:** {format_text(msg['content'])}")

# === Input ===
user_input = st.chat_input("Ask a legal question about eviction, custody, etc:")
if user_input is not None and user_input.strip():
    st.session_state.chat_history[st.session_state.current_session].append({"role": "You", "content": user_input})
    response = get_bot_response(user_input)
    st.session_state.chat_history[st.session_state.current_session].append({"role": "LegalBot", "content": response})
    save_chat_history(st.session_state.chat_history)
    st.rerun()

    
