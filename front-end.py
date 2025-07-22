import streamlit as st
from dotenv import load_dotenv
import os
from pipeline_setup import load_rag_chain

# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="SA Legal Assistant", page_icon="⚖️")
st.title("🇿🇦 South African Legal Assistant ⚖️")

@st.cache_resource
def get_chain():
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        st.error("❌ HUGGINGFACEHUB_API_TOKEN is missing. Please set it in your .env file.")
        st.stop()
    return load_rag_chain(hf_token=hf_token)

rag_chain = get_chain()

# Session state to hold history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Ask a legal question:", key="user_input")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        result = rag_chain.invoke({
            "question": user_input,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((user_input, result["answer"]))
        st.markdown(f"**Answer:** {result['answer']}")

# Display chat history
if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
