# import os
import streamlit as st
from dotenv import load_dotenv

# your existing imports
from rag_pipeline import get_hybrid_response  

# NEW import for casefiles agent
from case_agent import get_casefile_response  

load_dotenv()

# --- Page setup ---
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- Sidebar (login + options) ---
with st.sidebar:
    st.title("⚖️ Legal Assistant")
    st.markdown("AI-powered legal help (educational only).")

    # Authentication (dummy for now, you can expand later)
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username and password:  # simple check; replace with DB later
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Enter username and password.")
        st.stop()
    else:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

# --- Chat UI (only if logged in) ---
st.title("💬 Legal Assistant Chat")

# Select knowledge source
mode = st.radio(
    "Choose knowledge source:",
    ["General (LegalDocs)", "Casefiles"],
    index=0,
    horizontal=True
)

# Chat history init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_q = st.chat_input("Ask a legal question...")
if user_q:
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    if mode == "Casefiles":
        reply = get_casefile_response(user_q, st.session_state.chat_history)
    else:
        reply = get_hybrid_response(user_q, st.session_state.chat_history)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)

# Footer disclaimer
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This assistant provides information for educational purposes only. "
    "It is not a substitute for professional legal advice."
)
