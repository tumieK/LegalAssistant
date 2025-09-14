import streamlit as st
from auth import login_user, register_user
from rag_pipeline import get_hybrid_response  # your chatbot functions

import sys

# Patch sqlite to use pysqlite3
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


# --- Session state init ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Authentication UI ---
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Logged in as {username}")
            st.rerun()  # forces UI to show chatbot
        else:
            st.error("Invalid credentials")
else:
    st.subheader(f"Welcome, {st.session_state.username}")
    # --- Chatbot UI ---
    user_q = st.chat_input("Ask a legal question...")
    if user_q:
        # save user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        # get response
        reply = get_hybrid_response(user_q, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        # display messages
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
