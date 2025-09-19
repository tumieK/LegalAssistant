import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import get_hybrid_response  
from case_agent import get_casefile_response  
from merge_agent import get_merged_response  

load_dotenv()

# --- Page setup ---
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- Sidebar (login + options) ---
with st.sidebar:
    st.title("⚖️ Legal Assistant")
    st.markdown("AI-powered legal help (educational only).")

    # Session state defaults
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    # Login form
    if not st.session_state.logged_in:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username and password:  # simple check; replace with DB later
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Enter username and password.")
    else:
        st.success(f"Logged in as {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

# --- Main UI ---
if st.session_state.logged_in:
    st.title("💬 Legal Assistant Chat")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- Chat input ---
user_q = st.chat_input("Ask a legal question...")
if user_q:
    # Save user query
    st.session_state.chat_history.append({"role": "user", "content": user_q})

    # ✅ Immediately display the user’s message
    with st.chat_message("user"):
        st.markdown(user_q)

    # Step 1: Get responses from both agents
    general_reply = get_hybrid_response(user_q, st.session_state.chat_history)
    casefile_reply = get_casefile_response(user_q, st.session_state.chat_history)

    # Step 2: Merge them with merger agent
    reply = get_merged_response(
        user_q,
        general_reply,
        casefile_reply,
    )

    # Save assistant reply
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Display assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)

    # --- Footer disclaimer ---
    st.markdown("---")
    st.caption(
        "⚠️ Disclaimer: This assistant provides information for educational purposes only. "
        "It is not a substitute for professional legal advice."
    )
else:
    st.info("👆 Please log in from the sidebar to start using the assistant.")
