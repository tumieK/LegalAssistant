# import streamlit as st
# from dotenv import load_dotenv

# from rag_pipeline import get_hybrid_response  
# from case_agent import get_casefile_response  
# from merge_agent import get_merged_response  

# load_dotenv()

# # --- Page setup ---
# st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# # --- Sidebar (login + options) ---
# with st.sidebar:
#     st.title("⚖️ Legal Assistant")
#     st.markdown("AI-powered legal help (educational only).")

#     # Session state defaults
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False
#     if "username" not in st.session_state:
#         st.session_state.username = ""

#     # Login form (basic demo)
#     if not st.session_state.logged_in:
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")

#         if st.button("Login"):
#             if username and password:  # simple check; replace with DB later
#                 st.session_state.logged_in = True
#                 st.session_state.username = username
#                 st.success(f"Welcome, {username}!")
#                 st.rerun()
#             else:
#                 st.error("Enter username and password.")
#     else:
#         st.success(f"Logged in as {st.session_state.username}")
#         if st.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.username = ""
#             st.rerun()

# # --- Main UI ---

# if not st.session_state.logged_in:
#     # Custom CSS for background + centered login box
#     st.markdown(
#         """
#         <style>
#         /* Full-page background */
#         .stApp {
#             background-image: url("file://C:/Users/CSI/LegalAssistant/images/background.jpg");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }
#         /* Center container */
#         .login-container {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             height: 90vh;
#         }
#         /* Login box styling */
#         .login-box {
#             background: rgba(255, 255, 255, 0.85);
#             padding: 2rem;
#             border-radius: 12px;
#             box-shadow: 0 4px 15px rgba(0,0,0,0.3);
#             width: 300px;
#             text-align: center;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     # Login box
#     st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)

#     st.subheader("🔑 Login")
#     username = st.text_input("Username (Main UI)")
#     password = st.text_input("Password (Main UI)", type="password")

#     if st.button("Login (Main UI)"):
#         if username and password:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success(f"Welcome, {username}!")
#             st.rerun()
#         else:
#             st.error("Enter username and password.")

#     st.markdown("</div></div>", unsafe_allow_html=True)

# # --- Chat UI (after login) ---
# if st.session_state.logged_in:
#     st.title("💬 Legal Assistant Chat")

#     # Initialize chat history
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Display past messages
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     # --- Chat input ---
#     user_q = st.chat_input("Ask a legal question...")
#     if user_q:
#         # Save user query
#         st.session_state.chat_history.append({"role": "user", "content": user_q})

#         # ✅ Immediately display the user’s message
#         with st.chat_message("user"):
#             st.markdown(user_q)

#         # Step 1: Get responses from both agents
#         general_reply = get_hybrid_response(user_q, st.session_state.chat_history)
#         casefile_reply = get_casefile_response(user_q, st.session_state.chat_history)

#         # Step 2: Merge them with merger agent
#         reply = get_merged_response(
#             user_q,
#             general_reply,
#             casefile_reply,
#         )

#         # Save assistant reply
#         st.session_state.chat_history.append({"role": "assistant", "content": reply})

#         # Display assistant reply
#         with st.chat_message("assistant"):
#             st.markdown(reply)

#         # --- Footer disclaimer ---
#         st.markdown("---")
#         st.caption(
#             "⚠️ Disclaimer: This assistant provides information for educational purposes only. "
#             "It is not a substitute for professional legal advice."
#         )

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import get_hybrid_response  
from case_agent import get_casefile_response  
from merge_agent import get_merged_response  

load_dotenv()

# --- Page setup ---
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- Session state defaults ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Full-page login with Streamlit image ---
if not st.session_state.logged_in:
    # Show background image (fills width)
    st.image("images/background.jpg", use_column_width=True)

    # Create empty space before login box (to center vertically a bit)
    st.write("\n" * 5)

    # Center login box horizontally
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username and password:  # replace with real authentication
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Enter username and password.")

# --- Chat UI (after login) ---
if st.session_state.logged_in:
    st.title(f"💬 Legal Assistant Chat - Logged in as {st.session_state.username}")

    # Sidebar for chat history
    with st.sidebar:
        st.header("📜 Chat History")
        if st.session_state.chat_history:
            for i, msg in enumerate(st.session_state.chat_history):
                role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                st.markdown(f"**{role}:** {msg['content']}")
        else:
            st.caption("No history yet.")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_history = []
            st.rerun()

    # Display past messages in chat UI
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
