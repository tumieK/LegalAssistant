import streamlit as st
import random
import json
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import get_hybrid_response  
from case_agent import get_casefile_response  
from merge_agent import get_merged_response  
from streamlit_js_eval import streamlit_js_eval
from intent import get_intent 
from answer_lenght import adjust_answer_length 

load_dotenv()

# --- Load JSON files ---
with open("data/users.json", "r") as f:
    users = json.load(f)

with open("data/lawyers.json", "r") as f:
    lawyers_db = json.load(f)

# --- Helper functions ---
def check_login(username, password):
    """Check if username/password match JSON file"""
    return any(u["username"] == username and u["password"] == password for u in users)

def get_lawyers(legal_issue):
    """Return lawyers for a legal category"""
    return lawyers_db.get(legal_issue, [])

def format_lawyers_list(lawyers):
    if not lawyers:
        return "Sorry, no pro bono lawyers found nearby for your case."
    message = "Here are some pro bono lawyers near you:\n\n"
    for i, lawyer in enumerate(lawyers, 1):
        message += f"{i}. {lawyer['name']}, {lawyer['address']}\n"
    return message

# --- Page setup ---
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- Session state defaults ---
for key, default in [
    ("logged_in", False), 
    ("username", ""), 
    ("chat_sessions", {"Session 1": []}), 
    ("current_session", "Session 1"), 
    ("show_lawyers", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

    # --- Full-page login & registration ---
if not st.session_state.logged_in:
    st.image("images/background.jpg", width=600)
    st.write("\n" * 3)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        option = st.radio("Choose an option", ["Login", "Register"])

        if option == "Login":
            st.subheader("🔑 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username and password:
                    if check_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Enter username and password.")

        else:  # --- Registration ---
            st.subheader("📝 Register New Account")
            role = st.selectbox("I am a...", ["User", "Lawyer"])
            specialisation = st.selectbox(
            "Select your legal specialisation",
            ["Eviction", "Divorce"] ) 
            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")

            if role == "Lawyer":
                name = st.text_input("Full Name")
                address = st.text_input("Address")
                lat = st.number_input("Latitude", format="%.6f")
                lng = st.number_input("Longitude", format="%.6f")

            if st.button("Register"):
                if new_username and new_password:
                    if any(u["username"] == new_username for u in users):
                        st.error("⚠️ Username already exists.")
                    else:
                        if role == "User":
                            users.append({"username": new_username, "password": new_password})
                            with open("data/users.json", "w") as f:
                                json.dump(users, f, indent=2)
                        else:
                            # Lawyer
                            if "General" not in lawyers_db:
                                lawyers_db["General"] = []
                            lawyers_db["General"].append({
                                "name": name,
                                "address": address,
                                "lat": lat,
                                "lng": lng
                            })
                            with open("data/lawyers.json", "w") as f:
                                json.dump(lawyers_db, f, indent=2)

                        st.success("✅ Registration successful! Please login now.")
                else:
                    st.error("Please fill in all required fields.")


# --- Chat UI (after login) ---
if st.session_state.logged_in:
    st.title(f"💬 Legal Assistant Chat - Logged in as {st.session_state.username}")

    # Sidebar
    with st.sidebar:
        st.header("📂 Chat Sessions")
        st.subheader("⚙️ Settings")
        llm_choice = st.selectbox(
            "Choose AI Model",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        )
        if st.button("➕ New Session"):
            new_name = f"Session {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_name] = []
            st.session_state.current_session = new_name

        session_names = list(st.session_state.chat_sessions.keys())
        selected = st.selectbox("Choose a session", session_names, index=session_names.index(st.session_state.current_session))
        st.session_state.current_session = selected

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_sessions = {"Session 1": []}
            st.session_state.current_session = "Session 1"
            st.session_state.show_lawyers = False
            st.rerun()

        st.markdown("---")
        st.subheader("📍 Pro Bono Lawyer Finder")
        if st.button("Find Pro Bono Lawyers"):
            st.session_state.show_lawyers = True

    # Current session history
    current_history = st.session_state.chat_sessions[st.session_state.current_session]

    # --- Display chat history ---
    if current_history:
        for msg in current_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("This session has no messages yet. Start chatting below 👇")

    # --- Lawyer Finder Section ---
    LEGAL_KEYWORDS = ["Eviction", "Divorce"]

    if st.session_state.show_lawyers:
        st.info("We will use your location to find the nearest pro bono lawyers.")
        legal_issue = st.selectbox("Select Legal Issue", options=LEGAL_KEYWORDS)

        # --- Get user location ---
        location = streamlit_js_eval(
            js_expressions="""
            new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(
                    pos => resolve({latitude: pos.coords.latitude, longitude: pos.coords.longitude}),
                    err => resolve(null)
                )
            })
            """,
            key="lawyer_location",
            default=None
        )

        if location:
            lat, lng = location["latitude"], location["longitude"]

            # Get lawyers from JSON
            lawyers = get_lawyers(legal_issue)
            reply = format_lawyers_list(lawyers)
            current_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

            # --- Map display ---
            if lawyers:
                df = pd.DataFrame(lawyers)  # Must contain 'lat' and 'lng' keys
                st.map(df)
        else:
            st.warning("Could not get your location. Please allow location access in your browser.")

    # --- Chat input (always visible after login) ---
    user_q = st.chat_input("Ask a legal question...")
    if user_q:
        current_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Step 1: Detect intent
        intent = get_intent(user_q, llm_choice)

        # Step 2: Route based on intent
        if intent == "LEGAL_QUESTION":
            general_reply = get_hybrid_response(user_q, current_history, llm_choice)
            casefile_reply = get_casefile_response(user_q, current_history)
            reply = get_merged_response(user_q, general_reply, casefile_reply, llm_choice)

        elif intent == "FIND_LAWYERS":
            st.session_state.show_lawyers = True
            reply = "Sure, let me help you find a pro bono lawyer nearby. Please select the issue from the left panel."

        elif intent == "GREETING":
            reply = "Hello 👋! How can I help you with your legal concerns today?"

        elif intent == "SMALLTALK":
            reply = "I’m here mainly for legal assistance ⚖️, but happy to chat a little!"

        else:
            reply = "Could you clarify if this is a legal question, or do you need lawyer recommendations?"

        # Step 2.5: Adjust answer length based on question type
        reply = adjust_answer_length(user_q, reply, llm_choice)

        # Step 3: Display assistant reply
        current_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        # --- Footer disclaimer ---
        st.markdown("---")
        st.caption("⚠️ Disclaimer: This assistant provides information for educational purposes only. It is not a substitute for professional legal advice.")
