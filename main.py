# import streamlit as st
# import random
# import json
# import pandas as pd
# from dotenv import load_dotenv
# from rag_pipeline import get_hybrid_response
# from case_agent import get_casefile_response
# from merge_agent import get_merged_response
# from streamlit_js_eval import streamlit_js_eval
# from intent import get_intent
# from answer_lenght import adjust_answer_length
# from geocode import geocode_address
# import time

# # --- Load environment ---

# load_dotenv()

# # --- Load JSON files ---

# with open("data/users.json", "r") as f:
#     users = json.load(f)

# with open("data/lawyers.json", "r") as f:
#     lawyers_db = json.load(f)

# # --- Helper functions ---

# def check_login(username, password):
#     """Check if username/password match JSON file"""
#     return any(u["username"] == username and u["password"] == password for u in users)

# def get_lawyers(legal_issue):
#     """Return lawyers for a legal category"""
#     return lawyers_db.get(legal_issue, [])

# def format_lawyers_list(lawyers):
#     if not lawyers:
#         return "Sorry, no pro bono lawyers found nearby for your case."
#     message = "Here are some pro bono lawyers near you:\n\n"
#     for i, lawyer in enumerate(lawyers, 1):
#         cell_display = f" (📞 {lawyer['cell']})" if 'cell' in lawyer else ""
#         message += f"{i}. {lawyer['name']}, {lawyer['address']}{cell_display}\n"
#     return message

# # --- Page setup ---

# st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# # --- Session state defaults ---

# for key, default in [
#     ("logged_in", False),
#     ("username", ""),
#     ("chat_sessions", {"Session 1": []}),
#     ("current_session", "Session 1"),
#     ("show_lawyers", False),
#     ("chat_history", []),
#     ("current_topic", None)
# ]:
#     if key not in st.session_state:
#         st.session_state[key] = default

# # --- Login & Registration ---

# if not st.session_state.logged_in:
#     st.image("images/background.jpg", width=600)
#     st.write("\n" * 3)
#     col1, col2, col3 = st.columns([1, 2, 1])

#     with col2:
#         option = st.radio("Choose an option", ["Login", "Register"])

#         # --- Login section ---
#         if option == "Login":
#             st.subheader("🔑 Login")
#             username = st.text_input("Username")
#             password = st.text_input("Password", type="password")
#             if st.button("Login"):
#                 if username and password:
#                     if check_login(username, password):
#                         st.session_state.logged_in = True
#                         st.session_state.username = username
#                         st.rerun()
#                     else:
#                         st.error("Invalid username or password.")
#                 else:
#                     st.error("Enter username and password.")

#         # --- Registration section ---
#         else:
#             st.subheader("📝 Register New Account")
#             role = st.selectbox("I am a...", ["User", "Lawyer"])

#             new_username = st.text_input("Choose a Username")
#             new_password = st.text_input("Choose a Password", type="password")

#             if role == "Lawyer":
#                 st.markdown("### 🏛️ Lawyer Details")
#                 name = st.text_input("Full Name")
#                 address = st.text_input("Office Address (e.g. 54 Groenewound Street, Universitas, Bloemfontein)")
#                 cell = st.text_input("Cell Number (e.g. +27 82 123 4567)")

#             if st.button("Register"):
#                 if not new_username or not new_password:
#                     st.error("⚠️ Please fill in all required fields.")
#                 elif any(u["username"] == new_username for u in users):
#                     st.error("⚠️ Username already exists.")
#                 else:
#                     if role == "User":
#                         users.append({"username": new_username, "password": new_password})
#                         with open("data/users.json", "w") as f:
#                             json.dump(users, f, indent=2)
#                         st.success("✅ Registration successful! Please login now.")
#                     else:
#                         if not name or not address or not cell:
#                             st.error("⚠️ Please enter your full name, office address, and cell number.")
#                         else:
#                             with st.spinner("🔍 Locating your office..."):
#                                 lat, lon = geocode_address(address)

#                             if lat and lon:
#                                 st.success(f"✅ Location found: {lat:.5f}, {lon:.5f}")

#                                 specialization = "General"
#                                 if specialization not in lawyers_db:
#                                     lawyers_db[specialization] = []

#                                 lawyers_db[specialization].append({
#                                     "name": name,
#                                     "address": address,
#                                     "cell": cell,
#                                     "lat": lat,
#                                     "lon": lon
#                                 })

#                                 with open("data/lawyers.json", "w") as f:
#                                     json.dump(lawyers_db, f, indent=2)

#                                 st.success("🎉 Lawyer registered successfully! Please log in to continue.")
#                             else:
#                                 st.error("❌ Could not find your address on the map. Please recheck it and try again.")

# # --- Chat UI (after login) ---

# if st.session_state.logged_in:
#     st.title(f"💬 Legal Assistant Chat - Logged in as {st.session_state.username}")

#     # Sidebar
#     with st.sidebar:
#         st.header("📂 Chat Sessions")
#         st.subheader("⚙️ Settings")
#         llm_choice = st.selectbox(
#             "Choose AI Model",
#             ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
#         )

#         if st.button("➕ New Session"):
#             new_name = f"Session {len(st.session_state.chat_sessions) + 1}"
#             st.session_state.chat_sessions[new_name] = []
#             st.session_state.current_session = new_name

#         session_names = list(st.session_state.chat_sessions.keys())
#         selected = st.selectbox(
#             "Choose a session", 
#             session_names, 
#             index=session_names.index(st.session_state.current_session)
#         )
#         st.session_state.current_session = selected

#         if st.button("Logout"):
#             st.session_state.logged_in = False
#             st.session_state.username = ""
#             st.session_state.chat_sessions = {"Session 1": []}
#             st.session_state.current_session = "Session 1"
#             st.session_state.show_lawyers = False
#             st.rerun()

#         st.markdown("---")
#         st.subheader("📍 Pro Bono Lawyer Finder")
#         if st.button("Find Pro Bono Lawyers"):
#             st.session_state.show_lawyers = True

#     current_history = st.session_state.chat_sessions[st.session_state.current_session]

#     if current_history:
#         for msg in current_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
#     else:
#         st.info("This session has no messages yet. Start chatting below 👇")

#     LEGAL_KEYWORDS = ["Eviction", "Divorce", "General"]

#     if st.session_state.show_lawyers:
#         st.info("We will use your location to find the nearest pro bono lawyers.")
#         legal_issue = st.selectbox("Select Legal Issue", options=LEGAL_KEYWORDS)

#         location = streamlit_js_eval(
#             js_expressions="""
#             new Promise((resolve, reject) => {
#                 navigator.geolocation.getCurrentPosition(
#                     pos => resolve({latitude: pos.coords.latitude, longitude: pos.coords.longitude}),
#                     err => resolve(null)
#                 )
#             })
#             """,
#             key="lawyer_location",
#             default=None
#         )

#         if location:
#             lat, lng = location["latitude"], location["longitude"]
#             lawyers = get_lawyers(legal_issue)
#             reply = format_lawyers_list(lawyers)
#             current_history.append({"role": "assistant", "content": reply})
#             with st.chat_message("assistant"):
#                 st.markdown(reply)

#             if lawyers:
#                 df = pd.DataFrame(lawyers)
#                 st.map(df)
#         else:
#             st.warning("Could not get your location. Please allow location access in your browser.")

#     # --- Chat input ---
#     user_q = st.chat_input("💬 Ask your legal question...")
#     if user_q:
#         current_history.append({"role": "user", "content": user_q})
#         with st.chat_message("user"):
#             st.markdown(user_q)

#         with st.chat_message("assistant"):
#             with st.spinner("🔍 Classifying your question..."):
#                 intent_result = get_intent(user_q)

#                 if isinstance(intent_result, str):
#                     label = intent_result
#                     explanation = "No explanation provided."
#                 else:
#                     label = intent_result["label"]
#                     explanation = intent_result["explanation"]

#             st.markdown(f"🧠 **Your question was classified as:** `{label}`\n\n_{explanation}_")
#             st.markdown("⏳ Please wait while I gather the most relevant legal information...")

#         st.session_state.current_topic = label
#         current_history.append({"role": "assistant", "content": f"Intent classified as: {label} ({explanation})"})

#         time.sleep(1.5)

#         if label == "GREETING":
#             reply = "Hello! 👋 I'm LegalBot, your South African eviction-law assistant. How can I help you today?"

#         elif label == "OTHER":
#             reply = (
#                 "I'm only trained to assist with **South African eviction and rental law**.\n\n"
#                 "Please ask something related to **eviction notices, rent disputes, lease termination, or unlawful evictions.**"
#             )

#         else:
#             with st.chat_message("assistant"):
#                 with st.spinner("⚖️ Gathering legal references..."):
#                     reply = get_hybrid_response(user_q, current_history)
#                 reply = f"**Category:** {label}\n\n{reply}"
#                 st.markdown(reply)

#         current_history.append({"role": "assistant", "content": reply})
#         st.markdown("---")
#         st.caption("⚠️ Disclaimer: This assistant provides information for educational purposes only. It is not a substitute for professional legal advice.")

import streamlit as st
import random
import json
import pandas as pd
from dotenv import load_dotenv
from rag_pipeline import get_hybrid_response
from case_agent import get_casefile_response
from merge_agent import get_merged_response
from streamlit_js_eval import streamlit_js_eval
from intent import get_intent  # Updated to use the new context-aware version
from answer_lenght import adjust_answer_length
from geocode import geocode_address
import time

# --- Load environment ---
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
        cell_display = f" (📞 {lawyer['cell']})" if 'cell' in lawyer else ""
        message += f"{i}. {lawyer['name']}, {lawyer['address']}{cell_display}\n"
    return message


# --- Page setup ---
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# --- Session state defaults ---
for key, default in [
    ("logged_in", False),
    ("username", ""),
    ("chat_sessions", {"Session 1": []}),
    ("current_session", "Session 1"),
    ("show_lawyers", False),
    ("chat_history", []),
    ("current_topic", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Login & Registration ---
if not st.session_state.logged_in:
    #st.image("images/background.jpg", width=600)
    st.write("\n" * 3)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        option = st.radio("Choose an option", ["Login", "Register"])

        # --- Login section ---
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

        # --- Registration section ---
        else:
            st.subheader("📝 Register New Account")
            role = st.selectbox("I am a...", ["User", "Lawyer"])

            new_username = st.text_input("Choose a Username")
            new_password = st.text_input("Choose a Password", type="password")

            if role == "Lawyer":
                st.markdown("### 🏛️ Lawyer Details")
                name = st.text_input("Full Name")
                address = st.text_input("Office Address (e.g. 54 Groenewound Street, Universitas, Bloemfontein)")
                cell = st.text_input("Cell Number (e.g. +27 82 123 4567)")

            if st.button("Register"):
                if not new_username or not new_password:
                    st.error("⚠️ Please fill in all required fields.")
                elif any(u["username"] == new_username for u in users):
                    st.error("⚠️ Username already exists.")
                else:
                    if role == "User":
                        users.append({"username": new_username, "password": new_password})
                        with open("data/users.json", "w") as f:
                            json.dump(users, f, indent=2)
                        st.success("✅ Registration successful! Please login now.")
                    else:
                        if not name or not address or not cell:
                            st.error("⚠️ Please enter your full name, office address, and cell number.")
                        else:
                            with st.spinner("🔍 Locating your office..."):
                                lat, lon = geocode_address(address)

                            if lat and lon:
                                st.success(f"✅ Location found: {lat:.5f}, {lon:.5f}")

                                specialization = "General"
                                if specialization not in lawyers_db:
                                    lawyers_db[specialization] = []

                                lawyers_db[specialization].append({
                                    "name": name,
                                    "address": address,
                                    "cell": cell,
                                    "lat": lat,
                                    "lon": lon
                                })

                                with open("data/lawyers.json", "w") as f:
                                    json.dump(lawyers_db, f, indent=2)

                                st.success("🎉 Lawyer registered successfully! Please log in to continue.")
                            else:
                                st.error("❌ Could not find your address on the map. Please recheck it and try again.")


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
        selected = st.selectbox(
            "Choose a session",
            session_names,
            index=session_names.index(st.session_state.current_session)
        )
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

    current_history = st.session_state.chat_sessions[st.session_state.current_session]

    # Display previous chat
    if current_history:
        for msg in current_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("This session has no messages yet. Start chatting below 👇")

    LEGAL_KEYWORDS = ["Eviction", "Divorce", "General"]

    # --- Lawyer Finder ---
    if st.session_state.show_lawyers:
        st.info("We will use your location to find the nearest pro bono lawyers.")
        legal_issue = st.selectbox("Select Legal Issue", options=LEGAL_KEYWORDS)

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
            lawyers = get_lawyers(legal_issue)
            reply = format_lawyers_list(lawyers)
            current_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

            if lawyers:
                df = pd.DataFrame(lawyers)
                st.map(df)
        else:
            st.warning("Could not get your location. Please allow location access in your browser.")

    # --- Chat input ---
    user_q = st.chat_input("💬 Ask your legal question...")
    if user_q:
        current_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # 🧠 Intent classification (now with context)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Classifying your question..."):
                intent_result = get_intent(
                    user_q,
                    chat_history=current_history,  # <-- context-aware classification
                    model_choice=llm_choice
                )

                label = intent_result.get("label", "UNKNOWN")
                explanation = intent_result.get("explanation", "Could not determine explanation.")

            st.markdown(f"🧠 **Your question was classified as:** `{label}`\n\n_{explanation}_")

            if label not in ["GREETING", "SMALLTALK", "UNKNOWN"]:
                st.markdown("⏳ Please wait while I gather the most relevant legal information...")

        # Save topic
        st.session_state.current_topic = label
        current_history.append({"role": "assistant", "content": f"Intent classified as: {label} ({explanation})"})

        time.sleep(1)

        # --- Handle responses based on intent ---
        if label == "GREETING":
            reply = "Hello! 👋 I'm LegalBot, your South African eviction-law assistant. How can I help you today?"

        elif label == "UNKNOWN":
            reply = (
                "Hmm, I'm not sure I understand that. "
                "Could you please clarify or ask a specific question about your housing or eviction issue?"
            )

        else:
            with st.chat_message("assistant"):
                with st.spinner("⚖️ Gathering legal references..."):
                    reply = get_hybrid_response(user_q, current_history)
                reply = f"**Category:** {label}\n\n{reply}"
                st.markdown(reply)

        current_history.append({"role": "assistant", "content": reply})
        st.markdown("---")
        st.caption("⚠️ Disclaimer: This assistant provides information for educational purposes only. It is not a substitute for professional legal advice.")
