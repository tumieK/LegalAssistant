# import streamlit as st
# import random
# import json
# from dotenv import load_dotenv
# from rag_pipeline import get_hybrid_response  
# from case_agent import get_casefile_response  
# from merge_agent import get_merged_response  

# # For browser geolocation
# from streamlit_js_eval import streamlit_js_eval

# load_dotenv()

# # --- Page setup ---
# st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

# # --- Session state defaults ---
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "username" not in st.session_state:
#     st.session_state.username = ""
# if "chat_sessions" not in st.session_state:  # multiple sessions
#     st.session_state.chat_sessions = {"Session 1": []}
# if "current_session" not in st.session_state:
#     st.session_state.current_session = "Session 1"
# if "show_lawyers" not in st.session_state:
#     st.session_state.show_lawyers = False

# # --- Full-page login with Streamlit image ---
# if not st.session_state.logged_in:
#     st.image("images/background.jpg", use_container_width=False)
#     st.write("\n" * 5)
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.subheader("🔑 Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if username and password:
#                 st.session_state.logged_in = True
#                 st.session_state.username = username
#                 st.rerun()
#             else:
#                 st.error("Enter username and password.")

# # --- Chat UI (after login) ---
# if st.session_state.logged_in:
#     st.title(f"💬 Legal Assistant Chat - Logged in as {st.session_state.username}")

#     # Sidebar: Sessions + Pro Bono Finder
#     with st.sidebar:
#         st.header("📂 Chat Sessions")

#         # Create new session
#         if st.button("➕ New Session"):
#             new_name = f"Session {len(st.session_state.chat_sessions) + 1}"
#             st.session_state.chat_sessions[new_name] = []
#             st.session_state.current_session = new_name

#         # Select session
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

#     # Current session history
#     current_history = st.session_state.chat_sessions[st.session_state.current_session]

#     # --- Display chat history (main section only) ---
#     if current_history:
#         for msg in current_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
#     else:
#         st.info("This session has no messages yet. Start chatting below 👇")

#     # --- Helper Functions for Pro Bono Locator ---
#     LEGAL_KEYWORDS = {
#         "Eviction": "tenant lawyer, housing lawyer, eviction lawyer",
#         "Maintenance": "child support lawyer, family law lawyer",
#         "Divorce": "divorce lawyer, family law lawyer",
#         "General": "pro bono lawyer"
#     }

#     def generate_dummy_lawyers(lat: float, lng: float, legal_issue: str = "General"):
#         """Generate fake nearby lawyers without using Google Maps API."""
#         sample_names = {
#             "Eviction": ["Tenant Rights SA", "Housing Justice Center", "Eviction Aid Clinic"],
#             "Maintenance": ["Family Care Legal Aid", "Child Support SA", "Custody Help Desk"],
#             "Divorce": ["Divorce Mediation SA", "Family Law Pro Bono", "Separation Support Lawyers"],
#             "General": ["Community Legal Aid", "Justice for All Foundation", "Free Legal Support SA"]
#         }
#         names = sample_names.get(legal_issue, sample_names["General"])

#         lawyers = []
#         for i in range(3):  # 3 random results
#             lawyers.append({
#                 "name": random.choice(names),
#                 "address": f"Street {random.randint(1,99)}, Local Area",
#                 "lat": lat + random.uniform(-0.01, 0.01),
#                 "lng": lng + random.uniform(-0.01, 0.01)
#             })
#         return lawyers

#     def format_lawyers_list(lawyers):
#         if not lawyers:
#             return "Sorry, no pro bono lawyers found nearby for your case."
#         message = "Here are some pro bono lawyers near you:\n\n"
#         for i, lawyer in enumerate(lawyers, 1):
#             message += f"{i}. {lawyer['name']}, {lawyer['address']}\n"
#         return message

#     # --- Pro Bono Finder Section ---
#     if st.session_state.show_lawyers:
#         st.info("We will use your location to find the nearest pro bono lawyers.")
#         legal_issue = st.selectbox("Select Legal Issue", options=list(LEGAL_KEYWORDS.keys()))

#         # Get user location from browser
#         location = streamlit_js_eval(
#             js_expressions="navigator.geolocation.getCurrentPosition(pos => pos.coords)",
#             key="lawyer_location",
#             default=None
#         )

#         if location:
#             lat, lng = location["latitude"], location["longitude"]

#             # Generate dummy lawyers
#             lawyers = generate_dummy_lawyers(lat, lng, legal_issue)
#             reply = format_lawyers_list(lawyers)

#             current_history.append({"role": "assistant", "content": reply})
#             with st.chat_message("assistant"):
#                 st.markdown(reply)

#             if lawyers:
#                 st.map([[l["lat"], l["lng"]] for l in lawyers])
#         else:
#             st.warning("Could not get your location. Please allow location access in your browser.")

#     # --- Chat input ---
#     user_q = st.chat_input("Ask a legal question...")
#     if user_q:
#         current_history.append({"role": "user", "content": user_q})
#         with st.chat_message("user"):
#             st.markdown(user_q)

#         # Normal AI responses
#         general_reply = get_hybrid_response(user_q, current_history)
#         casefile_reply = get_casefile_response(user_q, current_history)
#         reply = get_merged_response(user_q, general_reply, casefile_reply)

#         current_history.append({"role": "assistant", "content": reply})
#         with st.chat_message("assistant"):
#             st.markdown(reply)

#     # --- Footer disclaimer ---
#     st.markdown("---")
#     st.caption(
#         "⚠️ Disclaimer: This assistant provides information for educational purposes only. "
#         "It is not a substitute for professional legal advice."
#     )

import streamlit as st
import random
import json
from dotenv import load_dotenv
from rag_pipeline import get_hybrid_response  
from case_agent import get_casefile_response  
from merge_agent import get_merged_response  

# For browser geolocation
from streamlit_js_eval import streamlit_js_eval

load_dotenv()

# --- Load JSON files ---
with open("data/users.json", "r") as f:
    users = json.load(f)

with open("data/lawyers.json", "r") as f:
    lawyers_db = json.load(f)

# --- Helper functions ---
def check_login(username, password):
    """Check if username/password match JSON file"""
    for u in users:
        if u["username"] == username and u["password"] == password:
            return True
    return False

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
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Session 1": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Session 1"
if "show_lawyers" not in st.session_state:
    st.session_state.show_lawyers = False

# --- Full-page login ---
if not st.session_state.logged_in:
    st.image("images/background.jpg", use_container_width=False)
    st.write("\n" * 5)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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

# --- Chat UI (after login) ---
if st.session_state.logged_in:
    st.title(f"💬 Legal Assistant Chat - Logged in as {st.session_state.username}")

    # Sidebar
    with st.sidebar:
        st.header("📂 Chat Sessions")

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

    # Current session history
    current_history = st.session_state.chat_sessions[st.session_state.current_session]

    # --- Display chat history ---
    if current_history:
        for msg in current_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        st.info("This session has no messages yet. Start chatting below 👇")

    # --- Pro Bono Finder Section ---
    LEGAL_KEYWORDS = ["Eviction", "Maintenance", "Divorce", "General"]

    if st.session_state.show_lawyers:
        st.info("We will use your location to find the nearest pro bono lawyers.")
        legal_issue = st.selectbox("Select Legal Issue", options=LEGAL_KEYWORDS)

        location = streamlit_js_eval(
            js_expressions="navigator.geolocation.getCurrentPosition(pos => pos.coords)",
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

            if lawyers:
                st.map([[l["lat"], l["lng"]] for l in lawyers])
        else:
            st.warning("Could not get your location. Please allow location access in your browser.")

    # --- Chat input ---
    user_q = st.chat_input("Ask a legal question...")
    if user_q:
        current_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # AI responses
        general_reply = get_hybrid_response(user_q, current_history)
        casefile_reply = get_casefile_response(user_q, current_history)
        reply = get_merged_response(user_q, general_reply, casefile_reply)

        current_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # --- Footer disclaimer ---
    st.markdown("---")
    st.caption(
        "⚠️ Disclaimer: This assistant provides information for educational purposes only. "
        "It is not a substitute for professional legal advice."
    )
