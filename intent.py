# from openai import OpenAI

# client = OpenAI()

# def get_intent(user_query: str, model_choice: str = "gpt-4o-mini") -> dict:
#     """
#     Classify the user's question into one of the 10 legal eviction-related categories or other general intents.
#     Returns a dictionary with label and explanation.
#     """
#     categories = """
# 1. Eviction Notices and Procedures
# 2. Rent Disputes and Non-Payment
# 3. Lease Agreements and Termination
# 4. Illegal or Unlawful Evictions
# 5. Eviction and Vulnerable Tenants
# 6. Social Housing and Government Property
# 7. Court and Legal Procedures
# 8. Remedies and Assistance
# 9. Landlord Misconduct or Harassment
# 10. Eviction After Property Sale or Inheritance
# Other categories:
# - GREETING (greetings or pleasantries)
# - FIND_LAWYERS (user asks for legal help or contact)
# - SMALLTALK (casual conversation)
# - UNKNOWN (unclear or unrelated to law)
#     """

#     system_prompt = f"""
# You are an intent classifier for a South African legal chatbot that helps users with eviction and housing issues.
# Classify the user’s question into ONE of the following categories and briefly explain why.
# Return the result in JSON format with two keys:
# - "label": the category name exactly as written
# - "explanation": a one-sentence reason for the classification

# Categories:
# {categories}
#     """

#     response = client.chat.completions.create(
#         model=model_choice,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_query}
#         ]
#     )

#     import json
#     try:
#         content = response.choices[0].message.content.strip()
#         result = json.loads(content)
#     except Exception:
#         # Fallback in case the model doesn't output perfect JSON
#         result = {"label": "UNKNOWN", "explanation": "Could not parse classification output."}

#     return result
# --- After importing and getting intent ---

# from openai import OpenAI
# import json

# client = OpenAI()

# def get_intent(user_query: str, model_choice: str = "gpt-4o-mini") -> dict:
#     """
#     Classify the user's question into one of the 10 legal eviction-related categories or other general intents.
#     Returns a dictionary with label and explanation.
#     """
#     categories = """
# 1. Eviction Notices and Procedures
# 2. Rent Disputes and Non-Payment
# 3. Lease Agreements and Termination
# 4. Illegal or Unlawful Evictions
# 5. Eviction and Vulnerable Tenants
# 6. Social Housing and Government Property
# 7. Court and Legal Procedures
# 8. Remedies and Assistance
# 9. Landlord Misconduct or Harassment
# 10. Eviction After Property Sale or Inheritance
# Other categories:
# - GREETING (greetings or pleasantries)
# - FIND_LAWYERS (user asks for legal help or contact)
# - SMALLTALK (casual conversation)
# - UNKNOWN (unclear or unrelated to law)
#     """

#     system_prompt = f"""
# You are an intent classifier for a South African legal chatbot that helps users with eviction and housing issues.
# Classify the user’s question into ONE of the following categories and briefly explain why.
# Return the result in JSON format with two keys:
# - "label": the category name exactly as written
# - "explanation": a one-sentence reason for the classification

# Categories:
# {categories}
#     """

#     response = client.chat.completions.create(
#         model=model_choice,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_query}
#         ]
#     )

#     try:
#         content = response.choices[0].message.content.strip()
#         result = json.loads(content)
#     except Exception:
#         # Fallback if output isn't valid JSON
#         result = {"label": "UNKNOWN", "explanation": "Could not parse classification output."}

#     return result

# intent.py
import streamlit as st
from openai import OpenAI
import json

# --- Initialize OpenAI client ---
client = OpenAI()

# --- Intent classification with chat context ---
def get_intent(user_query: str, chat_history: list = None, model_choice: str = "gpt-4o-mini") -> dict:
    """
    Classify the user's question into one of the 10 legal eviction-related categories or other general intents.
    Takes previous chat history for context awareness.
    """
    if chat_history:
        context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])  # last 3 messages
    else:
        context = "No previous messages."

    categories = """
1. Eviction Notices and Procedures
2. Rent Disputes and Non-Payment
3. Lease Agreements and Termination
4. Illegal or Unlawful Evictions
5. Eviction and Vulnerable Tenants
6. Social Housing and Government Property
7. Court and Legal Procedures
8. Remedies and Assistance
9. Landlord Misconduct or Harassment
10. Eviction After Property Sale or Inheritance
Other categories:
- GREETING (greetings or pleasantries)
- FIND_LAWYERS (user asks for legal help or contact)
- SMALLTALK (casual conversation)
- UNKNOWN (unclear or unrelated to law)
    """

    system_prompt = f"""
You are an intent classifier for a South African legal chatbot about eviction and housing issues.
Classify the **user’s latest question** based on both its content and the recent chat context.

If the user is clearly asking about or following up on a previous eviction-related question,
infer the most likely related category from context (even if the question is short like "please explain" or "what do you mean?").

Return JSON with:
- "label": one category name exactly as written
- "explanation": one-sentence justification for the classification

Categories:
{categories}

Recent context:
{context}
    """

    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    try:
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
    except Exception:
        result = {"label": "UNKNOWN", "explanation": "Could not parse classification output."}

    return result


# --- Generate chatbot response ---
def generate_response(user_query: str, category: str) -> str:
    """
    Generate a legal or general chatbot response based on the classified intent.
    """
    if category in ["GREETING"]:
        return "Hello there 👋! How can I assist you with your housing or eviction questions today?"

    elif category in ["SMALLTALK"]:
        return "I'm always here to chat — but my main goal is to help you with housing or eviction-related issues 😊."

    elif category == "FIND_LAWYERS":
        return (
            "You can contact your nearest **Rental Housing Tribunal** or **Legal Aid South Africa** for free legal advice. "
            "Would you like me to list offices near your province?"
        )

    elif category == "UNKNOWN":
        return (
            "I'm still learning and don’t have enough context about your question. "
            "Could you please rephrase or give a bit more detail about your housing or eviction issue?"
        )

    # --- Default legal response ---
    system_prompt = f"""
You are LegalBot, a helpful South African AI legal assistant specializing in **eviction and housing law**.
Provide a clear and legally sound answer in plain English (no legalese) about **{category}**.
If applicable, list key rights, procedures, and next steps based on South African law.
End with a short actionable summary.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return response.choices[0].message.content.strip()


# --- Streamlit App UI ---
st.set_page_config(page_title="LegalBot SA 🇿🇦", page_icon="⚖️", layout="wide")
st.title("⚖️ LegalBot South Africa")
st.caption("Your AI-powered legal assistant for eviction and housing issues in South Africa 🇿🇦")

# --- Initialize session state ---
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"default": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "default"

# --- Chat input ---
user_q = st.chat_input("Ask your legal question here...")
if user_q:
    # Save user message
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_q})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_q)

    # --- Assistant classification and reply ---
    with st.chat_message("assistant"):
        with st.spinner("🔍 Classifying your question..."):
            intent_result = get_intent(
                user_q,
                st.session_state.chat_sessions[st.session_state.current_session]
            )

            label = intent_result.get("label", "UNKNOWN")
            explanation = intent_result.get("explanation", "Could not determine explanation.")

        st.markdown(f"🧠 **Your question was classified as:** `{label}`\n\n_{explanation}_")

        if label not in ["GREETING", "SMALLTALK", "UNKNOWN"]:
            st.markdown("⏳ Please wait while I gather the most relevant legal information...")

        # Generate and show chatbot response
        answer = generate_response(user_q, label)
        st.markdown(answer)

        # Save assistant message
        st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": answer})

# --- Display previous messages ---
for message in st.session_state.chat_sessions[st.session_state.current_session]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Optional: Show recent context (debugging only) ---
if st.checkbox("🔍 Show classification context (debug only)"):
    ctx = st.session_state.chat_sessions[st.session_state.current_session][-3:]
    st.code("\n".join([f"{m['role']}: {m['content']}" for m in ctx]))
