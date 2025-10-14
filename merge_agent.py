# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# import os

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# def get_merged_response(user_q, general_reply, casefile_reply, model_name="gpt-4o-mini"):
#     """
#     Merge responses from general legal docs and casefiles into one coherent answer.
#     """
#     llm = ChatOpenAI(
#         openai_api_key=OPENAI_API_KEY,
#         model_name=model_name,
#         temperature=0.0
#     )

#     prompt = ChatPromptTemplate.from_template("""
#     You are a legal assistant tasked with merging two sources of information.

#     User question:
#     {user_q}

#     General legal docs response:
#     {general_reply}

#     Casefiles response:
#     {casefile_reply}

#     Task:
#     - Compare both responses.
#     - Merge them into a single coherent, well-structured answer.
#     - Ensure correctness, avoid contradictions, and highlight if case law supports or clarifies the general rules.
#     - Keep it educational (not legal advice).

#     Final answer:
#     """)

#     chain = prompt | llm
#     resp = chain.invoke({
#         "user_q": user_q,
#         "general_reply": general_reply,
#         "casefile_reply": casefile_reply
#     })
#     return resp.content.strip()

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def extract_sources(text: str) -> str:
    """
    Extracts a 'Sources' section from a response, if it exists.
    Returns it cleanly formatted (without duplication).
    """
    match = re.search(r"(\*\*Sources?:\*\*.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def get_merged_response(user_q, general_reply, casefile_reply, model_name="gpt-4o-mini"):
    """
    Merge responses from general legal docs (RAG) and casefiles into one coherent answer.
    Preserve any source information from the RAG response or add fallback if missing.
    """
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=0.0
    )

    # extract any sources section from the general reply before merging
    sources_section = extract_sources(general_reply)
    general_reply_clean = re.sub(r"(\*\*Sources?:\*\*.*)", "", general_reply, flags=re.IGNORECASE | re.DOTALL).strip()

    prompt = ChatPromptTemplate.from_template("""
    You are a South African legal assistant tasked with merging two sources of information.

    User question:
    {user_q}

    General legal documents response (from your knowledge base):
    {general_reply}

    Casefiles response (context from prior cases or legal precedents):
    {casefile_reply}

    Your task:
    - Merge both responses into one clear, educational, and professional answer.
    - Avoid repeating identical points.
    - If case law clarifies or modifies a general rule, mention that naturally.
    - Write in plain, accurate South African legal English.
    - Do not use uncertain language like "maybe" or "might".
    - Do NOT remove any citations or factual details.
    - Do not include "Sources" in the middle of the answer — we’ll handle that separately.

    Final merged answer:
    """)

    chain = prompt | llm
    resp = chain.invoke({
        "user_q": user_q,
        "general_reply": general_reply_clean,
        "casefile_reply": casefile_reply
    })

    merged_answer = resp.content.strip()

    # --- Preserve or add a Sources section ---
    if sources_section:
        merged_answer += f"\n\n{sources_section}"
    else:
        merged_answer += "\n\n**Source:** Based on knowledge from LegalBot's training and understanding of South African law."

    return merged_answer
