from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def get_merged_response(user_q, general_reply, casefile_reply):
    """
    Merge responses from general legal docs and casefiles into one coherent answer.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a legal assistant tasked with merging two sources of information.

    User question:
    {user_q}

    General legal docs response:
    {general_reply}

    Casefiles response:
    {casefile_reply}

    Task:
    - Compare both responses.
    - Merge them into a single coherent, well-structured answer.
    - Ensure correctness, avoid contradictions, and highlight if case law supports or clarifies the general rules.
    - Keep it educational (not legal advice).

    Final answer:
    """)

    chain = prompt | llm
    resp = chain.invoke({
        "user_q": user_q,
        "general_reply": general_reply,
        "casefile_reply": casefile_reply
    })
    return resp.content.strip()
