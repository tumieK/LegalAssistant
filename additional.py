import requests
import os
import certifi
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# --- Load OpenAI key ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in the environment.")

# --- Fetch case text ---


def fetch_case_text(url):
    r = requests.get(url, verify=False)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")
    return "\n".join([p.get_text() for p in paragraphs])

url = "https://www.saflii.org/"
case_text = fetch_case_text(url)

# --- Setup LLM ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

# --- Prompt and Chain ---
template = """
Summarize the following case text. Extract important legal definitions and explain them:
{text}
"""
prompt = PromptTemplate(input_variables=["text"], template=template)
chain = prompt | llm
summary = chain.invoke({"text": case_text})

print(summary)

