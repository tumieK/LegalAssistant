from openai import OpenAI

client = OpenAI()

def adjust_answer_length(user_q: str, raw_answer: str, model_choice: str = "gpt-4o-mini") -> str:
    """
    Adjusts the verbosity of an answer depending on the type of user question.
    - Short questions (facts, numbers, deadlines, yes/no) → concise reply
    - Broader/open-ended questions → detailed explanation
    """

    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": """You are a legal assistant. 
Decide how to present the answer depending on the question:
- If the question asks for a specific timeframe, number, or fact → respond concisely (1–3 sentences).
- If the question is open-ended, process-oriented, or explanatory → respond with a detailed explanation (4+ sentences, context, and reasoning).
Do NOT remove legal accuracy. Just adjust the verbosity."""},
            {"role": "user", "content": f"User question: {user_q}\n\nDraft answer: {raw_answer}"}
        ]
    )

    return response.choices[0].message.content.strip()
