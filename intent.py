from openai import OpenAI

client = OpenAI()

def get_intent(user_query: str, model_choice: str = "gpt-4o-mini") -> str:
    """
    Classify user intent with the new OpenAI SDK.
    """
    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": "You are an intent classifier for a legal assistant. "
                                          "Return only one of: LEGAL_QUESTION, FIND_LAWYERS, GREETING, SMALLTALK, UNKNOWN."},
            {"role": "user", "content": user_query}
        ]
    )

    intent = response.choices[0].message.content.strip().upper()

    valid_intents = {"LEGAL_QUESTION", "FIND_LAWYERS", "GREETING", "SMALLTALK"}
    if intent not in valid_intents:
        return "UNKNOWN"
    return intent
