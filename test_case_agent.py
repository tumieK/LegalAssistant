from case_agent import get_casefile_response

if __name__ == "__main__":
    question = input("Question: ").strip()
    print("\n--- Answer ---\n")
    print(get_casefile_response(question, chat_history=[]))
