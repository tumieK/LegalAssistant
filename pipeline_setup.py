# Settinng up RAG Pipeline with LangChain and Zephyr
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
import os

def load_rag_chain():
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_abc123456yourrealapitoken"  # your token

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    retriever = vectordb.as_retriever()

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",  # Free model
        task="conversational",
        temperature=0.3,
        max_new_tokens=1024,
    )


    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

if __name__ == "__main__":
    chain = load_rag_chain()
    chat_history = []
    question = "What does the South African constitution say about freedom of expression?"
    result = chain.invoke({"question": question, "chat_history": chat_history})
    print("Answer:\n", result["answer"])
