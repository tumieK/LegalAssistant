from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

import transformers
import torch
import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def load_rag_chain(hf_token: Optional[str] = None, persist_dir: str = "./chroma_db") -> ConversationalRetrievalChain:
    """
    Initialize and return a ConversationalRetrievalChain for RAG using local Transformers pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Use local Hugging Face token for gated model access
        if HF_TOKEN:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
        elif not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            raise ValueError("HUGGINGFACE_HUB_TOKEN must be provided.")

        # Initialize sentence-transformers embeddings
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load vector store
        logger.info("Loading Chroma vector store...")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Load your local Transformers pipeline
        logger.info("Initializing local Transformers pipeline...")

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            token=HF_TOKEN
        )

        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
           # device_map="auto"
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # Create RAG chain
        logger.info("Creating ConversationalRetrievalChain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            verbose=False,
        )

        logger.info("RAG chain successfully initialized")
        return qa_chain

    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise

#Testing  function!
if __name__ == "__main__":
    try:
        HF_TOKEN = HF_TOKEN
        rag_chain = load_rag_chain(hf_token=HF_TOKEN)

        # Test the chain with a question
        chat_history = []
        question = "What is the purpose of legal aid in eviction cases?"

        result = rag_chain.invoke({"question": question, "chat_history": chat_history})

        print("\nAnswer:")
        print(result["answer"])

        print("\nSource Documents:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'No source')}")
    except Exception as err:
        print(f"Error running RAG chain: {err}")
