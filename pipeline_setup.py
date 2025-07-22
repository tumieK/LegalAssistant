from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

import transformers
import torch
import os
import logging
from typing import Optional

def load_rag_chain(hf_token: Optional[str] = None, persist_dir: str = "./chroma_db") -> ConversationalRetrievalChain:
    """
    Initialize and return a ConversationalRetrievalChain for RAG using local Transformers pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Use local Hugging Face token for gated model access
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        elif not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
            raise ValueError("HUGGINGFACE_HUB_TOKEN must be provided.")

        # Initialize sentence-transformers embeddings
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

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
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )

        pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.3,
            device_map="auto"
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

if __name__ == "__main__":
    try:
        rag_chain = load_rag_chain(hf_token="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")
        result = rag_chain.invoke({"question": "What is a civil summons?", "chat_history": []})
        print(result["answer"])
    except Exception as e:
        print(f"Error: {str(e)}")
