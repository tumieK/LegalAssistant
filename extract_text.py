#Script for loading,extracting and storing in ChromaDB
import os
from PyPDF2 import PdfReader
from chromadb import Client
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Setup ChromaDB with persistence
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("legal_docs")

# Load embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Folder with your files
folder_path = "./legal_docs"

# Loop through all files
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    if file_name.endswith(".pdf"):
        print(f" Processing PDF: {file_name}")
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
    elif file_name.endswith(".txt"):
        print(f" Processing TXT: {file_name}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
    else:
        print(f" Skipping unsupported file: {file_name}")
        continue
    
    # Embed and add to ChromaDB
    vector = embedder.embed_query(text)
    collection.add(documents=[text], embeddings=[vector], ids=[file_name])
    print(f" Stored: {file_name}")

# Save ChromaDB to disk
print(" All documents processed and saved.")
