import os
import re
from typing import List
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Helper: sanitize doc_id for ChromaDB
def sanitize_doc_id(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    doc_id = os.path.splitext(base)[0]
    return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

# Helper: extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Helper: chunk text
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ChromaDB client
def get_chroma_client():
    return chromadb.Client(Settings(persist_directory="data/chroma"))

# Ingest a PDF into the vector DB
def ingest_pdf(pdf_path: str, chunk_size: int = 500) -> str:
    doc_id = sanitize_doc_id(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size)
    client = get_chroma_client()
    if doc_id in [c.name for c in client.list_collections()]:
        return f"Document '{doc_id}' already ingested."
    collection = client.create_collection(doc_id)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    embeddings = [openai_client.embeddings.create(input=[chunk], model="text-embedding-3-small").data[0].embedding for chunk in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=[f"{doc_id}_{i}" for i in range(len(chunks))])
    return f"Ingested '{pdf_path}' as '{doc_id}' with {len(chunks)} chunks."

# List all documents in the vector DB
def list_docs() -> List[str]:
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]

# Delete a document from the vector DB
def delete_doc(doc_id: str) -> str:
    client = get_chroma_client()
    if doc_id not in [c.name for c in client.list_collections()]:
        return f"Document '{doc_id}' not found."
    client.delete_collection(doc_id)
    return f"Deleted document '{doc_id}'."

# Search for similar text in the vector DB
def search_text(doc_id: str, query: str, top_k: int = 5) -> List[str]:
    client = get_chroma_client()
    if doc_id not in [c.name for c in client.list_collections()]:
        return [f"Document '{doc_id}' not found."]
    collection = client.get_collection(doc_id)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    q_embedding = openai_client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else ["No results found."] 