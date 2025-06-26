import os
import re
from typing import List
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
from openai import OpenAI

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

# Helper: sanitize doc_id for ChromaDB
def sanitize_doc_id(pdf_path: str) -> str:
    base = os.path.basename(pdf_path)
    doc_id = os.path.splitext(base)[0]
    return re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)

# Main QnA function
def answer_question(pdf_path: str, question: str, chunk_size: int = 500, top_k: int = 5) -> str:
    # 1. Extract and chunk text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size)
    doc_id = sanitize_doc_id(pdf_path)

    # 2. Set up ChromaDB (local persistent)
    client = chromadb.Client(Settings(persist_directory="data/chroma"))
    if doc_id not in [c.name for c in client.list_collections()]:
        collection = client.create_collection(doc_id)
        # 3. Embed and store chunks
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_client = OpenAI(api_key=openai_api_key)
        embeddings = [openai_client.embeddings.create(input=[chunk], model="text-embedding-3-small").data[0].embedding for chunk in chunks]
        collection.add(documents=chunks, embeddings=embeddings, ids=[f"{doc_id}_{i}" for i in range(len(chunks))])
    else:
        collection = client.get_collection(doc_id)

    # 4. Embed question
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key)
    q_embedding = openai_client.embeddings.create(input=[question], model="text-embedding-3-small").data[0].embedding

    # 5. Retrieve top_k relevant chunks
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    context = "\n".join(results["documents"][0])

    # 6. Use LLM to answer
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip() 