from typing import List, Optional
import chromadb
from chromadb.config import Settings
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
import json

class VectorStore:
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))

    def store_document(self, doc_id: str, chunks: List[str], embeddings: List[List[float]], metadata: Optional[dict] = None) -> None:
        """
        Store document chunks and their embeddings in ChromaDB.
        Args:
            doc_id: Unique identifier for the document
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Optional metadata about the document
        """
        # Create or get collection for the document
        collection = self.client.get_or_create_collection(name=doc_id)
        
        # Add chunks and embeddings
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[metadata or {}] * len(chunks)
        )

    def query_similar(self, doc_id: str, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Query similar chunks from a document using embedding similarity.
        Args:
            doc_id: Document identifier
            query_embedding: Query embedding vector
            top_k: Number of similar chunks to return
        Returns:
            List of similar text chunks
        """
        collection = self.client.get_collection(name=doc_id)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0]  # First list since we only have one query

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks from the store."""
        self.client.delete_collection(name=doc_id)

    def list_documents(self) -> List[str]:
        """List all document IDs in the store."""
        return [collection.name for collection in self.client.list_collections()] 

mcp = FastMCP("vector_store")
store = VectorStore()

@mcp.tool()
def add_document(doc_id: str, chunks: List[str], embeddings: List[str], metadata: Optional[dict] = None) -> str:
    # embeddings are JSON strings, decode them
    decoded_embeddings = [json.loads(e) if isinstance(e, str) else e for e in embeddings]
    store.store_document(doc_id, chunks, decoded_embeddings, metadata)
    return f"Document '{doc_id}' stored."

@mcp.tool()
def search_embeddings(doc_id: str, query_embedding: str, top_k: int = 5) -> List[str]:
    # query_embedding is a JSON string, decode it
    embedding = json.loads(query_embedding) if isinstance(query_embedding, str) else query_embedding
    return store.query_similar(doc_id, embedding, top_k)

@mcp.tool()
def delete_document(doc_id: str) -> str:
    store.delete_document(doc_id)
    return f"Document '{doc_id}' deleted."

@mcp.tool()
def list_documents() -> List[str]:
    return store.list_documents()

if __name__ == "__main__":
    mcp.run(transport="stdio") 