from mcp.server.fastmcp import FastMCP
from typing import List, Optional
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

# Import your PDF extractor class
# You'll need to make sure pdf_extractor.py is in the same directory
from pdf_extractor import PDFExtractor

mcp = FastMCP(
    name="combined_document_processor"
)

# Initialize PDF extractor
extractor = PDFExtractor()

def get_api_key():
    """Get OpenAI API key from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith('OPENAI_API_KEY='):
                    return line.strip().split('=')[1]
    return None

@mcp.tool()
def extract_pdf_contents(pdf_path: str, pages: Optional[str] = None) -> str:
    """
    Extracts text from a PDF file.
    Args:
        pdf_path: Path to the PDF file.
        pages: Comma-separated page numbers (optional).
    Returns:
        Extracted text as a string.
    """
    return extractor.extract_content(pdf_path, pages)

@mcp.tool()
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Splits the input text into chunks of approximately chunk_size characters.
    Args:
        text: The input text to chunk.
        chunk_size: The maximum size of each chunk (default: 500).
    Returns:
        List of text chunks (as strings).
    """
    return [str(text[i:i+chunk_size]) for i in range(0, len(text), chunk_size)]

@mcp.tool()
def embed_chunks(text_chunks: List[str]) -> List[str]:
    """
    Generates vector embeddings for a list of text chunks using OpenAI embeddings.
    Args:
        text_chunks: List of text chunks.
    Returns:
        List of embedding vectors as JSON strings (one per chunk).
    """
    try:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("Could not find OPENAI_API_KEY in .env file")
        embedder = OpenAIEmbeddings(api_key=api_key)
        vectors = embedder.embed_documents(text_chunks)
        return [json.dumps(vec) for vec in vectors]
    except Exception as e:
        return [f"Error: {str(e)}"]

@mcp.tool()
def process_pdf_to_embeddings(pdf_path: str, chunk_size: int = 500, pages: Optional[str] = None) -> dict:
    """
    Complete pipeline: Extract PDF content, chunk it, and generate embeddings.
    Args:
        pdf_path: Path to the PDF file.
        chunk_size: The maximum size of each chunk (default: 500).
        pages: Comma-separated page numbers (optional).
    Returns:
        Dictionary containing chunks and their embeddings.
    """
    try:
        # Step 1: Extract PDF content
        text = extract_pdf_contents(pdf_path, pages)
        
        # Step 2: Chunk the text
        chunks = chunk_text(text, chunk_size)
        
        # Step 3: Generate embeddings
        embeddings = embed_chunks(chunks)
        
        return {
            "original_text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.resource("pdf://status")
def pdf_status_resource() -> str:
    """Get status of PDF processing capabilities"""
    return "PDF extraction, chunking, and embedding services are active"

if __name__ == "__main__":
    mcp.run(transport="stdio")
