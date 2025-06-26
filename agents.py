from typing import List, Any, Optional
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import time
from server.vector_store import VectorStore
from server.tracing import get_tracer
from server.llm_monitoring import LLMMonitor
import re
import os
import nest_asyncio

nest_asyncio.apply()

def run_async(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # If there's a running loop, use create_task and gather
        return asyncio.get_event_loop().run_until_complete(coro)
    else:
        return asyncio.run(coro)

class DocumentAgents:
    """
    Holds LLM and agent configuration. Each method returns a LangChain Tool for a document understanding step.
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo-preview")

    def pdf_extractor_tool(self) -> Tool:
        """LangChain Tool for PDF extraction via MCP stdio tool."""
        def extract(pdf_path: str) -> str:
            async def extract_async():
                server_params = StdioServerParameters(
                    command="python",
                    args=["server/pdf_processing_server.py"],
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "extract_pdf_contents",
                            arguments={"pdf_path": pdf_path}
                        )
                        return result.content[0].text
            return run_async(extract_async())
        return Tool(
            name="PDF Extractor",
            description="Extracts text from a PDF file using the MCP pdfextractor tool.",
            func=extract
        )

    def chunker_tool(self) -> Tool:
        """LangChain Tool for chunking via MCP stdio tool."""
        def chunk(text: str, chunk_size: int = 500) -> List[str]:
            async def chunk_async():
                server_params = StdioServerParameters(
                    command="python",
                    args=["server/pdf_processing_server.py"],
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "chunk_text",
                            arguments={"text": text, "chunk_size": chunk_size}
                        )
                        return [c.text for c in result.content]
            return run_async(chunk_async())
        return Tool(
            name="Chunker",
            description="Splits text into chunks using the MCP chunker tool.",
            func=chunk
        )

    def embedder_tool(self) -> Tool:
        """LangChain Tool for embedding via MCP stdio tool."""
        def embed(input_data: Any) -> List[List[float]]:
            chunks = input_data["text_chunks"] if isinstance(input_data, dict) else input_data
            async def embed_async():
                server_params = StdioServerParameters(
                    command="python",
                    args=["server/pdf_processing_server.py"],
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "embed_chunks",
                            arguments={"text_chunks": chunks}
                        )
                        if result.content and result.content[0].text.startswith("Error"):
                            raise ValueError(result.content[0].text)
                        return [json.loads(c.text) for c in result.content]
            return run_async(embed_async())
        return Tool(
            name="Embedder",
            description="Embeds text chunks using the MCP embedder tool.",
            func=embed
        )

    def summarizer_tool(self) -> Tool:
        """LangChain Tool for summarization via MCP stdio tool."""
        def summarize(text_or_chunks: Any) -> str:
            async def summarize_async():
                server_params = StdioServerParameters(
                    command="python",
                    args=["server/summarizer_qna_server.py"],
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(
                            "summarize_text",
                            arguments={"text": text_or_chunks}
                        )
                        return result.content[0].text
            return run_async(summarize_async())
        return Tool(
            name="Summarizer",
            description="Summarizes text or chunks using the MCP summarizer tool.",
            func=summarize
        )

    def qna_tool(self, question: str, doc_id: str, top_k: int = 5) -> str:
        async def qna_async():
            server_params = StdioServerParameters(
                command="python",
                args=["server/summarizer_qna_server.py"],
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "answer_question",
                        arguments={"question": question, "doc_id": doc_id, "top_k": top_k}
                    )
                    return result.content[0].text
        return run_async(qna_async())

class DocumentProcessingPipeline:
    """
    Orchestrates the document processing workflow using LangChain tools.
    Provides methods to get a summary and answer questions about the document.
    """
    def __init__(self, pdf_path: str, chunk_size: int = 500):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.text: Optional[str] = None
        self.chunks: Optional[List[str]] = None
        self.embeddings: Optional[List[List[float]]] = None
        # Sanitize doc_id for ChromaDB: use file name, replace spaces and invalid chars
        base_name = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base_name)[0]
        doc_id = re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)  # Only allow valid chars
        doc_id = doc_id.strip('_')  # Remove leading/trailing underscores
        if len(doc_id) < 3:
            doc_id = f"doc_{doc_id}"
        self.doc_id = doc_id
        self.agents = DocumentAgents()
        self.vector_store = VectorStore()
        self.tracer = get_tracer("pdf_processor")
        self.monitor = LLMMonitor()
        self._process_document()

    def _process_document(self):
        """Runs the pipeline: extract -> chunk -> embed -> store."""
        with self.tracer.start_as_current_span("process_document") as span:
            span.set_attribute("document_path", self.pdf_path)
            start_time = time.time()

            # Extract text
            with self.tracer.start_span("extract_text") as extract_span:
                self.text = self.agents.pdf_extractor_tool().run(self.pdf_path)
                extract_span.set_attribute("text_length", len(self.text))

            # Chunk text
            with self.tracer.start_span("chunk_text") as chunk_span:
                self.chunks = self.agents.chunker_tool().run(self.text, self.chunk_size)
                chunk_span.set_attribute("chunk_count", len(self.chunks))

            # Generate embeddings
            with self.tracer.start_span("generate_embeddings") as embed_span:
                start_embed = time.time()
                self.embeddings = self.agents.embedder_tool().run({"text_chunks": self.chunks})
                embed_time = (time.time() - start_embed) * 1000  # ms
                embed_span.set_attribute("embedding_count", len(self.embeddings))

                # Log embeddings to Arize Phoenix (no print)
                for chunk, embedding in zip(self.chunks, self.embeddings):
                    self.monitor.log_embedding(
                        text=chunk,
                        embedding=embedding,
                        metadata={
                            "document_path": self.pdf_path,
                            "chunk_size": self.chunk_size
                        }
                    )

            # Store in vector database
            with self.tracer.start_span("store_vectors") as store_span:
                self.vector_store.store_document(
                    doc_id=self.doc_id,
                    chunks=self.chunks,
                    embeddings=self.embeddings,
                    metadata={"path": self.pdf_path}
                )
                store_span.set_attribute("stored_vectors", len(self.embeddings))

            # Set overall processing time
            span.set_attribute("processing_time_ms", (time.time() - start_time) * 1000)

    def get_summary(self) -> str:
        """Returns a summary of the document."""
        with self.tracer.start_as_current_span("get_summary") as span:
            start_time = time.time()
            summary = self.agents.summarizer_tool().run(self.text)
            
            # Log LLM interaction
            self.monitor.log_llm_interaction(
                prompt=f"Summarize the following text: {self.text[:100]}...",
                response=summary,
                model="gpt-4-turbo-preview",
                metadata={
                    "document_path": self.pdf_path,
                    "operation": "summarize"
                },
                latency_ms=(time.time() - start_time) * 1000
            )
            
            return summary

    def ask_question(self, question: str, top_k: int = 5) -> str:
        """Answers a question based on the document content."""
        with self.tracer.start_as_current_span("ask_question") as span:
            span.set_attribute("question", question)
            start_time = time.time()

            # Get question embedding
            with self.tracer.start_span("embed_question") as embed_span:
                question_embedding = self.agents.embedder_tool().run({"text_chunks": [question]})[0]
                self.monitor.log_embedding(
                    text=question,
                    embedding=question_embedding,
                    metadata={"operation": "question_embedding"}
                )

            # Get relevant chunks from vector store
            with self.tracer.start_span("retrieve_chunks") as retrieve_span:
                relevant_chunks = self.vector_store.query_similar(
                    doc_id=self.doc_id,
                    query_embedding=question_embedding,
                    top_k=top_k
                )
                retrieve_span.set_attribute("retrieved_chunks", len(relevant_chunks))

            # Generate answer
            with self.tracer.start_span("generate_answer") as answer_span:
                answer = self.agents.qna_tool(question, self.doc_id, top_k)
                # Log RAG interaction
                self.monitor.log_rag_interaction(
                    question=question,
                    answer=answer,
                    retrieved_chunks=relevant_chunks,
                    metadata={
                        "document_path": self.pdf_path,
                        "top_k": top_k,
                        "latency_ms": (time.time() - start_time) * 1000
                    }
                )
                return answer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agentic Document Processing Pipeline (LangChain-style)")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--question", type=str, help="Ask a question about the document")
    parser.add_argument("--summary", action="store_true", help="Get a summary of the document")
    args = parser.parse_args()

    pipeline = DocumentProcessingPipeline(args.pdf_path)
    if args.summary:
        print("\nSummary:\n" + pipeline.get_summary())
    if args.question:
        print(f"\nQ: {args.question}\nA: {pipeline.ask_question(args.question)}")

    