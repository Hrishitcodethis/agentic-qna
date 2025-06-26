import os
import re
import asyncio
import nest_asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

class DocumentProcessingPipeline:
    def __init__(self, pdf_path: str, chunk_size: int = 500):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.doc_id = self._sanitize_doc_id(pdf_path)
        self.text = None
        self.chunks = None
        self.embeddings = None
        self._process_document()

    def _sanitize_doc_id(self, pdf_path):
        base = os.path.basename(pdf_path)
        doc_id = os.path.splitext(base)[0]
        doc_id = re.sub(r'[^a-zA-Z0-9._-]', '_', doc_id)
        return doc_id

    async def _call_mcp_tool(self, server_script, tool_name, arguments):
        server_params = StdioServerParameters(command="python", args=[server_script])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result.content

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def _process_document(self):
        # 1. Extract text
        self.text = self._run_async(self._call_mcp_tool(
            "server/pdf_extractor.py", "extract_pdf_contents", {"pdf_path": self.pdf_path}
        ))[0]
        # If it's an mcp.types.TextContent, extract the .text attribute
        if hasattr(self.text, "text"):
            self.text = self.text.text
        print(f"[DEBUG] Extracted text type: {type(self.text)}, value: {str(self.text)[:200]}")
        # 2. Chunk text
        self.chunks = self._run_async(self._call_mcp_tool(
            "server/pdf_processing_server.py", "chunk_text", {"text": self.text, "chunk_size": self.chunk_size}
        ))
        # 3. Embed chunks
        self.embeddings = self._run_async(self._call_mcp_tool(
            "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": self.chunks}
        ))
        # 4. Store in vector DB
        self._run_async(self._call_mcp_tool(
            "server/vector_store.py", "add_document", {
                "doc_id": self.doc_id,
                "chunks": self.chunks,
                "embeddings": self.embeddings
            }
        ))

    def get_summary(self) -> str:
        text = self.text
        # Robustly extract text if it's a dict or list of dicts
        if isinstance(text, dict) and "text" in text:
            text = text["text"]
        elif isinstance(text, list):
            # If it's a list of dicts, extract their 'text' fields
            if all(isinstance(t, dict) and "text" in t for t in text):
                text = [t["text"] for t in text]
        summary = self._run_async(self._call_mcp_tool(
            "server/summarizer_qna_server.py", "summarize_text", {"text": text}
        ))[0]
        return summary

    def ask_question(self, question: str, top_k: int = 5) -> str:
        # 1. Embed question
        q_embedding = self._run_async(self._call_mcp_tool(
            "server/pdf_processing_server.py", "embed_chunks", {"text_chunks": [question]}
        ))[0]
        # 2. Retrieve relevant chunks
        relevant_chunks = self._run_async(self._call_mcp_tool(
            "server/vector_store.py", "search_embeddings", {
                "doc_id": self.doc_id,
                "query_embedding": q_embedding,
                "top_k": top_k
            }
        ))
        # Extract text from TextContent objects if needed
        relevant_chunks = [
            chunk.text if hasattr(chunk, "text") else chunk
            for chunk in relevant_chunks
        ]
        # 3. QnA
        answer = self._run_async(self._call_mcp_tool(
            "server/summarizer_qna_server.py", "answer_question", {
                "doc_id": self.doc_id,
                "question": question,
                "top_k": top_k
            }
        ))[0]
        return answer 