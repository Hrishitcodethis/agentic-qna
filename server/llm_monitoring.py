import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

class LLMMonitor:
    def __init__(self):
        """Initialize the LLM monitoring client with basic logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("llm_monitor")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler("logs/llm_monitor.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_embedding(self, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None):
        """
        Log an embedding operation.
        Args:
            text: The input text that was embedded
            embedding: The resulting embedding vector
            metadata: Optional metadata about the embedding operation
        """
        log_data = {
            "type": "embedding",
            "trace_id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "text_length": len(text),
            "embedding_dim": len(embedding),
            "metadata": metadata or {}
        }
        self.logger.info(f"Embedding operation: {json.dumps(log_data)}")

    def log_llm_interaction(
        self,
        prompt: str,
        response: str,
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None
    ):
        """
        Log an LLM interaction.
        Args:
            prompt: The input prompt
            response: The LLM's response
            model: The model name/identifier
            metadata: Optional metadata about the interaction
            latency_ms: Optional latency in milliseconds
        """
        log_data = {
            "type": "llm_interaction",
            "trace_id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency_ms": latency_ms,
            "metadata": metadata or {}
        }
        self.logger.info(f"LLM interaction: {json.dumps(log_data)}")

    def log_rag_interaction(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a RAG (Retrieval-Augmented Generation) interaction.
        Args:
            question: The user's question
            answer: The generated answer
            retrieved_chunks: The chunks retrieved from the vector store
            metadata: Optional metadata about the interaction
        """
        log_data = {
            "type": "rag_interaction",
            "trace_id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "question_length": len(question),
            "answer_length": len(answer),
            "chunks_count": len(retrieved_chunks),
            "metadata": metadata or {}
        }
        self.logger.info(f"RAG interaction: {json.dumps(log_data)}") 