# Document Understanding System

A modular, intelligent system for extracting, chunking, embedding, summarizing, and querying PDF documents using modern LLMs and vector search.

---

## Overview

This project enables users to upload PDF documents, generate summaries, and ask questions about their content. It leverages a multi-process architecture with specialized servers (MCP servers), agent-based orchestration, and ChromaDB for vector storage and retrieval.

---

## MCP Servers

The system is powered by two main MCP servers (in `server/`):

- **`pdf_processing_server.py`**  
  Handles:
  - PDF extraction (`extract_pdf_contents`)
  - Text chunking (`chunk_text`)
  - Embedding generation (`embed_chunks`)
  - Full pipeline (`process_pdf_to_embeddings`)

- **`summarizer_qna_server.py`**  
  Handles:
  - Summarization (`summarize_text`)
  - Question answering (`answer_question`)

A third server, **`vector_store.py`**, manages all vector storage and retrieval using ChromaDB.

---

## Agents

Agents are Python classes (see `agents.py`) that orchestrate the workflow by calling the appropriate MCP tools. The main agent is:

- **`DocumentAgents`**  
  Provides LangChain-compatible tools for:
  - PDF extraction
  - Chunking
  - Embedding
  - Summarization
  - QnA

These tools communicate with the MCP servers via stdio, enabling flexible, distributed processing.

---

## Workflow

1. **PDF Upload/Selection**  
   The user provides a PDF file.
2. **Extraction**  
   The system extracts text from the PDF using the `extract_pdf_contents` tool.
3. **Chunking**  
   The extracted text is split into manageable chunks.
4. **Embedding**  
   Each chunk is converted into a vector embedding.
5. **Storage**  
   Chunks and embeddings are stored in ChromaDB via the `vector_store.py` server.
6. **Summarization**  
   The user can request a summary, which is generated using the `summarize_text` tool.
7. **Question Answering**  
   The user can ask questions. The system:
   - Embeds the question
   - Retrieves relevant chunks from ChromaDB
   - Uses an LLM to answer based on the retrieved context

All orchestration is handled by the `DocumentProcessingPipeline` (see `modules/pipeline.py`), which uses the agents and MCP servers.

---

## How to Run

### 1. Install dependencies

```sh
pip install -r requirements.txt
# or, if you use uv:
uv pip install -r server/requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 3. Start the MCP servers

Open two terminals and run:

```sh
# Terminal 1: PDF processing (extraction, chunking, embedding)
python server/pdf_processing_server.py

# Terminal 2: Summarization and QnA
python server/summarizer_qna_server.py
```

### 4. (Optional) Start the vector store server

```sh
python server/vector_store.py
```

### 5. Use the CLI or app

- **CLI:**  
  Run `python cli.py` for an interactive assistant.
- **App:**  
  Run `python app.py` for a simple interactive chat.

---

## Vector Database: ChromaDB

- All embeddings and document chunks are stored in ChromaDB collections.
- Similarity search is performed using ChromaDB's query API.
- The vector database is persisted in the `vector_db/` directory by default.

---

## Tracing & Observability (Arize Phoenix)

This project supports distributed tracing and observability using **OpenTelemetry** and **Arize Phoenix**.

- **Tracing** is enabled in the CLI and pipeline via OpenTelemetry. Spans are created for key operations (extraction, chunking, embedding, summarization, QnA, etc.).
- **Arize Phoenix** is used as a tracing backend/exporter. You can view traces and monitor LLM and RAG operations in the Phoenix UI.

### How to Enable

1. **Install dependencies:**
   ```sh
   pip install opentelemetry-api opentelemetry-sdk arize-phoenix
   # or, if you use uv:
   uv pip install opentelemetry-api opentelemetry-sdk arize-phoenix
   ```
2. **Set environment variables:**
   Add these to your environment or `.env`:
   ```
   PHOENIX_ENDPOINT=<your_arize_phoenix_endpoint>
   PHOENIX_API_KEY=<your_arize_phoenix_api_key>
   ```
3. **Run your workflow as usual.**
   Traces will be sent to Arize Phoenix and can be viewed in the Phoenix UI.

---

## Extending

- Add new tools to the MCP servers for more document operations.
- Implement new agents for custom workflows.
- Swap out ChromaDB for another vector DB if needed.

---
