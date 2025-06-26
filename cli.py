import os
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from modules.pipeline import DocumentProcessingPipeline
import asyncio
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from arize.phoenix.opentelemetry import PhoenixSpanExporter

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")

if PHOENIX_ENDPOINT and PHOENIX_API_KEY:
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    phoenix_exporter = PhoenixSpanExporter(
        endpoint=PHOENIX_ENDPOINT,
        api_key=PHOENIX_API_KEY,
    )
    span_processor = BatchSpanProcessor(phoenix_exporter)
    provider.add_span_processor(span_processor)

# Helper to call an MCP tool
async def call_mcp_tool(server_script, tool_name, arguments):
    server_params = StdioServerParameters(command="python", args=[server_script])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return result.content

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        return asyncio.ensure_future(coro)
    else:
        return asyncio.run(coro)

def mcp_qna(pdf_path, question, top_k=5):
    # 1. Chunk the PDF
    chunks = run_async(call_mcp_tool(
        "server/pdf_processing_server.py",
        "chunk_pdf",
        {"pdf_path": pdf_path}
    ))
    # 2. Embed the chunks
    embeddings = run_async(call_mcp_tool(
        "server/pdf_processing_server.py",
        "embed_chunks",
        {"text_chunks": chunks}
    ))
    # 3. Store in vector DB
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    run_async(call_mcp_tool(
        "server/vector_store.py",
        "add_document",
        {"doc_id": doc_id, "chunks": chunks, "embeddings": embeddings}
    ))
    # 4. Embed the question
    q_embedding = run_async(call_mcp_tool(
        "server/pdf_processing_server.py",
        "embed_chunks",
        {"text_chunks": [question]}
    ))[0]
    # 5. Retrieve relevant chunks
    relevant_chunks = run_async(call_mcp_tool(
        "server/vector_store.py",
        "search_embeddings",
        {"doc_id": doc_id, "query_embedding": q_embedding, "top_k": top_k}
    ))
    # 6. Call QnA service
    answer = run_async(call_mcp_tool(
        "server/summarizer_qna_server.py",
        "answer_question",
        {"question": question, "context": "\n".join(relevant_chunks)}
    ))
    return answer

def prompt_for_pdf():
    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if not os.path.isfile(pdf_path):
            print(f"File not found: {pdf_path}")
            continue
        if not pdf_path.lower().endswith('.pdf'):
            print("Please provide a valid PDF file.")
            continue
        return pdf_path

def main():
    print("Welcome to the PDF Assistant!")
    pipeline = None
    pdf_path = None
    while True:
        print("\nPlease select an action:")
        print("1. Upload/select a PDF")
        print("2. Ask a question about the PDF (one-shot QnA)")
        print("3. Start an interactive chat with the PDF")
        print("4. Get a summary of the PDF")
        print("5. Exit")
        choice = input("\n> ").strip()
        if choice == "1":
            pdf_path = prompt_for_pdf()
            print("Loading and processing PDF...")
            pipeline = DocumentProcessingPipeline(pdf_path)
            print("PDF loaded!")
        elif choice == "2":
            if not pipeline:
                print("Please upload/select a PDF first (option 1).")
                continue
            question = input("Enter your question: ").strip()
            print("\nProcessing QnA...")
            answer = pipeline.ask_question(question)
            print(f"\nAnswer:\n{answer}")
        elif choice == "3":
            if not pipeline:
                print("Please upload/select a PDF first (option 1).")
                continue
            print("\nInteractive chat started. Type 'summary' for a summary, 'clear' to reload, 'exit' to end chat.")
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    print("Ending chat...")
                    break
                if user_input.lower() == "clear":
                    print("Reloading document...")
                    pipeline = DocumentProcessingPipeline(pdf_path)
                    print("Document reloaded.")
                    continue
                if user_input.lower() == "summary":
                    print("\nSummary:\n" + pipeline.get_summary())
                    continue
                print("\nAssistant: ", end="", flush=True)
                try:
                    answer = pipeline.ask_question(user_input)
                    print(answer)
                except Exception as e:
                    print(f"\nError: {e}")
        elif choice == "4":
            if not pipeline:
                print("Please upload/select a PDF first (option 1).")
                continue
            print("\nGenerating summary...")
            summary = pipeline.get_summary()
            print(f"\nSummary:\n{summary}")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

if __name__ == "__main__":
    main() 