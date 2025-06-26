from agents import DocumentProcessingPipeline
import os

def interactive_document_chat():
    print("===== Interactive Document Q&A and Summarization =====")
    pdf_path = input("Enter the path to your PDF document: ").strip()
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return
    pipeline = DocumentProcessingPipeline(pdf_path)
    print("\nDocument loaded and processed. You can now ask questions or type 'summary' to get a summary.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to reload the document.")
    print("==================================\n")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation...")
            break
        if user_input.lower() == "clear":
            print("Reloading document...")
            pipeline = DocumentProcessingPipeline(pdf_path)
            print("Document reloaded.")
            continue
        if user_input.lower() == "summary":
            print("\nSummary:\n" + pipeline.get_summary())
            continue
        # Otherwise, treat as a question
        print("\nAssistant: ", end="", flush=True)
        try:
            answer = pipeline.ask_question(user_input)
            print(answer)
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    interactive_document_chat()