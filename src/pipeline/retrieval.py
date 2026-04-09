from config import settings
from src.retrieval.hybrid import hybrid_search as retrieve
from src.retrieval.bm25 import setup_bm25
from src.generation.llm import generate_answer
from src.generation.prompt import build_prompt

# Configuration
TOP_K = settings.TOP_K


# ---------- QUERY LOOP ----------
def query_loop(collection_name: str, bm25_index):
    """
    Main interactive loop for testing the RAG system.
    """
    print("\n\n===== Evident AI Ready =====")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        if not query.strip():
            continue

        print("\n[INFO] Retrieving context...")
        # Uses the hybrid retrieval logic
        context = retrieve(query, collection_name, bm25_index=bm25_index, k=TOP_K)

        if not context:
            print("\n--- ANSWER ---")
            print("I couldn't find relevant information in the document.")
            print("\n" + "=" * 50 + "\n")
            continue

        print(f"[INFO] Generating answer for query: {query}")

        # Build prompt and generate answer
        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

        print("\n--- ANSWER ---")
        print(answer)

        # Logic to display or hide sources based on answer certainty
        if "don't know" in answer.lower() or "not found" in answer.lower():
            print("\n--- SOURCES ---")
            print("No reliable sources found.")
        else:
            print("\n--- SOURCES ---")
            for c in context:
                # Use .get() for safe metadata access
                page = c['metadata'].get('page', 'N/A')
                source = c['metadata'].get('source', 'Unknown')
                print(f"Page {page} | {source}")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    from src.ingestion.vector_store import generate_collection_name
    from src.pipeline.ingestion import ingestion_pipeline

    test_pdf_path = r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\claudes-constitution_webPDF_26-02.02a.pdf"
    coll_name = generate_collection_name(test_pdf_path)

    try:
        try:
            # Try loading existing
            bm25_idx = setup_bm25(coll_name)
        except ValueError:
            # Rebuild if missing
            print("[INFO] BM25 Index missing. Triggering sync via ingestion...")
            _, _, bm25_idx = ingestion_pipeline(test_pdf_path)

        query_loop(coll_name, bm25_idx)
    except Exception as e:
        print(f"[ERROR] Could not initialize retrieval: {e}")