import argparse
from config import settings
from src.ingestion.vector_store import generate_collection_name
from src.pipeline.ingestion import ingestion_pipeline
from src.pipeline.retrieval import setup_bm25, query_loop

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="EvidentAI RAG Pipeline")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the PDF file for ingestion",
        default=settings.CLAUDE_CONSTITUTION_PDF_PATH
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run ingestion and BM25 indexing then exit (used for CI/CD)"
    )
    args = parser.parse_args()

    # Use the path from arguments
    pdf_path = args.file

    # 1. Run Ingestion (Syncs vectors to Qdrant Cloud)
    print(f"[INFO] Starting ingestion for: {pdf_path}")
    doc, chunks, embeds = ingestion_pipeline(pdf_path)
    collection_name = generate_collection_name(pdf_path)

    # 2. Setup BM25 (Creates local .pkl for the GitHub Runner)
    try:
        bm25 = setup_bm25(collection_name, chunks)
    except Exception as e:
        print(f"[ERROR] BM25 Setup failed: {e}")
        return

    # 3. CI/CD Check: Exit if ingest-only flag is present
    if args.ingest_only:
        print(f"[INFO] Ingestion and BM25 indexing complete for {collection_name}. Exiting gracefully.")
        return

    # 4. Interactive Query Loop (Only runs locally/manually)
    query_loop(collection_name, bm25)

if __name__ == "__main__":
    main()