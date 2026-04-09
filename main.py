from config import settings
from src.ingestion.vector_store import generate_collection_name
from src.pipeline.ingestion import ingestion_pipeline
from src.pipeline.retrieval import setup_bm25, query_loop

# CHANGE THIS LINE:
PDF_PATH = settings.CLAUDE_CONSTITUTION_PDF_PATH

def main():
    # 1. Run Ingestion
    doc, chunks, embeds = ingestion_pipeline(PDF_PATH)
    collection_name = generate_collection_name(PDF_PATH)

    # 2. Setup BM25 (FIXED: Ensure only 2 arguments are passed)
    try:
        # We only pass collection_name and chunks
        bm25 = setup_bm25(collection_name, chunks)
    except Exception as e:
        print(f"[ERROR] BM25 Setup failed: {e}")
        return

    # 3. Enter Loop
    query_loop(collection_name, bm25)

if __name__ == "__main__":
    main()