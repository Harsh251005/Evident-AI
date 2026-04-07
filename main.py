from config import settings
from src.ingestion.vector_store import generate_collection_name
from src.pipeline.ingestion import ingestion_pipeline
from src.pipeline.retrieval import setup_bm25, query_loop

PDF_PATH = settings.STORY_PDF_PATH

# ---------- MAIN ----------
def main():
    # 1. Run Ingestion
    # If this skips, doc/chunks/embeds will be None
    doc, chunks, embeds = ingestion_pipeline(PDF_PATH)

    collection_name = generate_collection_name(PDF_PATH)

    # 2. Setup BM25
    # This now handles the "None" case by loading from disk if ingestion was skipped
    try:
        bm25 = setup_bm25(collection_name, chunks, [c.metadata for c in chunks] if chunks else None)
    except Exception as e:
        print(f"[ERROR] BM25 Setup failed: {e}")
        return

    # 3. Enter Loop
    query_loop(collection_name, bm25)

if __name__ == "__main__":
    main()