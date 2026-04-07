import os
from config import settings
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import split_texts
from src.ingestion.embedder import embed_texts
from src.ingestion.vector_store import (
    generate_collection_name,
    create_collection_if_not_exists,
    is_collection_empty,
    add_points,
)
from src.retrieval.bm25 import BM25Retriever

VECTOR_SIZE = settings.VECTOR_SIZE


def ingestion_pipeline(pdf_path: str, force_reingest: bool = False):
    # 1. Basic File Check
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[ERROR] File not found: {pdf_path}")

    # 2. Generate collection name immediately
    collection_name = generate_collection_name(pdf_path)

    # 3. Check if we can skip the entire process
    if not force_reingest and not is_collection_empty(collection_name):
        print(f"[INFO] Collection '{collection_name}' already exists and has data. Skipping all steps. ✅")
        return None, None, None

    # --- START OF HEAVY PROCESSING (Only runs if needed) ---

    print(f"[INFO] Loading PDF: {pdf_path}")
    doc = load_pdf(pdf_path)

    print(f"[INFO] Processing PDF: {pdf_path}")
    chunks = split_texts(doc)
    metadata = [chunk.metadata for chunk in chunks]

    print(f"[INFO] Creating Embeddings for {len(chunks)} chunks...")
    texts_to_embed = [chunk.page_content for chunk in chunks]
    embedded_texts = embed_texts(texts_to_embed)

    print("\n[INFO] Preparing vector store...")
    # Ensure collection is ready
    create_collection_if_not_exists(collection_name, VECTOR_SIZE)

    print(f"[INFO] Ingesting {len(chunks)} points into {collection_name}...")
    add_points(collection_name, embedded_texts, chunks, metadata)
    bm25_path = f"data/indices/{collection_name}.pkl"
    bm25_retriever = BM25Retriever(chunks)
    bm25_retriever.save(bm25_path)
    print("[INFO] Ingestion complete ✅")

    return doc, chunks, embedded_texts