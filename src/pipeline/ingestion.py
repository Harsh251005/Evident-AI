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
from src.retrieval.bm25 import setup_bm25


def ingestion_pipeline(pdf_path: str, force_reingest: bool = False, collection_name: str = None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"[ERROR] File not found: {pdf_path}")

    # 1. Use the provided name (from the uploader) or generate a default one
    if collection_name is None:
        collection_name = generate_collection_name(pdf_path)

    # 2. FIX: Unified naming. Ensure this matches what setup_bm25 looks for!
    # Most systems use _bm25.pkl suffix.
    bm25_path = f"data/indices/{collection_name}_bm25.pkl"

    # Check if we can truly skip
    vector_exists = not is_collection_empty(collection_name)
    bm25_exists = os.path.exists(bm25_path)

    if not force_reingest and vector_exists and bm25_exists:
        print(f"[INFO] Collection '{collection_name}' already indexed. Skipping. ✅")
        # Just load existing
        bm25_retriever = setup_bm25(collection_name)
        return None, None, bm25_retriever

    # --- START OF PROCESSING ---
    print(f"[INFO] Processing Document for Collection: {collection_name}")
    doc = load_pdf(pdf_path)
    chunks = split_texts(doc)

    # 1. Handle Vector Store
    if force_reingest or not vector_exists:
        print(f"[INFO] Ingesting {len(chunks)} points into Qdrant...")
        texts_to_embed = [chunk.page_content for chunk in chunks]
        embedded_texts = embed_texts(texts_to_embed)
        create_collection_if_not_exists(collection_name, settings.VECTOR_SIZE)
        add_points(collection_name, embedded_texts, chunks, [c.metadata for c in chunks])

    # 2. Handle BM25 Index
    # We pass 'chunks' here so if the file is missing, it builds it
    print(f"[INFO] Building BM25 index at {bm25_path}...")
    bm25_retriever = setup_bm25(collection_name, chunks=chunks)

    print("[INFO] Ingestion/Sync complete ✅")
    return doc, chunks, bm25_retriever