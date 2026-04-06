import os

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import (
    generate_collection_name,
    create_collection_if_not_exists,
    is_collection_empty,
    add_points,
)
from src.retrieval.retriever import retrieve

from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer


# ---------- CONFIG ----------
PDF_PATH = r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\claudes-constitution_webPDF_26-02.02a.pdf"
TOP_K = 10


# ---------- INGESTION PIPELINE ----------
def ingest_pdf(pdf_path: str):
    print("\n[INFO] Loading PDF...")
    documents = load_pdf(pdf_path)

    print("[INFO] Chunking text...")
    chunks = chunk_text(documents, chunk_size=400, overlap=100)

    texts = [c["text"] for c in chunks]
    metadata = [
        {
            "page": c["page"],
            "source": os.path.basename(pdf_path)
        }
        for c in chunks
    ]

    print("[INFO] Creating embeddings...")
    embeddings = embed_texts(texts)

    return embeddings, texts, metadata


# ---------- SETUP VECTOR STORE ----------
def setup_vector_store(pdf_path: str):
    print("\n[INFO] Preparing vector store...")

    collection_name = generate_collection_name(pdf_path)

    # Dummy embedding to get vector size (safe search)
    sample_embedding = embed_texts(["test"])[0]
    vector_size = len(sample_embedding)

    create_collection_if_not_exists(collection_name, vector_size)

    if is_collection_empty(collection_name):
        print("[INFO] Collection empty → ingesting PDF...")

        embeddings, texts, metadata = ingest_pdf(pdf_path)

        add_points(collection_name, embeddings, texts, metadata)

        print("[INFO] Ingestion complete ✅")
    else:
        print("[INFO] Collection already exists → skipping ingestion ✅")

    return collection_name


# ---------- QUERY LOOP ----------
def query_loop(collection_name: str):
    print("\n=== Evident AI Ready ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question(or 'exit'): ")

        if query.lower() == "exit":
            break

        print("\n[INFO] Retrieving context...")
        context = retrieve(query, collection_name, k=TOP_K)

        print("[INFO] Generating answer...")
        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- SOURCES ---")
        for c in context:
            print(f"Page {c['metadata']['page']} | {c['metadata']['source']}")

        print("\n" + "=" * 50 + "\n")


# ---------- MAIN ----------
def main():
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] File not found: {PDF_PATH}")
        return

    collection_name = setup_vector_store(PDF_PATH)

    query_loop(collection_name)


if __name__ == "__main__":
    main()