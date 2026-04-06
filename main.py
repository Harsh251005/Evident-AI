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
from src.retrieval.bm25 import BM25Retriever

from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer


# ---------- CONFIG ----------
PDF_PATH = r"D:\Harsh\Code\Resume Projects\EvidentAI\data\sample_pdf\claudes-constitution_webPDF_26-02.02a.pdf"
TOP_K = 5
VECTOR_SIZE = 1536  # OpenAI embedding dim


# ---------- INGESTION ----------
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

    print(f"[INFO] Generating {len(texts)} embeddings...")
    embeddings = embed_texts(texts)

    return embeddings, texts, metadata


# ---------- VECTOR STORE ----------
def setup_vector_store(pdf_path: str):
    print("\n[INFO] Preparing vector store...")

    collection_name = generate_collection_name(pdf_path)

    create_collection_if_not_exists(collection_name, VECTOR_SIZE)

    if is_collection_empty(collection_name):
        print("[INFO] Collection empty → ingesting PDF...")

        embeddings, texts, metadata = ingest_pdf(pdf_path)
        add_points(collection_name, embeddings, texts, metadata)

        print("[INFO] Ingestion complete ✅")

    else:
        print("[INFO] Collection exists → skipping ingestion ✅")

        print("[INFO] Re-loading PDF for BM25...")
        documents = load_pdf(pdf_path)
        chunks = chunk_text(documents, chunk_size=400, overlap=100)

        texts = [c["text"] for c in chunks]
        metadata = [
            {
                "page": c["page"],
                "source": os.path.basename(pdf_path)
            }
            for c in chunks
        ]

    return collection_name, texts, metadata


# ---------- BM25 ----------
def setup_bm25(texts, metadata):
    print("[INFO] Building BM25 index...")
    return BM25Retriever(texts, metadata)


# ---------- QUERY LOOP ----------
def query_loop(collection_name: str, bm25):
    print("\n=== Evident AI Ready ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        print("\n[INFO] Retrieving context...")
        context = retrieve(query, collection_name, bm25=bm25, k=TOP_K)

        if not context:
            print("\n--- ANSWER ---")
            print("I couldn't find relevant information in the document.")
            print("\n" + "=" * 50 + "\n")
            continue

        print(f"[INFO] Generating answer for query: {query}")

        prompt = build_prompt(context, query)
        answer = generate_answer(prompt)

        print("\n--- ANSWER ---")
        print(answer)

        # 🔥 Hide sources if answer is uncertain
        if "don't know" in answer.lower() or "not found" in answer.lower():
            print("\n--- SOURCES ---")
            print("No reliable sources found.")
        else:
            print("\n--- SOURCES ---")
            for c in context:
                print(f"Page {c['metadata']['page']} | {c['metadata']['source']}")

        print("\n" + "=" * 50 + "\n")


# ---------- MAIN ----------
def main():
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] File not found: {PDF_PATH}")
        return

    collection_name, texts, metadata = setup_vector_store(PDF_PATH)

    bm25 = setup_bm25(texts, metadata)

    query_loop(collection_name, bm25)


if __name__ == "__main__":
    main()