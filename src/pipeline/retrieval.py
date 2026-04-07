import os
from config import settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.retriever import retrieve
from src.generation.llm import generate_answer
from src.generation.prompt import build_prompt

TOP_K = settings.TOP_K
VECTOR_SIZE = settings.VECTOR_SIZE



def setup_bm25(collection_name, chunks=None, metadata=None):
    # Path where we expect the BM25 index to live
    index_path = f"data/indices/{collection_name}.pkl"

    # 1. Try to load existing index
    if os.path.exists(index_path):
        print(f"[INFO] Loading existing BM25 index for {collection_name}...")
        return BM25Retriever.load(index_path)

    # 2. If no index and no data provided, we have a problem
    if chunks is None or metadata is None:
        raise ValueError("BM25 index not found and no data provided to create one.")

    # 3. Build and Save
    print("[INFO] Building and saving new BM25 index...")
    bm25 = BM25Retriever(chunks)
    bm25.save(index_path)
    return bm25


# ---------- QUERY LOOP ----------
def query_loop(collection_name: str, bm25):

    print("\n\n===== Evident AI Ready =====")
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

        # Hide sources if answer is uncertain
        if "don't know" in answer.lower() or "not found" in answer.lower():
            print("\n--- SOURCES ---")
            print("No reliable sources found.")
        else:
            print("\n--- SOURCES ---")

            for c in context:
                # Use .get() to provide a fallback value
                page = c['metadata'].get('page', 'N/A')
                source = c['metadata'].get('source', 'Unknown')
                print(f"Page {page} | {source}")

        print("\n" + "=" * 50 + "\n")