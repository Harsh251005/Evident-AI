from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import client
from src.retrieval.reranker import rerank


# ---------- CONFIG ----------
INITIAL_K = 15
FINAL_K = 5
SCORE_THRESHOLD = 0.4   # tune between 0.3–0.6


def retrieve(query: str, collection_name: str, bm25=None, k: int = FINAL_K):
    # ---------- DENSE RETRIEVAL ----------
    query_embedding = embed_texts([query])[0]

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=INITIAL_K
    )

    dense_docs = [
        {
            "text": r.payload["text"],
            "metadata": {
                "page": r.payload["page"],
                "source": r.payload["source"]
            },
            "score": r.score
        }
        for r in results.points
    ]

    # ---------- SCORE FILTER ----------
    dense_docs = [d for d in dense_docs if d["score"] >= SCORE_THRESHOLD]

    # ---------- BM25 ----------
    bm25_docs = bm25.search(query, k=INITIAL_K) if bm25 else []

    # ---------- MERGE ----------
    all_docs = dense_docs + bm25_docs

    # Deduplicate by text
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["text"] not in seen:
            seen.add(doc["text"])
            unique_docs.append(doc)

    if not unique_docs:
        return []

    # ---------- RERANK ----------
    reranked_docs = rerank(query, unique_docs, top_k=INITIAL_K)

    # ---------- DIVERSITY (page-level) ----------
    final_docs = []
    seen_pages = set()

    for doc in reranked_docs:
        page = doc["metadata"]["page"]

        if page not in seen_pages:
            seen_pages.add(page)
            final_docs.append(doc)

        if len(final_docs) == k:
            break

    return final_docs