from config import settings
from src.ingestion.embedder import embed_texts
from src.ingestion.vector_store import client
from src.retrieval.reranker import rerank

# ---------- CONFIG ----------
INITIAL_K = settings.INITIAL_K
FINAL_K = settings.FINAL_K
SCORE_THRESHOLD = settings.SCORE_THRESHOLD


def retrieve(query: str, collection_name: str, bm25=None, k: int = FINAL_K):
    # 1. DENSE RETRIEVAL
    query_embedding = embed_texts([query])[0]
    results = client.query_points(collection_name=collection_name, query=query_embedding, limit=INITIAL_K)

    dense_docs = []
    for doc in results.points:
        payload = doc.payload
        dense_docs.append({
            "text": payload.get("text", ""),
            "metadata": {
                "page": payload.get("page") or payload.get("page_no") or 0,
                "source": payload.get("source", "Unknown PDF")
            },
            "score": doc.score
        })

    # Apply Score Filter to Dense results only
    dense_docs = [d for d in dense_docs if d["score"] >= SCORE_THRESHOLD]

    # 2. BM25 RETRIEVAL (Use the passed 'bm25' object directly)
    bm25_docs = []
    if bm25:
        raw_bm25_docs = bm25.search(query, k=INITIAL_K)
        for doc in raw_bm25_docs:
            meta = doc.get("metadata", {})
            bm25_docs.append({
                "text": doc.get("text", ""),
                "metadata": {
                    "page": meta.get("page") or meta.get("page_no") or 0,
                    "source": meta.get("source", "Unknown PDF")
                },
                "score": doc.get("score", 0)
            })

    # 3. HYBRID MERGE & DEDUPLICATION
    all_docs = dense_docs + bm25_docs

    seen_text = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc["text"].strip())  # Using hash is faster for long texts
        if content_hash not in seen_text:
            seen_text.add(content_hash)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    # 4. RERANKING
    reranked_docs = rerank(query, unique_docs, top_k=INITIAL_K)

    # 5. DIVERSITY FILTERING (Page-level)
    final_docs = []
    seen_pages = set()

    for doc in reranked_docs:
        page = doc["metadata"].get("page")
        if page not in seen_pages:
            seen_pages.add(page)
            final_docs.append(doc)

        if len(final_docs) == k:
            break

    # Fallback if diversity is too strict
    if len(final_docs) < k:
        for doc in reranked_docs:
            if doc not in final_docs:
                final_docs.append(doc)
            if len(final_docs) == k:
                break

    return final_docs