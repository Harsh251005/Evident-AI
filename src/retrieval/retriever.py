from langsmith import traceable
from config import settings
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.reranker import rerank_documents


@traceable(name="Hybrid Retrieval")
def retrieve(query: str, collection_name: str, bm25_index=None, k: int = settings.FINAL_K):
    """
    Combines Vector Search, BM25 Search, and Cross-Encoder Reranking.
    """
    # 1. DENSE RETRIEVAL
    query_vector = embed_texts([query])[0]
    dense_docs = vector_search(query_vector, collection_name, limit=settings.INITIAL_K)

    # Apply score thresholding
    dense_docs = [d for d in dense_docs if d["score"] >= settings.SCORE_THRESHOLD]

    # 2. BM25 RETRIEVAL
    sparse_docs = []
    if bm25_index:
        # Calling the function defined in bm25.py
        sparse_docs = bm25_search(bm25_index, query, k=settings.INITIAL_K)

    # 3. COMBINE & DEDUPLICATE
    all_docs = dense_docs + sparse_docs
    seen = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc["text"].strip())
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    # 4. RERANKING
    reranked_docs = rerank_documents(query, unique_docs, top_k=settings.INITIAL_K)

    # 5. DIVERSITY FILTERING
    final_docs = []
    seen_pages = set()
    for doc in reranked_docs:
        page = doc["metadata"].get("page")
        if page not in seen_pages:
            seen_pages.add(page)
            final_docs.append(doc)
        if len(final_docs) >= k:
            break

    return final_docs[:k]