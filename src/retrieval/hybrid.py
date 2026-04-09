from langsmith import traceable
from config import settings
from src.ingestion.embedder import embed_texts
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.reranker import rerank_documents


@traceable(name="Hybrid Retrieval Pipeline")
def hybrid_search(query, collection_name, bm25_index=None, k=settings.FINAL_K):
    """
    Orchestrates the full retrieval flow: Dense + Sparse -> Deduplicate -> Rerank.
    """
    # 1. Generate Query Embedding
    query_vector = embed_texts([query])[0]

    # 2. Get Dense (Vector) and Sparse (BM25) results
    dense_docs = vector_search(query_vector, collection_name, limit=settings.INITIAL_K)

    sparse_docs = []
    if bm25_index:
        sparse_docs = bm25_search(bm25_index, query, k=settings.INITIAL_K)

    # 3. Combine and Deduplicate based on text content
    all_results = dense_docs + sparse_docs
    unique_docs = []
    seen_hashes = set()

    for doc in all_results:
        text_hash = hash(doc["text"].strip())
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)

    # 4. Final Reranking
    return rerank_documents(query, unique_docs, top_k=k)