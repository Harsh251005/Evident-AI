from flashrank import Ranker, RerankRequest
from config import settings

# 1. Initialize Ranker once at the module level.
# ms-marco-MiniLM-L-12-v2 is the standard, fast choice.
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")


def rerank_documents(query, documents, top_k=settings.FINAL_K):
    """
    Uses FlashRank (optimized for CPU) to re-score documents based on query relevance.
    """
    if not documents:
        return []

    # 2. Format documents for FlashRank's required schema
    passages = [
        {"id": i, "text": doc["text"], "meta": doc.get("metadata", {})}
        for i, doc in enumerate(documents)
    ]

    # 3. Create the Rerank Request
    rerank_request = RerankRequest(query=query, passages=passages)

    # 4. Execute Reranking (This is where the speed happens)
    # FlashRank returns a list of dicts sorted by score automatically.
    results = ranker.rerank(rerank_request)

    # 5. Map back to your original format and apply the SCORE_THRESHOLD
    final_docs = []
    for res in results:
        # FlashRank scores are typically normalized (0-1)
        if res["score"] >= settings.SCORE_THRESHOLD:
            final_docs.append({
                "text": res["text"],
                "metadata": res["meta"],
                "score": float(res["score"])
            })

    # Return top_k results
    return final_docs[:top_k]