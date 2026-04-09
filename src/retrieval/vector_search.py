from src.ingestion.vector_store import client
from config import settings


def vector_search(query_vector, collection_name, limit=settings.INITIAL_K):
    """
    Performs a dense vector search in Qdrant.
    """
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    )

    formatted_docs = []
    for point in results.points:
        # Standardize output to match BM25 format
        formatted_docs.append({
            "text": point.payload.get("text", ""),
            "metadata": {
                "page": point.payload.get("page") or point.payload.get("page_no") or 0,
                "source": point.payload.get("source", "Unknown PDF")
            },
            "score": float(point.score)
        })

    return formatted_docs