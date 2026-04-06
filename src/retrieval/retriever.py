from src.retrieval.embedder import embed_texts
from src.retrieval.vector_store import client


def retrieve(query: str, collection_name: str, k: int = 3):
    query_embedding = embed_texts([query])[0]

    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k
    )

    formatted_results = []

    for r in results.points:
        formatted_results.append({
            "text": r.payload["text"],
            "metadata": {
                "page": r.payload["page"],
                "source": r.payload["source"]
            }
        })

    return formatted_results