import os
import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))


def rerank(query, documents, top_k=10):
    if not documents:
        return []

    texts = [doc["text"] for doc in documents]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_k
    )

    return [documents[r.index] for r in response.results]