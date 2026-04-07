from config import settings
from sentence_transformers import CrossEncoder
import numpy as np

reranker_model = CrossEncoder(settings.RERANKER_MODEL)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rerank(query, documents, top_k=10):
    if not documents:
        return []

    pairs = [(query, doc["text"]) for doc in documents]

    # Raw logit scores (e.g., 1.5, 3.4)
    raw_scores = reranker_model.predict(pairs)

    # Squash them into 0-1 range
    probabilities = sigmoid(raw_scores)

    # Attach the clean scores
    for i, doc in enumerate(documents):
        doc["score"] = float(probabilities[i])

    # Sort by the new clean score
    documents.sort(key=lambda x: x["score"], reverse=True)

    return documents[:top_k]