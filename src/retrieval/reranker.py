import numpy as np
from sentence_transformers import CrossEncoder
from config import settings

# Initialize the model using the configuration path
reranker_model = CrossEncoder(settings.RERANKER_MODEL)


def sigmoid(x):
    """Squashes raw scores into a 0-1 range for easier thresholding."""
    return 1 / (1 + np.exp(-x))


def rerank_documents(query, documents, top_k=settings.FINAL_K):
    """
    Uses a Cross-Encoder to re-score documents based on query relevance.
    """
    if not documents:
        return []

    # Create (query, document_text) pairs for the model
    pairs = [(query, doc["text"]) for doc in documents]

    # Generate raw relevance scores
    raw_scores = reranker_model.predict(pairs)
    probabilities = sigmoid(raw_scores)

    # Attach the new scores to the document objects
    for i, doc in enumerate(documents):
        doc["score"] = float(probabilities[i])

    # Sort by the new cross-encoder score
    documents.sort(key=lambda x: x["score"], reverse=True)

    return documents[:top_k]