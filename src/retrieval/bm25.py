from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.ingestion.vector_store import client
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _get_collection_documents(
    collection_name: str,
) -> list[Document]:
    """
    Fetch all documents from a Qdrant collection.
    """

    documents = []

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload

        documents.append(
            Document(
                page_content=payload["text"],
                metadata={
                    "source": payload.get("source"),
                    "page_no": payload.get("page_no"),
                },
            )
        )

    return documents


def bm25_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Perform BM25 retrieval against all chunks in a collection.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    documents = _get_collection_documents(collection_name)

    if not documents:
        logger.warning(
            f"No documents found in {collection_name}"
        )
        return []

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for idx in ranked_indices:
        doc = documents[idx]

        doc.metadata["score"] = float(scores[idx])

        results.append(doc)

    logger.info(
        f"BM25 retrieved {len(results)} chunks "
        f"from {collection_name}"
    )

    return results