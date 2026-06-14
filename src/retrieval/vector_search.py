from langchain_core.documents import Document
from openai import OpenAI
from qdrant_client.models import ScoredPoint

from config.settings import settings
from src.ingestion.vector_store import client
from src.utils.logger import get_logger

logger = get_logger(__name__)

openai_client = OpenAI()


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a user query.
    """

    response = openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=query,
    )

    return response.data[0].embedding


def vector_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Perform dense vector search against a qdrant collection.

    Returns:
        List[Document]
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info(
        f"Vector search | collection={collection_name} | top_k={top_k}"
    )

    query_embedding = embed_query(query)

    results: list[ScoredPoint] = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    ).points

    documents = []

    for point in results:

        logger.info(
            f"Retrieved chunk | score={point.score:.4f}"
        )

        payload = point.payload

        documents.append(
            Document(
                page_content=payload["text"],
                metadata={
                    "source": payload.get("source"),
                    "page_no": payload.get("page_no"),
                    "score": point.score,
                },
            )
        )

    logger.info(
        f"Retrieved {len(documents)} chunks from {collection_name}"
    )

    return documents