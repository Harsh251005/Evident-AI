import time

from langchain_core.documents import Document
from openai import OpenAI, APIError, APIConnectionError
from qdrant_client.models import ScoredPoint

from config.settings import settings
from src.ingestion.vector_store import client, _with_retry
from src.utils.logger import get_logger

logger = get_logger(__name__)

openai_client = OpenAI()


def embed_query(query: str, attempts: int = 3) -> list[float]:
    """
    Generate embedding for a user query.
    """

    for attempt in range(1, attempts + 1):
        try:
            response = openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=query,
            )
            return response.data[0].embedding

        except (APIConnectionError, APIError):
            if attempt == attempts:
                raise
            logger.warning(
                f"Embedding call failed (attempt {attempt}/{attempts}) — retrying"
            )
            time.sleep(2 * attempt)


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

    results: list[ScoredPoint] = _with_retry(
        client.query_points,
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    ).points

    documents = []

    for point in results:

        payload = point.payload

        logger.info(
            f"Retrieved chunk | "
            f"source={payload.get('source')} | "
            f"page={payload.get('page_no')} | "
            f"score={point.score:.4f}"
        )

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