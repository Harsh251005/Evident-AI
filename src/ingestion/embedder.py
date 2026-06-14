from openai import OpenAI

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

client = OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    """

    if not texts:
        logger.warning("Received empty text list for embedding")
        return []

    try:
        logger.info(
            f"Generating embeddings for {len(texts)} texts "
            f"using {settings.EMBEDDING_MODEL}"
        )

        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )

        return [item.embedding for item in response.data]

    except Exception as e:
        logger.exception("Embedding generation failed")
        raise RuntimeError("Failed to generate embeddings") from e

