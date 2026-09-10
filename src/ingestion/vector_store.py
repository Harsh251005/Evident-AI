from pathlib import Path
import hashlib
import time
import uuid

import httpx
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# upload_points() has its own built-in retry; plain calls like
# collection_exists()/get_collection() don't, and a CI runner's network
# path to a remote cluster can hit a transient reset that a laptop's
# connection doesn't (observed in practice, not hypothetical).
def _with_retry(fn, *args, attempts: int = 3, delay: float = 2.0, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (
            ResponseHandlingException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
        ) as e:
            if attempt == attempts:
                raise
            logger.warning(
                f"Qdrant call failed (attempt {attempt}/{attempts}): {e} — retrying"
            )
            time.sleep(delay * attempt)

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=60,
    # Qdrant Cloud clusters are managed and may run a slightly newer server
    # version than the pinned client — harmless, so skip the version check.
    check_compatibility=False,
)

def generate_collection_name(file_path: str) -> str:
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:8]

    clean_name = (
        file_path.stem
        .replace("temp_", "")
        .replace(" ", "_")
        .lower()
    )

    return f"{clean_name}_{file_hash}"


def create_collection_if_not_exists(
    collection_name: str,
    vector_size: int,
) -> None:

    if _with_retry(client.collection_exists, collection_name):
        logger.info(
            f"Collection '{collection_name}' already exists"
        )
        return

    _with_retry(
        client.create_collection,
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    logger.info(
        f"Created collection '{collection_name}'"
    )


def add_points(
    collection_name: str,
    embeddings: list[list[float]],
    chunks: list[Document],
) -> None:

    points = []

    for embedding, chunk in zip(embeddings, chunks):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source"),
                    "page_no": chunk.metadata.get("page_no"),
                },
            )
        )

    # Batched, retrying upload — a single upsert() call with hundreds of
    # embeddings in one request times out against a remote cloud cluster.
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        max_retries=3,
        wait=True,
    )

    logger.info(
        f"Inserted {len(points)} points into "
        f"'{collection_name}'"
    )

def collection_exists(collection_name: str) -> bool:
    return _with_retry(client.collection_exists, collection_name)


def collection_is_populated(collection_name: str) -> bool:
    """
    A collection can exist but be empty — e.g. after create_collection()
    succeeded but the point upload failed or was interrupted. Ingestion
    should only be skipped if the collection actually has data.
    """
    if not _with_retry(client.collection_exists, collection_name):
        return False

    return _with_retry(client.get_collection, collection_name).points_count > 0