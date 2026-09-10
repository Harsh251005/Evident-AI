from pathlib import Path
import hashlib
import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

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

    if client.collection_exists(collection_name):
        logger.info(
            f"Collection '{collection_name}' already exists"
        )
        return

    client.create_collection(
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
    return client.collection_exists(collection_name)


def collection_is_populated(collection_name: str) -> bool:
    """
    A collection can exist but be empty — e.g. after create_collection()
    succeeded but the point upload failed or was interrupted. Ingestion
    should only be skipped if the collection actually has data.
    """
    if not client.collection_exists(collection_name):
        return False

    return client.get_collection(collection_name).points_count > 0