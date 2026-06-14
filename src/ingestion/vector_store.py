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

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    logger.info(
        f"Inserted {len(points)} points into "
        f"'{collection_name}'"
    )

def collection_exists(collection_name: str) -> bool:
    return client.collection_exists(collection_name)