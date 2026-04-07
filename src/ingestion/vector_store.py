import hashlib
import uuid
import os

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# ---------- QDRANT CLIENT ----------
client = QdrantClient(url="http://localhost:6333")


# ---------- UTILS ----------

def generate_collection_name(file_path: str) -> str:
    """
    Generates unique collection name using file name + hash
    """
    file_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]

    clean_name = file_name.replace(".pdf", "").replace(" ", "_").lower()

    return f"{clean_name}_{file_hash}"


# ---------- COLLECTION MANAGEMENT ----------

def create_collection_if_not_exists(collection_name: str, vector_size: int):
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def is_collection_empty(collection_name: str) -> bool:
    try:
        info = client.get_collection(collection_name)
        return info.points_count == 0
    except:
        return True


# ---------- ADD DATA ----------

def add_points(collection_name, embeddings, texts, metadata):
    points = []

    for i in range(len(embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "text": texts[i],
                    "page": metadata[i]["page"],
                    "source": metadata[i].get("source", "unknown")
                }
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )