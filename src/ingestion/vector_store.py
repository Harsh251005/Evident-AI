import hashlib
import uuid
import os
from config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY  # This is the new part
)
def generate_collection_name(file_path: str) -> str:
    """
    Generates a unique, clean collection name.
    """
    file_name = os.path.basename(file_path)
    # Important: read file to get hash
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]

    # Clean the name: remove temp prefixes, extensions, and spaces
    clean_name = file_name.replace("temp_", "").replace(".pdf", "").replace(" ", "_").lower()
    return f"{clean_name}_{file_hash}"

def create_collection_if_not_exists(collection_name: str, vector_size: int):
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def is_collection_empty(collection_name: str) -> bool:
    try:
        info = client.get_collection(collection_name)
        return info.points_count == 0
    except:
        return True

def add_points(collection_name, embeddings, chunks, metadata):
    points = []
    for i in range(len(embeddings)):
        chunk = chunks[i]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "text": chunk.page_content,
                    # Ensure this key matches what evident_rag.py looks for!
                    "page": chunk.metadata.get("page", chunk.metadata.get("page_no", 0)),
                    "source": chunk.metadata.get("source", "unknown")
                }
            )
        )
    client.upsert(collection_name=collection_name, points=points)