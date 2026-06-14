from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_texts
from src.ingestion.vector_store import (
    generate_collection_name,
    collection_exists,
    create_collection_if_not_exists,
    add_points,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_document(file_path: str) -> str:
    """
    Complete ingestion pipeline.

    Steps:
    1. Generate collection name
    2. Check if document already exists
    3. Load PDF
    4. Chunk documents
    5. Generate embeddings
    6. Create collection
    7. Store vectors

    Returns:
        collection_name
    """

    collection_name = generate_collection_name(file_path)

    if collection_exists(collection_name):
        logger.info(
            f"Document already ingested: {collection_name}"
        )
        return collection_name

    logger.info(f"Starting ingestion for {file_path}")

    docs = load_pdf(file_path)

    if not docs:
        raise ValueError(
            f"No text extracted from PDF: {file_path}"
        )

    chunks = chunk_documents(docs)

    if not chunks:
        raise ValueError(
            f"No chunks generated from PDF: {file_path}"
        )

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embed_texts(texts)

    if not embeddings:
        raise ValueError(
            "Embedding generation failed"
        )

    create_collection_if_not_exists(
        collection_name=collection_name,
        vector_size=len(embeddings[0]),
    )

    add_points(
        collection_name=collection_name,
        embeddings=embeddings,
        chunks=chunks,
    )

    logger.info(
        f"Ingestion completed successfully: {collection_name}"
    )

    return collection_name
