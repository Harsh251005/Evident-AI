from pdf_loader import load_pdf
from chunker import chunk_documents
from embedder import embed_texts
from vector_store import (
    generate_collection_name,
    create_collection_if_not_exists,
    add_points,
    collection_exists
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingestion_test(file_path: str) -> None:

    collection_name = generate_collection_name(file_path)

    if collection_exists(collection_name):
        logger.info(
            f"Document already ingested: {collection_name}"
        )
        return

    docs = load_pdf(file_path)

    chunks = chunk_documents(docs)

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embed_texts(texts)

    collection_name = generate_collection_name(file_path)

    create_collection_if_not_exists(
        collection_name=collection_name,
        vector_size=len(embeddings[0]),
    )

    add_points(
        collection_name=collection_name,
        embeddings=embeddings,
        chunks=chunks,
    )


ingestion_test(
    file_path=r"D:\Harsh\Code\Resume Projects\EvidentAI\data\temp_langchain_demo.pdf"
)