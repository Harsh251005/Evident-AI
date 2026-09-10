from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.ingestion.vector_store import client, _with_retry
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Per-collection BM25 index cache. Rebuilding this on every query means a
# network scroll of the whole collection plus re-tokenizing every chunk —
# real, avoidable latency on a corpus that doesn't change between queries.
# Collections are immutable once ingested (see ingest_document), so caching
# for the lifetime of the process is safe.
_bm25_cache: dict[str, tuple[list[Document], BM25Okapi | None]] = {}


def _get_collection_documents(
    collection_name: str,
) -> list[Document]:
    """
    Fetch all documents from a Qdrant collection.
    """

    documents = []

    points, _ = _with_retry(
        client.scroll,
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload

        documents.append(
            Document(
                page_content=payload["text"],
                metadata={
                    "source": payload.get("source"),
                    "page_no": payload.get("page_no"),
                },
            )
        )

    return documents


def _get_bm25_index(
    collection_name: str,
) -> tuple[list[Document], BM25Okapi | None]:

    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    documents = _get_collection_documents(collection_name)

    bm25 = None
    if documents:
        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_docs)

    _bm25_cache[collection_name] = (documents, bm25)

    logger.info(
        f"Built BM25 index for '{collection_name}' "
        f"({len(documents)} chunks)"
    )

    return documents, bm25


def bm25_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Perform BM25 retrieval against all chunks in a collection.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    documents, bm25 = _get_bm25_index(collection_name)

    if not documents or bm25 is None:
        logger.warning(
            f"No documents found in {collection_name}"
        )
        return []

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for idx in ranked_indices:
        cached_doc = documents[idx]

        # Build a fresh Document rather than mutating the cached one in
        # place — `documents` is shared across every query against this
        # collection, so writing a per-query score onto it directly would
        # race with concurrent queries.
        results.append(
            Document(
                page_content=cached_doc.page_content,
                metadata={
                    **cached_doc.metadata,
                    "score": float(scores[idx]),
                },
            )
        )

    logger.info(
        f"BM25 retrieved {len(results)} chunks "
        f"from {collection_name}"
    )

    return results