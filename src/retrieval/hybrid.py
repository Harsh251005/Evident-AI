from collections import defaultdict

from langchain_core.documents import Document

from src.retrieval.vector_search import vector_search
from src.retrieval.bm25 import bm25_search
from src.utils.logger import get_logger

logger = get_logger(__name__)

RRF_K = 60

def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    top_k: int,
) -> list[Document]:

    scores = defaultdict(float)
    document_lookup = {}

    for ranking in rankings:

        for rank, doc in enumerate(ranking, start=1):

            doc_id = doc.page_content

            scores[doc_id] += 1 / (RRF_K + rank)

            document_lookup[doc_id] = doc

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        document_lookup[doc_id]
        for doc_id, _ in ranked_docs[:top_k]
    ]


def hybrid_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> list[Document]:

    vector_results = vector_search(
        query=query,
        collection_name=collection_name,
        top_k=top_k * 2,
    )

    bm25_results = bm25_search(
        query=query,
        collection_name=collection_name,
        top_k=top_k * 2,
    )

    results = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        top_k=top_k,
    )

    logger.info(
        f"Hybrid search returned {len(results)} chunks"
    )

    return results