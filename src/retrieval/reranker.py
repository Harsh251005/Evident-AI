from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ranker = Ranker(model_name=settings.RERANKER_MODEL, cache_dir="/tmp")


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = None,
) -> list[Document]:
    """
    Re-score candidate documents with a cross-encoder and return the top_k.
    """

    if not documents:
        return []

    top_k = top_k or settings.FINAL_K

    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(documents)
    ]

    results = _ranker.rerank(
        RerankRequest(query=query, passages=passages)
    )

    reranked_docs = [
        Document(
            page_content=result["text"],
            metadata={
                **result["meta"],
                "rerank_score": float(result["score"]),
            },
        )
        for result in results[:top_k]
    ]

    logger.info(
        f"Reranked {len(documents)} candidates -> top {len(reranked_docs)}"
    )

    return reranked_docs
