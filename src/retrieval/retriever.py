from langchain_core.documents import Document
from langsmith import traceable

from config.settings import settings
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank_documents


@traceable(name="retrieval")
def retrieve(
    query: str,
    collection_name: str,
    mode: str = "hybrid",
    top_k: int = None,
    rerank: bool = True,
) -> list[Document]:

    top_k = top_k or settings.FINAL_K
    fetch_k = settings.INITIAL_K if rerank else top_k

    if mode == "vector":
        candidates = vector_search(
            query,
            collection_name,
            fetch_k,
        )

    elif mode == "bm25":
        candidates = bm25_search(
            query,
            collection_name,
            fetch_k,
        )

    elif mode == "hybrid":
        candidates = hybrid_search(
            query,
            collection_name,
            fetch_k,
        )

    else:
        raise ValueError(
            f"Unsupported retrieval mode: {mode}"
        )

    if not rerank:
        return candidates

    return rerank_documents(
        query=query,
        documents=candidates,
        top_k=top_k,
    )
