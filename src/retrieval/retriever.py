from langchain_core.documents import Document
from langsmith import traceable

from src.retrieval.vector_search import vector_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search


@traceable(name="retrieval")
def retrieve(
    query: str,
    collection_name: str,
    mode: str = "hybrid",
    top_k: int = 5,
) -> list[Document]:

    if mode == "vector":
        return vector_search(
            query,
            collection_name,
            top_k,
        )

    if mode == "bm25":
        return bm25_search(
            query,
            collection_name,
            top_k,
        )

    if mode == "hybrid":
        return hybrid_search(
            query,
            collection_name,
            top_k,
        )

    raise ValueError(
        f"Unsupported retrieval mode: {mode}"
    )
