from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer
from langsmith import traceable

@traceable(name="rag_pipeline")
def answer_query(
    query: str,
    collection_name: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
) -> str:

    context_docs = retrieve(
        query=query,
        collection_name=collection_name,
        mode=retrieval_mode,
        top_k=top_k,
    )

    answer = generate_answer(
        query=query,
        context_docs=context_docs,
    )

    return answer