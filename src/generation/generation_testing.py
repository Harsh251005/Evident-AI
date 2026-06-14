from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer

query = "Why was langchain created? and what are the best practices?"

docs = retrieve(
    query=query,
    collection_name="langchain_demo_5b8dde8f",
    mode="hybrid",
    top_k=5,
)

answer = generate_answer(
    query=query,
    context_docs=docs,
)

print(answer)