from src.retrieval.vector_search import vector_search

# docs = vector_search(
#     query="What is this document about?",
#     collection_name="langchain_demo_5b8dde8f",
#     top_k=5,
# )
#
# for i, doc in enumerate(docs, start=1):
#     print(f"\n--- Result {i} ---")
#     print(f"Score: {doc.metadata['score']:.4f}")
#     print(doc.page_content[:300])

from src.retrieval.bm25 import bm25_search

docs = bm25_search(
    query="What is this document about?",
    collection_name="langchain_demo_5b8dde8f",
    top_k=5,
)

for doc in docs:
    print(doc.metadata["score"])
    print(doc.page_content[:200])
    print()