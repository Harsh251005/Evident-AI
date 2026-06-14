from src.retrieval.retriever import retrieve

docs = retrieve(
    query="What is this document about?",
    collection_name="langchain_demo_5b8dde8f"
)

for doc in docs:
    print(doc.metadata["score"])
    print(doc.page_content)
    print()