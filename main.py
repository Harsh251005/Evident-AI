import argparse

from src.pipeline.ingestion import ingest_document
from src.pipeline.rag_pipeline import answer_query


def main():
    parser = argparse.ArgumentParser(
        description="EvidentAI — ingest a PDF and optionally ask it a question."
    )
    parser.add_argument("file_path", help="Path to the PDF to ingest")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only ingest the document, skip the question prompt",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a question against the ingested document",
    )
    args = parser.parse_args()

    print(f"[main] Ingesting {args.file_path} ...")
    collection_name = ingest_document(args.file_path)
    print(f"[main] Ready. Collection: {collection_name}")

    if args.ingest_only:
        return

    query = args.query or "What is this document about?"
    print(f"[main] Query: {query}")

    answer = answer_query(query=query, collection_name=collection_name)
    print(f"[main] Answer:\n{answer}")


if __name__ == "__main__":
    main()
