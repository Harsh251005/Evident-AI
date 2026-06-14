"""
CLI entrypoint to run answer generation over the evaluation dataset.

Usage:
    python -m src.evaluation.run_generation
    python -m src.evaluation.run_generation --mode hybrid --top-k 5
"""

import argparse
from src.evaluation.generate_answers import generate_answers
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Generate RAG answers for the eval dataset.")
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.QDRANT_COLLECTION_NAME,
        help="Qdrant collection name (default: from settings)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "sparse"],
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    args = parser.parse_args()

    print(f"[run_generation] Collection : {args.collection}")
    print(f"[run_generation] Mode       : {args.mode}")
    print(f"[run_generation] Top-K      : {args.top_k}")
    print("[run_generation] Starting answer generation...\n")

    results = generate_answers(
        collection_name=args.collection,
        retrieval_mode=args.mode,
        top_k=args.top_k,
    )

    print(f"\n[run_generation] Done. {len(results)} answers generated.")


if __name__ == "__main__":
    main()