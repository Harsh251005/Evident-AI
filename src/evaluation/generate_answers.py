import json
from pathlib import Path

from src.evaluation.dataset_loader import load_dataset
from src.evaluation.models import GeneratedAnswer
from src.retrieval.retriever import retrieve
from src.generation.generator import generate_answer


OUTPUT_PATH = Path(__file__).parent / "generated_answers.json"


def generate_answers(
    collection_name: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    output_path: Path = OUTPUT_PATH,
) -> list[GeneratedAnswer]:
    """
    For each QA pair in the dataset:
      1. Retrieve context docs from the vector store.
      2. Generate an answer using the retrieved context.
      3. Store both alongside the original question + ground truth.

    Args:
        collection_name: Qdrant collection to retrieve from.
        retrieval_mode:   "hybrid", "dense", or "sparse".
        top_k:            Number of chunks to retrieve per question.
        output_path:      Where to write generated_answers.json.

    Returns:
        List of GeneratedAnswer objects.
    """
    dataset = load_dataset()
    results: list[GeneratedAnswer] = []

    for i, qa in enumerate(dataset, start=1):
        print(f"[generate_answers] ({i}/{len(dataset)}) {qa.question[:70]}...")

        try:
            # Retrieve — gives us List[Document]
            context_docs = retrieve(
                query=qa.question,
                collection_name=collection_name,
                mode=retrieval_mode,
                top_k=top_k,
            )

            # Generate — same call your pipeline makes
            answer = generate_answer(
                query=qa.question,
                context_docs=context_docs,
            )

            # Extract page_content from each Document for RAGAS
            retrieved_contexts = [doc.page_content for doc in context_docs]

        except Exception as e:
            print(f"[generate_answers] ERROR on question {i}: {e}")
            answer = ""
            retrieved_contexts = []

        results.append(
            GeneratedAnswer(
                question=qa.question,
                ground_truth=qa.ground_truth,
                source=qa.source,
                generated_answer=answer,
                retrieved_contexts=retrieved_contexts,
            )
        )

    # Persist to disk — run_eval.py will load this
    output_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[generate_answers] Saved {len(results)} results → {output_path}")

    return results