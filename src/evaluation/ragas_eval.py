from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas_embedder import RagasEmbeddingWrapper

embedding_model = LangchainEmbeddingsWrapper(RagasEmbeddingWrapper())


def run_ragas(results):
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for r in results:
        data["question"].append(r["question"])
        data["answer"].append(r["generated"])
        data["contexts"].append([c["text"] for c in r["context"]])
        data["ground_truth"].append(r["ground_truth"])

    dataset = Dataset.from_dict(data)

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        embeddings=embedding_model,
    )

    return scores