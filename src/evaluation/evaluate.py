import json
from tqdm import tqdm

from src.retrieval.retriever import retrieve
from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer
from src.evaluation.ragas_eval import run_ragas

def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation(dataset, collection_name, bm25):
    results = []

    for item in tqdm(dataset):
        query = item["question"]

        context = retrieve(query, collection_name, bm25=bm25, k=5)

        if not context:
            generated_answer = "I don't know"
        else:
            prompt = build_prompt(context, query)
            generated_answer = generate_answer(prompt)

        results.append({
            "question": query,
            "ground_truth": item["answer"],
            "generated": generated_answer,
            "context": context
        })

    return results


def evaluate_system(dataset, collection_name, bm25):
    results = run_evaluation(dataset, collection_name, bm25)

    print("\n[INFO] Running RAGAS...")
    ragas_scores = run_ragas(results)

    print("\n--- RAGAS RESULTS ---")
    print(ragas_scores)
