import json
import os
from pathlib import Path
from langsmith import evaluate, Client  # Added Client to handle the upload
from src.pipeline.evident_rag import EvidentAIRAG
from src.evaluation.metrics import calculate_citation_coverage, calculate_request_cost

# 1. Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT_DIR / "src" / "evaluation" / "dataset.json"
DATASET_NAME = "Claude_Constitution_Eval_v1"


def run_production_eval():
    client = Client()

    # Check for the new name
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
        print(f"✅ Found cloud dataset: {DATASET_NAME}")
    except Exception:
        print(f"📤 Uploading Claude Constitution dataset from {DATASET_PATH}...")
        with open(DATASET_PATH, "r") as f:
            examples = json.load(f)

        # This ensures we are uploading the exact local file you want
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"ground_truth": e.get("ground_truth", "")} for e in examples],
            dataset_id=dataset.id
        )
        print(f"✅ Sync complete: {DATASET_NAME} is now in the cloud.")

    rag_system = EvidentAIRAG(collection_name="claudes-constitution_webpdf_26-02.02a_09559b3b")

    def predict(inputs: dict):
        print(f"🧐 Querying: {inputs['question'][:50]}...")
        response = rag_system.chain.invoke(inputs["question"])

        # Safety: Ensure we always return a string, never None
        return response if response is not None else "Error: No response generated."

    # 5. Define the Evaluator
    def production_evaluator(run, example) -> dict:
        answer = run.outputs.get("output", "")
        # Get the question from example inputs for the cost calculation
        question = example.inputs.get("question", "")

        return {
            "results": [
                {"key": "citation_coverage", "score": calculate_citation_coverage(answer)},
                {"key": "cost_usd", "score": calculate_request_cost(question, answer)}
            ]
        }

    print(f"🚀 Running Evaluation on {DATASET_NAME}...")

    # 6. Run Evaluation using the CLOUD name now (this fixes the ID error)
    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[production_evaluator],
        experiment_prefix="prod-gate-check"
    )

    print(f"\n✅ Evaluation complete!")
    print(f"🔗 View Results: {results.url}")
    print(f"🆔 Experiment Name (COPY THIS): {results.experiment_name}")
    return results.experiment_name


if __name__ == "__main__":
    run_production_eval()