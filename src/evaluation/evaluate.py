import os
import sys
import json
import pandas as pd
from tqdm import tqdm

# Fix ModuleNotFoundError: Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Turn off tracing globally during metric initialization
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langsmith import traceable, Client, trace
# Updated Ragas imports
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Internal Imports
from config.settings import CLAUDE_CONSTITUTION_PDF_PATH, PROMPTS
from src.retrieval.retriever import retrieve
from src.retrieval.bm25 import setup_bm25
from src.generation.llm import generate_answer
from src.ingestion.vector_store import generate_collection_name
from src.evaluation.ragas_eval import evaluator_llm, evaluator_embeddings
from src.evaluation.judge import grade_with_llm

# Initialize LangSmith Client
ls_client = Client()

@traceable(name="RAG_Pipeline_Step")
def run_pipeline(question, collection_name, bm25_index):
    """Encapsulates one full RAG turn to get a single Trace ID."""
    # 1. Retrieval
    context_docs = retrieve(question, collection_name=collection_name, bm25_index=bm25_index)
    context_texts = [d["text"] for d in context_docs]

    # 2. Generation
    context_str = "\n\n".join(context_texts)
    prompt = PROMPTS['qa_system_prompt'].format(context=context_str, question=question)
    answer = generate_answer(prompt)

    return answer, context_texts

def run_eval_pipeline():
    # 1. Setup Data & Names
    target_collection = generate_collection_name(CLAUDE_CONSTITUTION_PDF_PATH)
    print(f"🚀 Starting evaluation on: {target_collection}")

    # Load BM25 index or rebuild if missing
    try:
        bm25_index = setup_bm25(target_collection)
    except ValueError:
        print("[INFO] BM25 Index missing. Running ingestion to rebuild...")
        from src.pipeline.ingestion import ingestion_pipeline
        _, _, bm25_index = ingestion_pipeline(CLAUDE_CONSTITUTION_PDF_PATH)

    with open("src/evaluation/dataset.json", "r") as f:
        test_data = json.load(f)

    # 2. Prepare Metrics
    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm = evaluator_llm
        if hasattr(m, 'embeddings'):
            m.embeddings = evaluator_embeddings

    evaluation_results = []

    # 3. Main Loop
    for i, item in enumerate(tqdm(test_data, desc="Evaluating Questions")):
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        # Turn tracing ON for the actual execution
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Corrected: Use 'trace' context manager instead of 'run_helpers.traceable'
        with trace(name="Eval_Run", run_type="chain") as rt:
            answer, contexts = run_pipeline(question, target_collection, bm25_index)
            run_id = rt.id

        os.environ["LANGCHAIN_TRACING_V2"] = "false"

        # 4. RAGAS Scoring
        from ragas import SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=answer,
            reference=ground_truth
        )

        row_scores = {"question": question, "run_id": str(run_id)}
        for metric in metrics:
            try:
                score = metric.single_turn_score(sample)
                row_scores[metric.name] = score
                ls_client.create_feedback(run_id, key=metric.name, score=score)
            except Exception as e:
                row_scores[metric.name] = 0.0

        # 5. LLM Judge Scoring
        try:
            score, response = grade_with_llm(ground_truth, answer)
            row_scores["judge_score"] = score
            row_scores["judge_critique"] = response
        except Exception as e:
            row_scores["judge_critique"] = f"Judge Error: {e}"

        evaluation_results.append(row_scores)

    # 6. Reporting
    df = pd.DataFrame(evaluation_results)
    df.to_csv("evaluation_report.csv", index=False)

    print("\n" + "=" * 30)
    print("📈 FINAL EVALUATION SUMMARY")
    print("=" * 30)
    print(df.mean(numeric_only=True))

if __name__ == "__main__":
    run_eval_pipeline()