"""
Master evaluation runner for EvidentAI.

What this does:
  1. Loads generated_answers.json (produced by run_generation.py)
  2. Runs RAGAS metrics  — faithfulness, answer_relevancy, context_precision, context_recall
  3. Runs LLM-as-Judge   — GPT-4.1-mini critiques each answer against ground truth
  4. Writes eval_report.json — consumed by the CI quality gate

Usage:
    python run_eval.py
    python run_eval.py --answers src/evaluation/generated_answers.json
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# RAGAS native OpenAI client — bypasses langchain_community entirely
from ragas.llms import LlmProvider
from ragas.embeddings import EmbeddingProvider

from langchain_openai import ChatOpenAI

from src.evaluation.models import (
    EvalReport,
    GeneratedAnswer,
    LLMJudgeResult,
    RAGASScores,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ANSWERS_PATH = Path("src/evaluation/generated_answers.json")
REPORT_PATH  = Path("eval_report.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
RAGAS_THRESHOLD     = 0.70
LLM_JUDGE_THRESHOLD = 0.75

# ── LLM-as-Judge prompt ────────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Given a question, a generated answer, and the ground truth answer, evaluate the quality
of the generated answer on the following criteria:
  - Factual accuracy compared to the ground truth
  - Completeness — does it cover the key points?
  - Conciseness — no hallucinated or irrelevant content

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{
  "score": <float between 0.0 and 1.0>,
  "critique": "<one or two sentence explanation>"
}}

Question:         {question}
Generated Answer: {generated_answer}
Ground Truth:     {ground_truth}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_generated_answers(path: Path) -> list[GeneratedAnswer]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GeneratedAnswer(**entry) for entry in raw]


def build_ragas_dataset(answers: list[GeneratedAnswer]) -> Dataset:
    return Dataset.from_dict({
        "question":     [a.question for a in answers],
        "answer":       [a.generated_answer for a in answers],
        "contexts":     [a.retrieved_contexts for a in answers],
        "ground_truth": [a.ground_truth for a in answers],
    })


def run_ragas(answers: list[GeneratedAnswer]) -> RAGASScores:
    print("[run_eval] Running RAGAS evaluation...")

    # Native RAGAS OpenAI provider — no langchain_community dependency
    ragas_llm        = LlmProvider.openai(model="gpt-4.1-mini")
    ragas_embeddings = EmbeddingProvider.openai(model="text-embedding-3-small")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings

    dataset = build_ragas_dataset(answers)
    result  = evaluate(dataset, metrics=metrics)
    df      = result.to_pandas()

    scores = RAGASScores(
        faithfulness      = round(float(df["faithfulness"].mean()), 4),
        answer_relevancy  = round(float(df["answer_relevancy"].mean()), 4),
        context_precision = round(float(df["context_precision"].mean()), 4),
        context_recall    = round(float(df["context_recall"].mean()), 4),
    )

    print(f"[run_eval] RAGAS composite: {scores.composite}")
    return scores


def run_llm_judge(answers: list[GeneratedAnswer]) -> list[LLMJudgeResult]:
    print("[run_eval] Running LLM-as-Judge evaluation...")

    # Use LangChain's ChatOpenAI directly — no RAGAS wrappers needed here
    llm     = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    results : list[LLMJudgeResult] = []

    for i, a in enumerate(answers, start=1):
        print(f"[run_eval] Judging ({i}/{len(answers)}) {a.question[:60]}...")

        prompt = JUDGE_PROMPT.format(
            question=a.question,
            generated_answer=a.generated_answer,
            ground_truth=a.ground_truth,
        )

        try:
            response = llm.invoke(prompt)
            raw      = response.content.strip()
            # Strip markdown fences if GPT wraps response anyway
            raw      = raw.replace("```json", "").replace("```", "").strip()
            parsed   = json.loads(raw)
            score    = float(parsed["score"])
            critique = str(parsed["critique"])
        except Exception as e:
            print(f"[run_eval] Judge parse error on question {i}: {e}")
            score    = 0.0
            critique = "Evaluation failed."

        results.append(
            LLMJudgeResult(
                question=a.question,
                generated_answer=a.generated_answer,
                ground_truth=a.ground_truth,
                score=score,
                critique=critique,
            )
        )

    mean = round(sum(r.score for r in results) / len(results), 4)
    print(f"[run_eval] LLM-Judge mean score: {mean}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS + LLM-as-Judge evaluation.")
    parser.add_argument("--answers", type=Path, default=ANSWERS_PATH)
    parser.add_argument("--report",  type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not args.answers.exists():
        raise FileNotFoundError(
            f"Generated answers not found at {args.answers}. "
            "Run `python -m src.evaluation.run_generation` first."
        )

    answers       = load_generated_answers(args.answers)
    ragas_scores  = run_ragas(answers)
    judge_results = run_llm_judge(answers)

    judge_mean  = round(sum(r.score for r in judge_results) / len(judge_results), 4)
    gate_passed = (
        ragas_scores.composite >= RAGAS_THRESHOLD
        and judge_mean         >= LLM_JUDGE_THRESHOLD
    )

    report = EvalReport(
        ragas_scores         = ragas_scores,
        ragas_composite      = ragas_scores.composite,
        llm_judge_mean_score = judge_mean,
        llm_judge_results    = judge_results,
        total_samples        = len(answers),
        quality_gate_passed  = gate_passed,
        thresholds           = {
            "ragas_composite": RAGAS_THRESHOLD,
            "llm_judge_mean":  LLM_JUDGE_THRESHOLD,
        },
        metadata={
            "model":        "gpt-4.1-mini",
            "answers_file": str(args.answers),
        },
    )

    args.report.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVAL REPORT SUMMARY")
    print("=" * 50)
    print(f"  Faithfulness       : {ragas_scores.faithfulness}")
    print(f"  Answer Relevancy   : {ragas_scores.answer_relevancy}")
    print(f"  Context Precision  : {ragas_scores.context_precision}")
    print(f"  Context Recall     : {ragas_scores.context_recall}")
    print(f"  RAGAS Composite    : {ragas_scores.composite}  (threshold: {RAGAS_THRESHOLD})")
    print(f"  LLM Judge Mean     : {judge_mean}  (threshold: {LLM_JUDGE_THRESHOLD})")
    print(f"  Quality Gate       : {'✅ PASSED' if gate_passed else '❌ FAILED'}")
    print("=" * 50)
    print(f"\n[run_eval] Report saved → {args.report}")

    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()