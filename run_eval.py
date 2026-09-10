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
import asyncio
import json
import os
import re
from pathlib import Path

from openai import AsyncOpenAI

# Loads .env into os.environ — every other entrypoint (main.py,
# run_generation.py, app.py) gets this transitively by importing
# config.settings somewhere in their chain; this script didn't, so running
# `python run_eval.py` standalone failed with a missing OPENAI_API_KEY.
import config.settings  # noqa: F401

# RAGAS native OpenAI provider (ragas.metrics.collections) — async,
# instructor-based, bypasses langchain_community entirely
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

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
# The CI quality gate checks a single composite quality_score (mean of RAGAS
# composite, LLM-judge mean, and citation coverage) against QUALITY_THRESHOLD.
QUALITY_THRESHOLD = 0.80

CITATION_PATTERN = re.compile(r"\(Page\s+(\d+)\)", re.IGNORECASE)

# How many questions to score concurrently against the RAGAS metrics.
RAGAS_CONCURRENCY = 5

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


_ZERO_SCORE = {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0,
}


async def _score_one(metrics: dict, answer: GeneratedAnswer) -> dict:
    # A question whose generation/retrieval failed upstream (empty answer or
    # no retrieved context, after generate_answers.py's own retries were
    # exhausted) has no faithfulness/relevancy to measure — RAGAS's
    # Faithfulness metric hard-raises on an empty response rather than
    # scoring it. Treat it as a real 0 for this question instead of
    # crashing the entire batch over one bad sample.
    if not answer.generated_answer or not answer.retrieved_contexts:
        print(
            f"[run_eval] WARNING: empty answer/context for "
            f"'{answer.question[:60]}...' — scoring as 0, not evaluating"
        )
        return dict(_ZERO_SCORE)

    try:
        faithfulness_result, relevancy_result, precision_result, recall_result = (
            await asyncio.gather(
                metrics["faithfulness"].ascore(
                    user_input=answer.question,
                    response=answer.generated_answer,
                    retrieved_contexts=answer.retrieved_contexts,
                ),
                metrics["answer_relevancy"].ascore(
                    user_input=answer.question,
                    response=answer.generated_answer,
                ),
                metrics["context_precision"].ascore(
                    user_input=answer.question,
                    reference=answer.ground_truth,
                    retrieved_contexts=answer.retrieved_contexts,
                ),
                metrics["context_recall"].ascore(
                    user_input=answer.question,
                    retrieved_contexts=answer.retrieved_contexts,
                    reference=answer.ground_truth,
                ),
            )
        )
    except Exception as e:
        # One question's RAGAS scoring shouldn't take down the other 49 —
        # score it 0 and keep going, same as the empty-answer case above.
        print(
            f"[run_eval] WARNING: RAGAS scoring failed for "
            f"'{answer.question[:60]}...': {e} — scoring as 0"
        )
        return dict(_ZERO_SCORE)

    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
        "context_precision": precision_result.value,
        "context_recall": recall_result.value,
    }


async def _run_ragas_async(answers: list[GeneratedAnswer]) -> list[dict]:
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4.1-mini", client=client)
    embeddings = RagasOpenAIEmbeddings(client=client, model="text-embedding-3-small")

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    semaphore = asyncio.Semaphore(RAGAS_CONCURRENCY)

    async def bounded_score(answer: GeneratedAnswer) -> dict:
        async with semaphore:
            return await _score_one(metrics, answer)

    return await asyncio.gather(*(bounded_score(a) for a in answers))


def run_ragas(answers: list[GeneratedAnswer]) -> RAGASScores:
    print("[run_eval] Running RAGAS evaluation...")

    per_question = asyncio.run(_run_ragas_async(answers))

    def _mean(key: str) -> float:
        return round(sum(r[key] for r in per_question) / len(per_question), 4)

    scores = RAGASScores(
        faithfulness      = _mean("faithfulness"),
        answer_relevancy  = _mean("answer_relevancy"),
        context_precision = _mean("context_precision"),
        context_recall    = _mean("context_recall"),
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


def compute_citation_coverage(answers: list[GeneratedAnswer]) -> float:
    """
    Fraction of answers that cite at least one page number, where every
    cited page number is actually among the retrieved pages for that answer.

    A missing citation, or a citation pointing at a page that was never
    retrieved (a fabricated reference), counts as uncovered.
    """
    print("[run_eval] Computing citation coverage...")

    covered = 0

    for a in answers:
        cited_pages = {
            int(page) for page in CITATION_PATTERN.findall(a.generated_answer)
        }

        is_covered = bool(cited_pages) and cited_pages.issubset(
            set(a.retrieved_pages)
        )

        if is_covered:
            covered += 1

    coverage = round(covered / len(answers), 4)
    print(f"[run_eval] Citation coverage: {coverage} ({covered}/{len(answers)})")
    return coverage


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

    answers           = load_generated_answers(args.answers)
    ragas_scores      = run_ragas(answers)
    judge_results     = run_llm_judge(answers)
    citation_coverage = compute_citation_coverage(answers)

    judge_mean    = round(sum(r.score for r in judge_results) / len(judge_results), 4)
    quality_score = round(
        (ragas_scores.composite + judge_mean + citation_coverage) / 3,
        4,
    )
    gate_passed = quality_score >= QUALITY_THRESHOLD

    report = EvalReport(
        ragas_scores         = ragas_scores,
        ragas_composite      = ragas_scores.composite,
        llm_judge_mean_score = judge_mean,
        llm_judge_results    = judge_results,
        citation_coverage    = citation_coverage,
        quality_score        = quality_score,
        total_samples        = len(answers),
        quality_gate_passed  = gate_passed,
        thresholds           = {
            "quality_score": QUALITY_THRESHOLD,
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
    print(f"  RAGAS Composite    : {ragas_scores.composite}")
    print(f"  LLM Judge Mean     : {judge_mean}")
    print(f"  Citation Coverage  : {citation_coverage}")
    print(f"  Quality Score      : {quality_score}  (threshold: {QUALITY_THRESHOLD})")
    print(f"  Quality Gate       : {'✅ PASSED' if gate_passed else '❌ FAILED'}")
    print("=" * 50)
    print(f"\n[run_eval] Report saved → {args.report}")

    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()