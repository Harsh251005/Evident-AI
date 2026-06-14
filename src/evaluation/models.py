"""
Pydantic models for the EvidentAI evaluation system.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single entry from dataset.json"""

    question: str
    ground_truth: str
    source: int = Field(..., description="Page number the QA pair originates from")


class GeneratedAnswer(BaseModel):
    """QAPair extended with RAG pipeline output — saved to generated_answers.json"""

    question: str
    ground_truth: str
    source: int
    generated_answer: str = ""
    retrieved_contexts: list[str] = Field(
        default_factory=list,
        description="Context chunks retrieved by the pipeline — required by RAGAS",
    )


class RAGASScores(BaseModel):
    """Aggregated RAGAS metric scores across the full dataset."""

    faithfulness: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_relevancy: float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    context_recall: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def composite(self) -> float:
        """Mean of all four RAGAS metrics."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        return round(sum(scores) / len(scores), 4)


class LLMJudgeResult(BaseModel):
    """LLM-as-Judge verdict for a single QA pair."""

    question: str
    generated_answer: str
    ground_truth: str
    critique: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvalReport(BaseModel):
    """Final report written to eval_report.json — consumed by the CI quality gate."""

    ragas_scores: RAGASScores
    ragas_composite: float
    llm_judge_mean_score: float
    llm_judge_results: list[LLMJudgeResult]
    total_samples: int
    quality_gate_passed: bool = False
    thresholds: dict = Field(
        default_factory=lambda: {
            "ragas_composite": 0.70,
            "llm_judge_mean": 0.75,
        }
    )
    metadata: Optional[dict] = None