"""IntelliDoc RAG — Evaluation Metrics (RAGAS-inspired)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.generation.llm_client import LLMClient
from app.generation.prompts import EVALUATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    metric: str
    score: float
    reasoning: str


class EvaluationMetrics:
    """RAGAS-inspired evaluation metrics using LLM-as-judge."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def faithfulness(
        self, question: str, answer: str, context: str
    ) -> MetricResult:
        """Measure if the answer is grounded in the provided context.

        Score: 0.0 (hallucinated) to 1.0 (fully grounded).
        """
        return self._evaluate(
            question=question,
            answer=answer,
            context=context,
            ground_truth="N/A",
            criterion=(
                "Faithfulness: Is every claim in the answer supported by the context? "
                "Score 1.0 if fully grounded, 0.0 if fabricated."
            ),
            metric_name="faithfulness",
        )

    def answer_relevance(
        self, question: str, answer: str, context: str
    ) -> MetricResult:
        """Measure if the answer actually addresses the question.

        Score: 0.0 (irrelevant) to 1.0 (perfectly relevant).
        """
        return self._evaluate(
            question=question,
            answer=answer,
            context=context,
            ground_truth="N/A",
            criterion=(
                "Answer Relevance: Does the answer directly address the question? "
                "Score 1.0 if fully relevant, 0.0 if off-topic."
            ),
            metric_name="answer_relevance",
        )

    def context_precision(
        self, question: str, context: str, ground_truth: str
    ) -> MetricResult:
        """Measure if retrieved context is relevant to answering the question.

        Score: 0.0 (irrelevant context) to 1.0 (all context is relevant).
        """
        return self._evaluate(
            question=question,
            answer="N/A",
            context=context,
            ground_truth=ground_truth,
            criterion=(
                "Context Precision: Are the retrieved documents relevant to the question? "
                "Score 1.0 if all context is useful, 0.0 if none is relevant."
            ),
            metric_name="context_precision",
        )

    def context_recall(
        self, question: str, context: str, ground_truth: str
    ) -> MetricResult:
        """Measure if the context contains all info needed for the ground truth answer.

        Score: 0.0 (missing info) to 1.0 (complete coverage).
        """
        return self._evaluate(
            question=question,
            answer="N/A",
            context=context,
            ground_truth=ground_truth,
            criterion=(
                "Context Recall: Does the context contain all the information needed "
                "to produce the ground truth answer? Score 1.0 if complete, 0.0 if missing."
            ),
            metric_name="context_recall",
        )

    def _evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        criterion: str,
        metric_name: str,
    ) -> MetricResult:
        """Run LLM-as-judge evaluation for a single metric."""
        prompt = EVALUATION_PROMPT.format(
            context=context[:3000],  # Truncate for token limits
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            criterion=criterion,
        )

        try:
            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            # Parse JSON response
            result = json.loads(response)
            return MetricResult(
                metric=metric_name,
                score=float(result.get("score", 0.0)),
                reasoning=result.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return MetricResult(
                metric=metric_name,
                score=0.0,
                reasoning=f"Evaluation parsing error: {e}",
            )
