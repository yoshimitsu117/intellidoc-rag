"""IntelliDoc RAG — Evaluation Pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.metrics import EvaluationMetrics, MetricResult
from app.generation.llm_client import LLMClient
from app.generation.chain import RAGChain

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """Single evaluation sample."""

    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)

    # Filled after evaluation
    generated_answer: str = ""
    metrics: dict[str, MetricResult] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Full evaluation report across all samples."""

    samples: list[EvalSample]
    aggregate_scores: dict[str, float]
    total_samples: int

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "aggregate_scores": self.aggregate_scores,
            "samples": [
                {
                    "question": s.question,
                    "ground_truth": s.ground_truth,
                    "generated_answer": s.generated_answer,
                    "metrics": {
                        name: {"score": m.score, "reasoning": m.reasoning}
                        for name, m in s.metrics.items()
                    },
                }
                for s in self.samples
            ],
        }


class Evaluator:
    """End-to-end RAG evaluation pipeline.

    Loads a test dataset, runs the RAG pipeline on each question,
    and evaluates the results using RAGAS-style metrics.
    """

    def __init__(self, rag_chain: RAGChain, llm_client: LLMClient):
        self.rag_chain = rag_chain
        self.metrics = EvaluationMetrics(llm_client)

    def load_dataset(self, path: str | Path) -> list[EvalSample]:
        """Load evaluation dataset from JSON file.

        Expected format:
        [
            {"question": "...", "ground_truth": "..."},
            ...
        ]
        """
        data = json.loads(Path(path).read_text())
        samples = []
        for item in data:
            samples.append(
                EvalSample(
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                )
            )
        logger.info(f"Loaded {len(samples)} evaluation samples from {path}")
        return samples

    def evaluate(self, samples: list[EvalSample]) -> EvalReport:
        """Run full evaluation on a set of samples.

        For each sample:
        1. Query the RAG pipeline
        2. Compute all metrics
        3. Aggregate scores
        """
        for i, sample in enumerate(samples):
            logger.info(
                f"Evaluating sample {i + 1}/{len(samples)}: "
                f"'{sample.question[:50]}...'"
            )

            # Generate answer using RAG
            response = self.rag_chain.query(sample.question)
            sample.generated_answer = response.answer
            sample.contexts = [s.get("source", "") for s in response.sources]

            # Build context string for metrics
            context_str = response.answer  # Use retrieval context here

            # Compute metrics
            sample.metrics["faithfulness"] = self.metrics.faithfulness(
                question=sample.question,
                answer=sample.generated_answer,
                context=context_str,
            )
            sample.metrics["answer_relevance"] = self.metrics.answer_relevance(
                question=sample.question,
                answer=sample.generated_answer,
                context=context_str,
            )
            sample.metrics["context_precision"] = self.metrics.context_precision(
                question=sample.question,
                context=context_str,
                ground_truth=sample.ground_truth,
            )
            sample.metrics["context_recall"] = self.metrics.context_recall(
                question=sample.question,
                context=context_str,
                ground_truth=sample.ground_truth,
            )

        # Aggregate scores
        metric_names = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        aggregate = {}
        for name in metric_names:
            scores = [s.metrics[name].score for s in samples if name in s.metrics]
            aggregate[name] = sum(scores) / max(len(scores), 1)

        report = EvalReport(
            samples=samples,
            aggregate_scores=aggregate,
            total_samples=len(samples),
        )

        logger.info(f"Evaluation complete: {aggregate}")
        return report
