"""IntelliDoc RAG — Hybrid Search (BM25 + Vector)."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Sequence

from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.config import get_settings

logger = logging.getLogger(__name__)


class BM25:
    """Lightweight BM25 implementation for lexical retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_dl: float = 0.0
        self.df: Counter = Counter()
        self.n_docs: int = 0
        self._documents: list[dict] = []

    def index(self, documents: list[dict]) -> None:
        """Build the BM25 index from document dicts.

        Args:
            documents: List of dicts with 'content' key.
        """
        self._documents = documents
        self.corpus = []
        self.doc_len = []
        self.df = Counter()

        for doc in documents:
            tokens = self._tokenize(doc["content"])
            self.corpus.append(tokens)
            self.doc_len.append(len(tokens))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

        self.n_docs = len(self.corpus)
        self.avg_dl = sum(self.doc_len) / max(self.n_docs, 1)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using BM25 scoring."""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.corpus):
            score = self._score(query_tokens, doc_tokens, self.doc_len[i])
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []

        for idx, score in scores[:top_k]:
            result = {**self._documents[idx], "bm25_score": score}
            results.append(result)

        return results

    def _score(
        self, query_tokens: list[str], doc_tokens: list[str], doc_len: int
    ) -> float:
        tf = Counter(doc_tokens)
        score = 0.0

        for token in query_tokens:
            if token not in self.df:
                continue

            idf = math.log(
                (self.n_docs - self.df[token] + 0.5)
                / (self.df[token] + 0.5)
                + 1
            )

            token_tf = tf.get(token, 0)
            numerator = token_tf * (self.k1 + 1)
            denominator = token_tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_dl
            )
            score += idf * numerator / denominator

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()


class HybridSearchEngine:
    """Hybrid retrieval combining BM25 lexical search with vector similarity.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval
    strategies into a single ranked list.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        alpha: float | None = None,
    ):
        settings = get_settings()
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.bm25 = BM25()
        self._indexed = False

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        documents = []
        for i in range(len(all_data["ids"])):
            documents.append(
                {
                    "id": all_data["ids"][i],
                    "content": all_data["documents"][i],
                    "metadata": all_data["metadatas"][i],
                }
            )

        self.bm25.index(documents)
        self._indexed = True
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform hybrid search combining vector + BM25 results.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            Merged and re-ranked list of results.
        """
        # Vector search
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for fusion
        )

        # BM25 search
        if not self._indexed:
            self.build_bm25_index()

        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            vector_results, bm25_results, top_k
        )

        logger.info(
            f"Hybrid search for '{query[:50]}...' returned {len(fused)} results"
        )
        return fused

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict]:
        """Merge results using Reciprocal Rank Fusion (RRF).

        Score = alpha * (1 / (k + vector_rank)) + (1-alpha) * (1 / (k + bm25_rank))
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + self.alpha * (
                1.0 / (k + rank + 1)
            )
            doc_map[doc_id] = result

        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1 - self.alpha) * (
                1.0 / (k + rank + 1)
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            result = {**doc_map[doc_id], "hybrid_score": score}
            results.append(result)

        return results
