"""IntelliDoc RAG — RAG Chain Orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator

from app.generation.llm_client import LLMClient
from app.generation.prompts import format_context, build_rag_messages
from app.retrieval.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Structured RAG response with answer and sources."""

    answer: str
    sources: list[dict]
    query: str


class RAGChain:
    """End-to-end RAG chain: retrieve → augment → generate."""

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        llm_client: LLMClient,
        top_k: int = 5,
    ):
        self.search_engine = search_engine
        self.llm_client = llm_client
        self.top_k = top_k

    def query(self, question: str) -> RAGResponse:
        """Execute the full RAG pipeline synchronously.

        Args:
            question: User's natural language question.

        Returns:
            RAGResponse with answer and source citations.
        """
        # 1. Retrieve relevant chunks
        results = self.search_engine.search(question, top_k=self.top_k)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer your question. "
                "Please make sure documents have been ingested first.",
                sources=[],
                query=question,
            )

        # 2. Format context
        context = format_context(results)

        # 3. Build prompt & generate
        messages = build_rag_messages(question, context)
        answer = self.llm_client.generate(messages)

        # 4. Extract source metadata
        sources = [
            {
                "source": r.get("metadata", {}).get("source", "Unknown"),
                "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
                "score": r.get("hybrid_score", r.get("score", 0)),
            }
            for r in results
        ]

        logger.info(
            f"RAG query completed: '{question[:50]}...' → "
            f"{len(results)} sources, {len(answer)} chars"
        )

        return RAGResponse(answer=answer, sources=sources, query=question)

    async def aquery(self, question: str) -> RAGResponse:
        """Execute the full RAG pipeline asynchronously."""
        results = self.search_engine.search(question, top_k=self.top_k)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant documents to answer your question.",
                sources=[],
                query=question,
            )

        context = format_context(results)
        messages = build_rag_messages(question, context)
        answer = await self.llm_client.agenerate(messages)

        sources = [
            {
                "source": r.get("metadata", {}).get("source", "Unknown"),
                "chunk_index": r.get("metadata", {}).get("chunk_index", 0),
                "score": r.get("hybrid_score", r.get("score", 0)),
            }
            for r in results
        ]

        return RAGResponse(answer=answer, sources=sources, query=question)

    async def astream(self, question: str) -> AsyncIterator[str]:
        """Stream the RAG response token by token.

        Yields:
            Answer tokens as they are generated.
        """
        results = self.search_engine.search(question, top_k=self.top_k)

        if not results:
            yield "I couldn't find any relevant documents to answer your question."
            return

        context = format_context(results)
        messages = build_rag_messages(question, context)

        async for token in self.llm_client.astream(messages):
            yield token
