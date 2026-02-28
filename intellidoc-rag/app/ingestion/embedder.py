"""IntelliDoc RAG — Embedding Generation."""

from __future__ import annotations

import logging
from typing import Sequence

import openai

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using OpenAI's embedding API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # OpenAI supports batch embedding — process in chunks of 2048
        batch_size = 2048
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            # Clean empty strings
            batch = [t if t.strip() else " " for t in batch]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Embedded batch {i // batch_size + 1} "
                f"({len(batch)} texts, model={self.model})"
            )

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string."""
        result = self.embed_texts([query])
        return result[0]
