"""IntelliDoc RAG — ChromaDB Vector Store Interface."""

from __future__ import annotations

import logging
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized ChromaDB collection '{settings.chroma_collection_name}' "
            f"({self.collection.count()} existing documents)"
        )

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> list[str]:
        """Add chunks with pre-computed embeddings to the vector store.

        Returns:
            List of generated document IDs.
        """
        ids = [
            f"{chunk.source}::chunk-{chunk.chunk_index}" for chunk in chunks
        ]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **{
                    k: v
                    for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=list(embeddings),
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks to vector store")
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Search for similar chunks using vector similarity.

        Returns:
            List of dicts with keys: id, content, metadata, score
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # cosine distance → similarity
                }
            )

        return hits

    def get_document_count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def list_sources(self) -> list[str]:
        """List unique source document names."""
        all_meta = self.collection.get(include=["metadatas"])
        sources = set()
        for meta in all_meta["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a source document."""
        all_data = self.collection.get(
            where={"source": source}, include=["metadatas"]
        )
        ids = all_data["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for source '{source}'")
        return len(ids)
