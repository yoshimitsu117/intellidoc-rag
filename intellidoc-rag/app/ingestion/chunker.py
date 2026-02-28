"""IntelliDoc RAG — Text Chunking Strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from app.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata and position tracking."""

    content: str
    metadata: dict
    chunk_index: int
    start_char: int
    end_char: int

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class RecursiveChunker:
    """Recursive character text splitter with configurable separators.

    Splits text hierarchically using multiple separators, preferring
    natural boundaries (paragraphs > sentences > words > characters).
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split a list of documents into chunks."""
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunks = self._split_text(doc.content)
            for i, (text, start, end) in enumerate(chunks):
                chunk = Chunk(
                    content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    },
                    chunk_index=i,
                    start_char=start,
                    end_char=end,
                )
                all_chunks.append(chunk)

            logger.info(
                f"Chunked '{doc.source}' into {len(chunks)} chunks "
                f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
            )

        return all_chunks

    def _split_text(
        self, text: str, separators: list[str] | None = None
    ) -> list[tuple[str, int, int]]:
        """Recursively split text using hierarchical separators.

        Returns list of (chunk_text, start_char, end_char).
        """
        if not text:
            return []

        separators = separators or self.separators
        final_chunks: list[tuple[str, int, int]] = []

        # Find the best separator
        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split by the chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge small splits into chunks
        current_chunk: list[str] = []
        current_length = 0
        char_offset = 0
        chunk_start = 0

        for i, split in enumerate(splits):
            split_len = len(split) + (len(separator) if i > 0 else 0)

            if current_length + split_len > self.chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                if len(merged) > self.chunk_size and new_separators:
                    # Recursively split oversized chunks
                    sub_chunks = self._split_text(merged, new_separators)
                    for sc_text, sc_start, sc_end in sub_chunks:
                        final_chunks.append(
                            (sc_text, chunk_start + sc_start, chunk_start + sc_end)
                        )
                else:
                    final_chunks.append(
                        (merged, chunk_start, chunk_start + len(merged))
                    )

                # Handle overlap
                overlap_texts: list[str] = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) > self.chunk_overlap:
                        break
                    overlap_texts.insert(0, prev)
                    overlap_len += len(prev) + len(separator)

                current_chunk = overlap_texts
                current_length = sum(len(t) for t in current_chunk) + len(separator) * max(0, len(current_chunk) - 1)
                chunk_start = char_offset - current_length

            current_chunk.append(split)
            current_length += split_len
            char_offset += len(split) + len(separator)

        # Add remaining text
        if current_chunk:
            merged = separator.join(current_chunk)
            final_chunks.append((merged, chunk_start, chunk_start + len(merged)))

        return final_chunks
