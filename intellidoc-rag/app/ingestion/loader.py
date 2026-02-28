"""IntelliDoc RAG — Document Loaders.

Supports PDF, plain text, and Markdown files.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import BinaryIO

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class DocumentLoader:
    """Unified document loader supporting multiple file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a document from a file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        content = self._read_file(path, ext)
        logger.info(f"Loaded document: {path.name} ({len(content)} chars)")

        return [
            Document(
                content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": ext,
                    "file_size": path.stat().st_size,
                },
            )
        ]

    def load_bytes(
        self, file_bytes: BinaryIO, filename: str
    ) -> list[Document]:
        """Load a document from uploaded bytes."""
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        raw = file_bytes.read()

        if ext == ".pdf":
            content = self._parse_pdf_bytes(raw)
        else:
            content = raw.decode("utf-8", errors="replace")

        logger.info(f"Loaded upload: {filename} ({len(content)} chars)")

        return [
            Document(
                content=content,
                metadata={
                    "source": filename,
                    "file_type": ext,
                    "file_size": len(raw),
                },
            )
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: Path, ext: str) -> str:
        if ext == ".pdf":
            return self._parse_pdf_file(path)
        return path.read_text(encoding="utf-8", errors="replace")

    def _parse_pdf_file(self, path: Path) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF support. "
                "Install it with: pip install PyPDF2"
            )

    def _parse_pdf_bytes(self, raw: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF support.")
