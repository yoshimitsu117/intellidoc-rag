"""Tests for IntelliDoc RAG pipeline."""

import pytest
from app.ingestion.loader import DocumentLoader, Document
from app.ingestion.chunker import RecursiveChunker, Chunk


class TestDocumentLoader:
    """Tests for the document loader."""

    def test_supported_extensions(self):
        loader = DocumentLoader()
        assert ".pdf" in loader.SUPPORTED_EXTENSIONS
        assert ".txt" in loader.SUPPORTED_EXTENSIONS
        assert ".md" in loader.SUPPORTED_EXTENSIONS

    def test_unsupported_extension_raises(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file("test.docx")

    def test_missing_file_raises(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.txt")

    def test_document_source_property(self):
        doc = Document(
            content="Hello world",
            metadata={"source": "test.txt"},
        )
        assert doc.source == "test.txt"

    def test_document_default_source(self):
        doc = Document(content="Hello")
        assert doc.source == "unknown"


class TestRecursiveChunker:
    """Tests for the recursive text chunker."""

    def test_basic_chunking(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(
            content="This is a test document. " * 20,
            metadata={"source": "test.txt"},
        )
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_small_document_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        doc = Document(
            content="Short document.",
            metadata={"source": "short.txt"},
        )
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "Short document."

    def test_chunk_metadata_preserved(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(
            content="A " * 100,
            metadata={"source": "meta.txt", "file_type": ".txt"},
        )
        chunks = chunker.chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "meta.txt"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_empty_document(self):
        chunker = RecursiveChunker()
        doc = Document(content="", metadata={"source": "empty.txt"})
        chunks = chunker.chunk_documents([doc])
        assert len(chunks) == 0

    def test_chunk_index_sequential(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        doc = Document(
            content="Word " * 100,
            metadata={"source": "test.txt"},
        )
        chunks = chunker.chunk_documents([doc])
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestBM25:
    """Tests for the BM25 retrieval engine."""

    def test_basic_search(self):
        from app.retrieval.hybrid_search import BM25

        bm25 = BM25()
        documents = [
            {"id": "1", "content": "Python is a programming language"},
            {"id": "2", "content": "Java is also a programming language"},
            {"id": "3", "content": "Cooking recipes for dinner"},
        ]
        bm25.index(documents)

        results = bm25.search("Python programming", top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "1"  # Python doc should rank first

    def test_empty_corpus(self):
        from app.retrieval.hybrid_search import BM25

        bm25 = BM25()
        bm25.index([])
        results = bm25.search("test query")
        assert results == []

    def test_bm25_scoring(self):
        from app.retrieval.hybrid_search import BM25

        bm25 = BM25()
        documents = [
            {"id": "1", "content": "machine learning deep learning"},
            {"id": "2", "content": "web development frontend backend"},
        ]
        bm25.index(documents)

        results = bm25.search("machine learning")
        assert results[0]["id"] == "1"
        assert results[0]["bm25_score"] > 0
