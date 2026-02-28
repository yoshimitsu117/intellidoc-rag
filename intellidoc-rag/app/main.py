"""IntelliDoc RAG — FastAPI Application.

Production-grade RAG Document Q&A System with hybrid search,
streaming responses, and evaluation pipeline.
"""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid_search import HybridSearchEngine
from app.generation.llm_client import LLMClient
from app.generation.chain import RAGChain

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global components (initialized on startup)
# --------------------------------------------------------------------------- #
vector_store: VectorStore | None = None
search_engine: HybridSearchEngine | None = None
rag_chain: RAGChain | None = None
embedder: Embedder | None = None
loader: DocumentLoader | None = None
chunker: RecursiveChunker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global vector_store, search_engine, rag_chain, embedder, loader, chunker

    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize components
    loader = DocumentLoader()
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = Embedder()
    vector_store = VectorStore()

    search_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedder=embedder,
        alpha=settings.hybrid_alpha,
    )

    llm_client = LLMClient()
    rag_chain = RAGChain(
        search_engine=search_engine,
        llm_client=llm_client,
        top_k=settings.retrieval_top_k,
    )

    logger.info("All components initialized successfully")
    yield
    logger.info("Shutting down IntelliDoc RAG")


# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="IntelliDoc RAG API",
    description="Production-grade RAG Document Q&A System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int


class DocumentInfo(BaseModel):
    source: str


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "documents": vector_store.get_document_count() if vector_store else 0,
    }


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    """Upload and ingest documents into the RAG system."""
    settings = get_settings()
    total_chunks = 0
    docs_processed = 0

    for file in files:
        if not file.filename:
            continue

        try:
            # Load document
            documents = loader.load_bytes(file.file, file.filename)

            # Chunk documents
            chunks = chunker.chunk_documents(documents)

            # Generate embeddings
            chunk_texts = [c.content for c in chunks]
            chunk_embeddings = embedder.embed_texts(chunk_texts)

            # Store in vector database
            vector_store.add_chunks(chunks, chunk_embeddings)

            # Rebuild BM25 index
            search_engine.build_bm25_index()

            total_chunks += len(chunks)
            docs_processed += 1

            # Save uploaded file
            upload_path = Path(settings.upload_dir) / file.filename
            file.file.seek(0)
            with open(upload_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            logger.info(
                f"Ingested '{file.filename}': {len(chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to ingest '{file.filename}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to ingest '{file.filename}': {str(e)}",
            )

    return IngestResponse(
        message=f"Successfully ingested {docs_processed} document(s)",
        documents_processed=docs_processed,
        chunks_created=total_chunks,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents and get a response with sources."""
    if not vector_store or vector_store.get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Please upload documents first.",
        )

    try:
        response = await rag_chain.aquery(request.question)
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream")
async def query_stream(request: QueryRequest):
    """Query documents with streaming response (SSE)."""
    if not vector_store or vector_store.get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested.",
        )

    async def event_stream() -> AsyncIterator[str]:
        async for token in rag_chain.astream(request.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/v1/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    sources = vector_store.list_sources() if vector_store else []
    return [DocumentInfo(source=s) for s in sources]


@app.post("/api/v1/evaluate")
async def run_evaluation(test_dataset: str = "eval_data.json"):
    """Run evaluation on a test dataset."""
    from app.evaluation.evaluator import Evaluator

    try:
        llm_client = LLMClient()
        evaluator = Evaluator(rag_chain=rag_chain, llm_client=llm_client)
        samples = evaluator.load_dataset(test_dataset)
        report = evaluator.evaluate(samples)
        return report.to_dict()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation dataset '{test_dataset}' not found.",
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
