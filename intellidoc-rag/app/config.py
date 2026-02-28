"""IntelliDoc RAG — Configuration & Settings."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- App ---
    app_name: str = "IntelliDoc RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # --- LLM ---
    llm_provider: str = Field(default="openai", description="openai or gemini")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-1.5-flash", description="Gemini model")

    # --- Embeddings ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    retrieval_top_k: int = 5
    hybrid_alpha: float = Field(
        default=0.5,
        description="Weight for vector vs BM25 (0=BM25 only, 1=vector only)",
    )

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "intellidoc"

    # --- Paths ---
    upload_dir: str = "./data/uploads"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    # Ensure directories exist
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    return settings
