[README.md](https://github.com/user-attachments/files/25622435/README.md)# 📚 IntelliDoc RAG

**Production-Grade Retrieval-Augmented Generation for Document Q&A**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge)

---

## 🧠 Overview

IntelliDoc RAG is a **production-ready** document question-answering system built on Retrieval-Augmented Generation. It ingests documents (PDF, TXT, Markdown), chunks and embeds them into a vector store, and uses hybrid retrieval (BM25 + semantic search) to answer user queries with **cited, grounded responses**.

Unlike demo RAG systems, IntelliDoc is engineered for **reliability and evaluation** — including a built-in RAGAS-style evaluation pipeline to measure retrieval quality, faithfulness, and answer relevance.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│                    (Streaming + REST API)                     │
├──────────────┬──────────────────┬────────────────────────────┤
│  Ingestion   │    Retrieval     │        Generation          │
│  Pipeline    │    Engine        │        Pipeline            │
│              │                  │                            │
│ ┌──────────┐ │ ┌──────────────┐ │ ┌────────────────────────┐ │
│ │  Loader  │ │ │ Vector Store │ │ │   LLM Client           │ │
│ │ (PDF/TXT)│ │ │ (ChromaDB)   │ │ │   (OpenAI / Gemini)    │ │
│ └────┬─────┘ │ └──────┬───────┘ │ └──────────┬─────────────┘ │
│      │       │        │         │            │               │
│ ┌────▼─────┐ │ ┌──────▼───────┐ │ ┌──────────▼─────────────┐ │
│ │ Chunker  │ │ │Hybrid Search │ │ │  Prompt Templates      │ │
│ │(Recursive)│ │ │(BM25+Vector) │ │ │  + RAG Chain           │ │
│ └────┬─────┘ │ └──────────────┘ │ └────────────────────────┘ │
│      │       │                  │                            │
│ ┌────▼─────┐ │                  │  ┌───────────────────────┐ │
│ │ Embedder │ │                  │  │  Evaluation Pipeline  │ │
│ │(OpenAI)  │ │                  │  │  (RAGAS-style metrics) │ │
│ └──────────┘ │                  │  └───────────────────────┘ │
└──────────────┴──────────────────┴────────────────────────────┘
```

---

## ✨ Features

- **Multi-format Ingestion** — PDF, TXT, and Markdown document support
- **Smart Chunking** — Recursive character text splitting with configurable overlap
- **Hybrid Retrieval** — BM25 lexical search + semantic vector search with score fusion
- **Streaming Responses** — Server-Sent Events for real-time answer streaming
- **Source Citations** — Every answer includes source document references
- **Evaluation Pipeline** — Built-in RAGAS-style metrics (faithfulness, relevance, context precision)
- **Multi-LLM Support** — OpenAI GPT and Google Gemini backends
- **Production-Ready** — Docker, health checks, structured logging, error handling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (or Google Gemini API key)

### 1. Clone & Install

```bash
git clone https://github.com/yoshimitsu117/intellidoc-rag.git
cd intellidoc-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Ingest Documents

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "files=@document.pdf" \
  -H "Content-Type: multipart/form-data"
```

### 5. Ask Questions

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in the document?"}'
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest` | Upload and ingest documents |
| `POST` | `/api/v1/query` | Query documents (JSON response) |
| `POST` | `/api/v1/query/stream` | Query with streaming response (SSE) |
| `POST` | `/api/v1/evaluate` | Run evaluation on test dataset |
| `GET`  | `/api/v1/documents` | List ingested documents |
| `GET`  | `/health` | Health check |

---

## 📊 Evaluation Metrics

The built-in evaluation pipeline measures:

| Metric | Description |
|--------|-------------|
| **Context Precision** | Are the retrieved documents relevant? |
| **Context Recall** | Were all relevant documents retrieved? |
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevance** | Does the answer address the question? |

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"test_dataset": "eval_data.json"}'
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **Vector Store** | ChromaDB |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **LLM** | OpenAI GPT-4o / Google Gemini |
| **Chunking** | Recursive Character Text Splitter |
| **Retrieval** | Hybrid (BM25 + Cosine Similarity) |
| **Evaluation** | Custom RAGAS-inspired pipeline |
| **Containerization** | Docker + Docker Compose |

---

## 📁 Project Structure

```
intellidoc-rag/
├── app/
│   ├── main.py              # FastAPI application & routes
│   ├── config.py             # Settings & environment config
│   ├── ingestion/
│   │   ├── loader.py         # Document loaders (PDF, TXT, MD)
│   │   ├── chunker.py        # Text chunking strategies
│   │   └── embedder.py       # Embedding generation
│   ├── retrieval/
│   │   ├── vector_store.py   # ChromaDB interface
│   │   └── hybrid_search.py  # BM25 + vector hybrid search
│   ├── generation/
│   │   ├── llm_client.py     # LLM API client
│   │   ├── prompts.py        # Prompt templates
│   │   └── chain.py          # RAG chain orchestration
│   └── evaluation/
│       ├── metrics.py        # Evaluation metrics
│       └── evaluator.py      # Evaluation pipeline
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Siddharth** — AI Engineer  
Building production-grade AI systems, not just demos.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/yoshimitsu117)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/yoshimitsu117)

