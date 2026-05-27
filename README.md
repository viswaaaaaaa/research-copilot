# 🔬 AI Research Co-pilot

> A production-oriented AI research assistant that combines multi-stage retrieval, reranking, persistent memory, and live tool integration to deliver grounded, citation-backed answers from research papers.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 🚀 Overview

AI Research Co-pilot is a local-first Retrieval-Augmented Generation (RAG) system designed for intelligent interaction with research papers.

Unlike basic PDF chatbots, this system implements a multi-stage retrieval pipeline with reranking, confidence-aware answer generation, observability logging, and guardrails for safer and more reliable responses.

The system can:

* Ingest and understand research papers
* Retrieve highly relevant context using hybrid retrieval
* Re-rank retrieved chunks for precision
* Detect low-confidence retrievals
* Prevent prompt injection attempts
* Persist user research memory across sessions
* Search the web or Arxiv when local context is insufficient

---

# ✨ Key Features

## 🔍 Multi-Stage Retrieval Pipeline

* Dense vector retrieval using ChromaDB
* MMR (Max Marginal Relevance) retrieval for diversity
* BM25 sparse retrieval for keyword-aware search
* Cross-encoder reranking for precision optimization

## 🧠 Confidence-Aware Generation

* Retrieval confidence scoring
* Confidence bands (high / medium / low)
* Low-confidence answer rejection
* Reduced hallucination behavior

## 🛡️ Guardrails & Safety

* Prompt injection detection
* Input validation
* Retrieval gating before LLM generation
* Strict grounded prompting

## 📊 Observability & Debugging

* Structured JSONL query logging
* Retrieval score tracking
* Source page tracing
* Answer analytics for debugging retrieval quality

## 🧬 Persistent Memory

* Stores user research context across sessions
* Adapts responses to research domain over time

## 🌐 Live Tool Integration

* Web search integration
* Arxiv paper retrieval
* Dynamic fallback when local retrieval confidence is weak

---

# 🏗️ System Architecture

```text
                ┌─────────────────────┐
                │   Research PDF      │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  PDF Ingestion      │
                │  + Chunking         │
                │  + Metadata         │
                │  + Filtering        │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Embedding Pipeline  │
                │ all-MiniLM-L6-v2    │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ ChromaDB Vector DB  │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Hybrid Retrieval    │
                │ MMR + BM25          │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Cross-Encoder       │
                │ Reranking           │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Confidence Scoring  │
                │ + Guardrails        │
                └─────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   High Confidence           Low Confidence
              │                       │
              ▼                       ▼
      Local RAG Answer      Web / Arxiv Search
              │
              ▼
      Llama 3.2 Generation
              │
              ▼
     Observability Logging
```

---

# ⚡ Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/viswaaaaaaa/research-copilot.git
cd research-copilot
```

## 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Pull Local LLM

```bash
ollama pull llama3.2
```

## 5. Configure Environment

```bash
cp .env.example .env
```

Add your Tavily API key inside `.env`.

## 6. Run Application

```bash
streamlit run ui/app.py
```

---

# 🧪 Example Capabilities

### Research Understanding

* “What is the main contribution of this paper?”
* “Explain the CR-FS method”
* “What datasets were used in experiments?”

### Confidence-Aware Responses

* Answers confidently when retrieval quality is high
* Explicitly states uncertainty when context is insufficient

### Production-Oriented Behaviors

* Prevents prompt injection attempts
* Logs retrieval analytics
* Tracks retrieval confidence
* Grounds answers strictly in retrieved context

---

# 🛠️ Tech Stack

| Category   | Tools                   |
| ---------- | ----------------------- |
| LLM        | Ollama + Llama 3.2      |
| Retrieval  | ChromaDB                |
| Embeddings | all-MiniLM-L6-v2        |
| Reranking  | CrossEncoder (MS MARCO) |
| Frameworks | LangChain               |
| UI         | Streamlit               |
| Language   | Python                  |

---

# 📈 Future Improvements

* Metadata-aware retrieval
* Section-aware chunking
* Semantic chunking
* Agentic multi-hop retrieval
* Research graph generation
* Citation graph visualization

---

# 📄 License

MIT License
