# Graph Paper AI 📄🤖

> **Vectorless RAG & Hierarchical Semantic Document Analysis for Academic Papers**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![LLM](https://img.shields.io/badge/Ollama-Qwen%202.5-orange.svg)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-green.svg)](https://github.com/features/actions)
[![Evaluation](https://img.shields.io/badge/DeepEval-Framework-purple.svg)](https://confident-ai.com/)

---

## 🌟 Overview

**Graph Paper AI** is a state-of-the-art **Vectorless RAG (Retrieval-Augmented Generation)** application designed specifically for analyzing complex scientific publications, technical manuals, and data-dense PDFs. 

Unlike traditional RAG systems that rely on naive text chunking and vector embeddings (which often fragment tables, lose document structure, and dilute metric comparisons), Graph Paper AI constructs a **hierarchical semantic document tree** directly from agentic PDF parses. It uses a multi-stage hybrid router to retrieve exact sections, tables, and figure contexts without embedding databases.

---

## ✨ Key Features

- **📄 Agentic PDF Extraction**: Utilizes **LlamaCloud API** for structural markdown extraction, parsing tables into clean HTML and preserving section headings.
- **🌲 Hierarchical Document Tree Parser**: Builds an in-memory document tree with parent-child relationships, preserving chapter hierarchy and page-bounded sections.
- **🚀 3-Stage Hybrid Fast Retriever**:
  - **Pass A (Regex Router)**: Instant (<1ms) deterministic extraction for explicit component references (e.g. `Table III`, `Fig 5`).
  - **Pass B (BM25 Lexical Search)**: Fast candidate filtering with section-title weighting and comparative intent boosting (`best`, `compare`, `results`).
  - **Pass C (SLM Reranker)**: Semantic reranking powered by a lightweight local model (`qwen2.5:1.5b-instruct-q4_K_M`) via Ollama.
- **🧩 Parent-Node Context Expansion & Deduplication**: Prevents section fragmentation by expanding sub-nodes back to full parent section contexts while eliminating duplicate text blocks.
- **🧠 Anchored LLM Generator**: Generates answers grounded strictly in retrieved context using `qwen2.5:3b` with exact inline section and page citations.
- **🔬 End-to-End Evaluation Framework**: Integrated test suite powered by **DeepEval** and NVIDIA-hosted LLM Judge (`openai/gpt-oss-120b`) assessing Faithfulness, Answer Relevancy, Contextual Precision, and Contextual Recall.
- **🐳 Docker & Docker Compose Support**: Fully containerized Streamlit application and local Ollama service for seamless deployment.
- **⚡ GitHub Actions CI/CD Pipeline**: Automated unit testing on every push/PR, release automation with container packaging, and automated dependency maintenance via Dependabot.
- **💻 Streamlit Web Application**: Modern interactive UI featuring reasoning trace inspection, source section highlights, and multi-language support (English / French).

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │   PDF Document File    │
                       └───────────┬────────────┘
                                   │ (LlamaCloud Agentic Parse)
                                   ▼
                       ┌────────────────────────┐
                       │  Markdown & HTML Tables│
                       └───────────┬────────────┘
                                   │ (_build_pure_text_tree)
                                   ▼
                       ┌────────────────────────┐
                       │ Hierarchical Doc Tree  │
                       └───────────┬────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
    ┌────────────────────────┐         ┌───────────────────────────┐
    │  User Query (Streamlit)│         │ DeepEval Synthetic Dataset│
    └────────────┬───────────┘         └────────────┬──────────────┘
                 │                                  │
                 ▼                                  ▼
    ┌───────────────────────────────────────────────────────────┐
    │              3-Stage Fast Tree Retriever                  │
    │  - Pass A: Regex Component Router (<1ms)                  │
    │  - Pass B: Title-Weighted BM25 Candidate Pre-filter       │
    │  - Pass C: Qwen 1.5B SLM Semantic Reranker                │
    └────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────────────────┐
    │         Parent-Node Expansion & Context Deduplicator      │
    └────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
    ┌───────────────────────────────────────────────────────────┐
    │            Ollama Generator (Qwen 2.5:3b)                 │
    │          Grounded Answer + Page/Section Citations         │
    └────────────────────────────┬──────────────────────────────┘
```

---

## 📂 Repository Structure

```
graph-paper-ai/
├── .github/
│   ├── dependabot.yml              # Weekly automated dependency update configuration
│   └── workflows/
│       ├── ci.yml                  # CI pipeline (linting & fast unit test execution)
│       └── release.yml             # Release workflow (tag-triggered build & artifact release)
├── app.py                          # Streamlit Interactive Web Application
├── Dockerfile                      # Container build definition for Streamlit UI
├── docker-compose.yml              # Multi-container setup (Streamlit + Ollama service)
├── src/
│   ├── extraction/
│   │   └── parser.py               # LlamaCloud integration & pure text tree builder
│   ├── query/
│   │   └── generator.py            # Grounded Ollama answer generation
│   ├── retrieval/
│   │   ├── tree_search.py          # 3-Stage FastTreeRetriever (Regex + BM25 + SLM)
│   │   └── retriever.py            # Parent-node expansion & context deduplication
│   ├── utils/
│   │   └── dictionary.py           # Multi-language UI translations
│   └── pipeline.py                 # Core Vectorless RAG orchestration
├── tests/                          # Modular unit test suite
│   ├── test_parser.py              # Tests for document tree parsing & sub-node splitting
│   ├── test_retriever.py           # Tests for parent expansion, deduplication & fallback
│   ├── test_tree_search.py        # Tests for 3-pass retriever logic & regex matching
│   └── test_pipeline.py           # E2E pipeline integration tests (Ollama mocked)
├── deepeval/                       # LLM evaluation suite
│   ├── generate_conversational.py  # Synthetic QA dataset generator (Goldens)
│   ├── llm_judge.py                # NVIDIA OpenAI-compatible LLM Judge wrapper
│   ├── model_callback.py           # DeepEval Pytest evaluation suite
│   └── report.py                   # Batch evaluation report script
├── test_data/                      # Cached trees & synthetic evaluation datasets
├── pyproject.toml                  # Project metadata & pytest configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option A: Local Python Environment

#### Prerequisites

1. **Python 3.11+**
2. **Ollama**: Download and install [Ollama](https://ollama.com/).
3. Pull required local models:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:1.5b-instruct-q4_K_M
   ```

#### Setup Steps

1. **Clone Repository**:
   ```bash
   git clone https://github.com/my_username/graph-paper-ai.git
   cd graph-paper-ai
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   LLAMACLOUD_API_KEY=llx-your-llamacloud-key
   OPENAI_API_KEY_NVDIA=nvapi-your-nvidia-api-key
   ```

5. **Run Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

### Option B: Docker Compose (Recommended for Containerized Deployment)

1. **Start Application & Ollama Service**:
   ```bash
   docker compose up --build
   ```

2. **Pull Ollama Models Inside Container**:
   ```bash
   docker exec -it graph-paper-ai-ollama-1 ollama pull qwen2.5:3b
   docker exec -it graph-paper-ai-ollama-1 ollama pull qwen2.5:1.5b-instruct-q4_K_M
   ```

3. Open `http://localhost:8501` in your browser.

---

## 🧪 Testing & CI/CD Workflows

### Running Unit Tests Locally

Run the unit test suite with integration tests excluded (no local Ollama service required):

```bash
pytest tests/ -v --tb=short -m "not integration"
```

### GitHub Actions Workflows

- **Continuous Integration (`.github/workflows/ci.yml`)**:
  - Automatically triggered on `push` to `main` and all `pull_request`s.
  - Runs unit tests and validates package installation on Python 3.11 & 3.12.
- **Automated Releases (`.github/workflows/release.yml`)**:
  - Triggered on tag pushes (e.g. `v1.0.0`).
  - Automatically builds source/wheel distributions, creates GitHub Releases, and packages Docker build artifacts.
- **Dependabot (`.github/dependabot.yml`)**:
  - Weekly automated checks for dependency and GitHub Actions updates.

---

## 🔬 Evaluation & Benchmarking

Graph Paper AI includes an automated evaluation pipeline powered by **DeepEval**.

### 1. Generate Synthetic Test Dataset
```bash
python deepeval/generate_conversational.py
```

### 2. Run Pytest Evaluation Suite
```bash
deepeval test run deepeval/model_callback.py
```

### 3. Generate Evaluation Report
```bash
python deepeval/report.py
```
<img width="679" height="178" alt="graph2" src="https://github.com/user-attachments/assets/c80e9f77-4052-414f-a40b-86e91ca9c1f6" />

Faithfulness and Answer Relevancy are strong — the model rarely hallucinates and stays on-topic. Contextual Precision/Relevancy are lower, which mostly reflects the retriever sometimes pulling in tangentially related sections alongside the right one — the next thing I'm iterating on.
---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for details.
