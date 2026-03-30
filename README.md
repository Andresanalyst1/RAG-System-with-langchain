#  Local RAG System — AI Recruiter Assistant

> Ask anything about my professional background. Powered by a fully local RAG pipeline — no data leaves your machine.

---

##  Overview

This project is a **Retrieval-Augmented Generation (RAG) system** built to let recruiters and hiring managers have a natural conversation about my professional profile. Instead of reading through a static CV, they can ask questions directly and get context-aware, accurate answers grounded in real information about my experience.

The stack runs **100% locally** using Ollama, meaning no API keys, no cloud costs, and no private data sent to third parties — a deliberate design choice that reflects real-world enterprise priorities around data privacy.

---

##  Architecture

```
andres_cardenas_profile.md
        │
        ▼
 Document Loader (auto-detected by type: MD, PDF, DOCX, TXT, URL)
        │
        ▼
 Text Splitter (RecursiveCharacterTextSplitter)
  chunk_size=500 | chunk_overlap=50
        │
        ▼
 Embeddings (mxbai-embed-large via Ollama)
        │
        ▼
 Vector Store (ChromaDB — persisted locally)
        │
        ▼
 Retriever (top-k=6 similarity search)
        │
        ▼
 LLM Chain (gemma3:4b via Ollama + LangChain prompt)
        │
        ▼
 Streaming response in terminal
```

---

##  Tech Stack

| Component       | Technology                          |
|----------------|--------------------------------------|
| LLM             | `gemma3:4b` via Ollama              |
| Embeddings      | `mxbai-embed-large:335m` via Ollama |
| Vector Store    | ChromaDB (local persistence)        |
| RAG Framework   | LangChain                           |
| Document Source | MD, PDF, DOCX, TXT, web pages       |

---

##  Features

- **Fully local inference** — no OpenAI or Anthropic API keys required
- **Persistent vector store** — embeddings are computed once and reused across sessions
- **Streaming responses** — answers stream token by token in real time
- **Recruiter-aware prompt** — the LLM is instructed to answer in the context of a professional conversation
- **Multi-document ingestion** — drop `.md`, `.pdf`, `.docx`, or `.txt` files into `docs/`, or add URLs via `.env`

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally

### 1. Pull the required models

```bash
ollama pull gemma3:4b
ollama pull mxbai-embed-large:335m
```

### 2. Clone the repository

```bash
git clone https://github.com/Andresanalyst1/RAG-System-with-langchain.git
cd RAG-System-with-langchain
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Edit `.env` if you want to use a different model or tune chunking parameters. The defaults work out of the box.

### 5. Run the assistant

```bash
python LLM.py
```

On first run, the vector store is built from all documents in `docs/`. Subsequent runs reuse the persisted embeddings for faster startup.

> **To add new documents:** drop files into `docs/` (or add URLs to `DOCS_URLS` in `.env`), delete the `chroma_langchain_md/` folder, and rerun to re-ingest.

---

## 💬 Example Questions

```
Ask your question: What kind of ML projects has Andres worked on?
Ask your question: Does he have experience with data pipelines?
Ask your question: What tools does he use for data analysis?
Ask your question: Has he deployed any models to production?
```

---

## Project Structure

```
RAG-System-with-langchain/
├── LLM.py                        # Main RAG chain and conversation loop
├── embedding.py                  # Document ingestion and vector store setup
├── evaluate.py                   # RAGAS evaluation script (run on demand)
├── eval_dataset.json             # Test questions and ground truths
├── docs/                         # Drop your documents here (MD, PDF, DOCX, TXT)
│   └── andres_cardenas_profile.md
├── requirements.txt              # Python dependencies
├── .env                          # Local config — model names and parameters (git-ignored)
└── README.md
```

---

## Roadmap

- [ ] Add a **Streamlit UI** for browser-based interaction (no terminal required)
- [x] Add **conversation memory** so the assistant tracks follow-up questions
- [x] Support **multi-document ingestion** (PDF, DOCX, web pages)
- [x] Add **RAG evaluation** with RAGAS metrics (faithfulness, answer relevancy)
- [ ] Containerize with **Docker** for reproducible environments
- [x] Add `.env` config for model names and parameters

---

## Evaluation

RAG quality is measured with [RAGAS](https://docs.ragas.io) using fully local Ollama models — no OpenAI key required.

```bash
python evaluate.py
```

**Metrics:**
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — is the answer on-topic with the question?

Edit `eval_dataset.json` to add or update test questions and ground truths.

---

## Design Decisions

**Why local?** Privacy is a first-class concern in enterprise AI. Running the entire stack locally means no profile data is sent to external APIs — a pattern directly applicable to internal company chatbots where confidential documents can't leave the network.

**Why ChromaDB?** Lightweight, zero-config persistence with a simple Python API. Appropriate for single-user or small-scale deployments where operational complexity should be minimal.

**Why RAG over fine-tuning?** RAG allows the knowledge base to be updated without retraining. Swapping in a new version of the profile document is all that's needed to update the assistant's knowledge.

---

## Author

**Andres Cardenas**
Data & AI enthusiast based in Brisbane, Australia.
[GitHub](https://github.com/Andresanalyst1) | [LinkedIn](https://www.linkedin.com/in/andres-cardenas-4b992a191/)

---

## License

MIT