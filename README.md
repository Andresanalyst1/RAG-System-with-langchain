# LOLA — RAG-Powered Recruiter Assistant
Click right here: https://lola-rag-system-with-langchain.streamlit.app/

> Ask anything about Andres' professional background. LOLA is an AI assistant powered by a RAG pipeline that answers recruiter questions with context-aware, grounded responses.

---

## Overview

This project is a **Retrieval-Augmented Generation (RAG) system** that lets recruiters and hiring managers have a natural conversation about Andres' professional profile. Instead of reading through a static CV, they can ask questions directly and get accurate, context-aware answers.

LOLA uses **Claude (Anthropic API)** for generation and the **HuggingFace Inference API** for embeddings. A **Streamlit web UI** provides a clean chat interface, while a CLI mode is also available for development.

---

## Architecture

```
docs/ (MD, PDF, DOCX, TXT) + URLs
        |
        v
 Document Loader (auto-detected by file type)
        |
        v
 Text Splitter (RecursiveCharacterTextSplitter)
  chunk_size=500 | chunk_overlap=50
        |
        v
 Embeddings (all-MiniLM-L6-v2 via HuggingFace Inference API)
        |
        v
 Vector Store (ChromaDB — persisted locally)
        |
        v
 Retriever (top-k similarity search)
        |
        v
 LLM Chain (Claude via Anthropic API + LangChain prompt)
        |
        v
 Streaming response (Streamlit UI or terminal)
```

---

## Tech Stack

| Component       | Technology                          |
|----------------|--------------------------------------|
| LLM             | Claude (Anthropic API)              |
| Embeddings      | `all-MiniLM-L6-v2` via HuggingFace Inference API |
| Vector Store    | ChromaDB (local persistence)        |
| RAG Framework   | LangChain                           |
| Web UI          | Streamlit                           |
| Evaluation      | RAGAS (faithfulness + relevancy)    |
| Document Source  | MD, PDF, DOCX, TXT, web pages      |

---

## Features

- **Streamlit chat UI** — browser-based interface with streaming responses and conversation history
- **CLI mode** — terminal-based chat loop for development and testing
- **Persistent vector store** — embeddings are computed once and reused across sessions
- **Streaming responses** — answers stream token by token in real time
- **Conversation memory** — configurable sliding window so the assistant tracks follow-up questions
- **Multi-document ingestion** — drop `.md`, `.pdf`, `.docx`, or `.txt` files into `docs/`, or add URLs via `.env`
- **RAG evaluation** — automated quality scoring with RAGAS (faithfulness + answer relevancy)
- **Lazy document loading** — documents are only parsed and chunked when the vector store is empty, for faster startup

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/settings/keys)
- A [HuggingFace access token](https://huggingface.co/settings/tokens) (free — "Read" permission is enough)

### 1. Clone the repository

```bash
git clone https://github.com/Andresanalyst1/RAG-System-with-langchain.git
cd RAG-System-with-langchain
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Then edit `.env` and set both `ANTHROPIC_API_KEY` and `HF_TOKEN`.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. On first run, the vector store is built from all documents in `docs/`. Subsequent runs reuse the persisted embeddings for faster startup.

### 4b. Run in CLI mode (optional)

```bash
python LLM.py
```

> **To update documents:** drop files into `docs/` (or add URLs to `DOCS_URLS` in `.env`), delete the `chroma_langchain_md/` folder, and restart to re-ingest.

---

## Example Questions

```
What is Andres' current role?
What ML projects has Andres worked on?
What certifications does Andres have?
What programming languages and tools does Andres use?
Does Andres have leadership experience?
```

---

## Make It Your Own

This repo is built around Andres' profile, but you can adapt it to any person or knowledge base in a few steps:

1. **Replace the documents** — delete `docs/andres_cardenas_profile.md` and drop your own files (`.md`, `.pdf`, `.docx`, `.txt`) into `docs/`. Then delete the `chroma_langchain_md/` folder so the vector store rebuilds on next run.
2. **Edit the prompt** — open `LLM.py` and update the `template` string. Change the name, persona, and instructions to match your use case.
3. **Update the UI branding** — open `app.py` and change the page title, sidebar text, and chat input placeholder.
4. **Write your own eval dataset** — replace the questions and ground truths in `eval_dataset.json` with ones relevant to your documents, then run `python evaluate.py`.

---

## Project Structure

```
RAG-System-with-langchain/
├── app.py                        # Streamlit web UI (chat interface)
├── LLM.py                       # LLM chain, prompt template, and CLI loop
├── embedding.py                  # Document ingestion and vector store setup
├── evaluate.py                   # RAGAS evaluation script
├── eval_dataset.json             # Test questions and ground truths
├── docs/                         # Drop your documents here (MD, PDF, DOCX, TXT)
│   └── andres_cardenas_profile.md
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template (copy to .env)
├── .env                          # Local config (git-ignored)
└── README.md
```

---

## Evaluation

RAG quality is measured with [RAGAS](https://docs.ragas.io). The judge LLM is the same Claude model used for generation. Embeddings for the answer relevancy metric use the HuggingFace Inference API (same model as the RAG pipeline).

```bash
python evaluate.py
```

**Metrics:**
- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — is the answer on-topic with the question?

Edit `eval_dataset.json` to add or update test questions and ground truths.

---

## Design Decisions

**Why RAG over fine-tuning?** RAG allows the knowledge base to be updated without retraining. Swapping in a new document is all that's needed to update the assistant's knowledge.

**Why ChromaDB?** Lightweight, zero-config persistence with a simple Python API. Appropriate for single-user or small-scale deployments where operational complexity should be minimal.

**Why hosted embeddings?** Embeddings are called via the HuggingFace Inference API instead of running a model locally. This keeps the deployment lightweight (no `torch`/`sentence-transformers` in the image), fits comfortably on Streamlit Community Cloud's free tier, and keeps cold starts fast. Generation uses Claude via the Anthropic API for higher quality responses.

---

## Author

**Andres Cardenas**
Data & AI enthusiast based in Brisbane, Australia.
[GitHub](https://github.com/Andresanalyst1) | [LinkedIn](https://www.linkedin.com/in/andres-cardenas-4b992a191/)

---

## License

MIT
