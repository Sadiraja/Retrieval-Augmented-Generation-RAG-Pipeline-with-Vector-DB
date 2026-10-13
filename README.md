# 🧠 Retrieval-Augmented Generation (RAG) Chatbot Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built using **LangChain**, featuring dual vector store support (**ChromaDB** and **FAISS**), hybrid embedding choices (100% offline local embeddings + OpenAI), conversational memory with query reformulation, an interactive **Streamlit Web UI**, and a robust **CLI tool**.

---

## 📌 Architecture Overview

```
                                 ┌──────────────────────────────────────┐
                                 │ Document Ingestion (.pdf, .docx, .md)│
                                 └──────────────────┬───────────────────┘
                                                    │
                                                    ▼
                                    Recursive Character Text Splitter
                                     (1000 chars, 200 overlap + Meta)
                                                    │
                                                    ▼
                                            Embedding Model
                         (Local MiniLM ONNX 384-dim  OR  OpenAI text-embedding-3)
                                                    │
                                                    ▼
                             ┌──────────────────────┴──────────────────────┐
                             ▼                                             ▼
                 ┌───────────────────────┐                     ┌───────────────────────┐
                 │   ChromaDB Store      │                     │      FAISS Store      │
                 │ (Persistent SQLite)   │                     │ (In-Memory / Serial.) │
                 └───────────┬───────────┘                     └───────────┬───────────┘
                             │                                             │
                             └──────────────────────┬──────────────────────┘
                                                    │
User Query ──► [Conversational Memory] ──► Query Reformulator
                                                    │
                                                    ▼
                                      Retriever (Top-K / MMR)
                                                    │
                                                    ▼
                                        Context-Grounded QA Prompt
                                                    │
                                                    ▼
                                              LLM Generator
                             (OpenAI GPT-4o-mini / Groq / Ollama / Local Extractive)
                                                    │
                                                    ▼
                                      Grounded Answer + Citations
```

---

## ✨ Features

- **Dual Vector Store Architecture**: Seamlessly switch between **ChromaDB** (persistent SQLite/DuckDB storage) and **FAISS** (high-speed similarity search by Meta AI) with identical retrieval interfaces.
- **100% Offline / Zero-Cost by Default**: Runs immediately out of the box using embedded 384-dimensional ONNX MiniLM embeddings and a local extractive fallback LLM. No paid API key required to test or run!
- **Cloud & Local LLM Integration**: Fully compatible with **OpenAI** (`gpt-4o-mini`, `gpt-4o`), **Groq** (`llama-3.3-70b-versatile`), and local **Ollama** (`llama3.2`).
- **Conversational Memory & Follow-Up Understanding**: History-aware query rewriting re-contextualizes follow-up questions before vector search.
- **Verifiable Citations**: Every generated answer tracks source documents, page numbers, and chunk references with text previews.
- **Interactive Streamlit Web UI**: Drag-and-drop document upload, real-time chunk indexing, streaming chat responses, and expandable citations.
- **Full CLI Tooling**: Ingest documents, run queries, or launch interactive terminal chat sessions.
- **End-to-End Test Suite**: Complete unit and integration tests with Pytest.

---

## 📁 Project Structure

```
├── app.py                     # Streamlit interactive Web Application
├── cli.py                     # Command-Line Interface (ingest, query, chat, stats)
├── config.py                  # Configuration & path management
├── embeddings.py              # Embedding model factory (Local MiniLM & OpenAI)
├── loaders.py                 # Document loaders (PDF, DOCX, TXT, Markdown)
├── chunking.py                # Recursive text splitter with metadata enrichment
├── vectorstore.py             # VectorStoreManager unifying ChromaDB and FAISS
├── rag_chain.py               # LangChain LCEL Conversational RAG pipeline
├── requirements.txt           # Project dependencies
├── conftest.py                # Pytest configuration
├── .env.example               # Environment variables template
├── sample_data/
│   └── knowledge_base.md      # Sample knowledge base for instant testing
└── tests/
    └── test_rag.py            # Pytest test suite
```

---

## 🚀 Quickstart Guide

### 1. Installation
Install the dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Configuration (Optional)
Copy `.env.example` to `.env` to configure your API keys (if using OpenAI or Groq):
```powershell
copy .env.example .env
```
Edit `.env` and fill in:
```env
OPENAI_API_KEY=your_openai_api_key_here
# or
GROQ_API_KEY=your_groq_api_key_here
```
*(If no API key is set, the system automatically falls back to local zero-cost processing!)*

---

## 🌐 Running the Streamlit Web Application

Launch the interactive web UI:
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### In the Web UI:
1. **Choose your LLM Provider** in the sidebar:
   - Select **⚡ Groq (Llama 3.3 70B - Recommended)**.
   - Enter your Groq API key (`gsk_...`) in the password box.
   - Click **"🔌 Test Groq"** to verify connection.
   - Click **"💾 Save Key"** to persist your key to `.env` so you don't have to enter it again.
2. **Choose your Vector Store** (ChromaDB or FAISS) in the sidebar.
3. Open the **Document Ingestion** section:
   - Click **"Index Sample Knowledge Base Now"** to test with the included knowledge base, or
   - Drag and drop your own PDF (e.g. resumes, manuals), DOCX, TXT, or MD files and click **"Process & Index Uploaded Documents"**.
4. Chat with the bot — watch real-time token streaming and expand the **"🔍 Citations & Retrieved Context"** accordion to inspect exact source chunks!

---

## 💻 Command-Line Interface (CLI)

### Ingest Documents
Ingest a file or directory into ChromaDB or FAISS:
```powershell
# Ingest sample knowledge base into ChromaDB
python cli.py ingest --path sample_data/knowledge_base.md --store chroma

# Ingest into FAISS
python cli.py ingest --path sample_data/knowledge_base.md --store faiss

# Ingest into both vector stores simultaneously
python cli.py ingest --path sample_data/ --store both
```

### Single Query
Ask a question directly from the terminal:
```powershell
python cli.py query -q "What are the core stages of the RAG workflow?" --store chroma

python cli.py query -q "What is the difference between ChromaDB and FAISS?" --store faiss
```

### Interactive Terminal Chat
Start a conversational multi-turn session with memory:
```powershell
python cli.py chat --store chroma
```
*(Type `clear` to reset memory, or `exit` to quit).*

### Inspect Vector DB Statistics
```powershell
python cli.py stats
```

### Reset Vector Database
```powershell
python cli.py reset --store all
```

---

## 🧪 Running the Automated Test Suite

Run the full test suite verifying embeddings, loaders, chunking, ChromaDB, FAISS, and the RAG pipeline:
```powershell
python -m pytest tests/test_rag.py -v
```
All 9 unit and integration tests run in ~6 seconds.

---

## 📜 License
MIT License.
