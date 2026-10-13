"""
Configuration module for the RAG Chatbot Pipeline.
Manages settings, paths, model options, and vector store parameters.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env if present, fallback to .env.example if needed
load_dotenv(override=True)
BASE_DIR = Path(__file__).resolve().parent
if not os.getenv("GROQ_API_KEY") and (BASE_DIR / ".env.example").exists():
    load_dotenv(BASE_DIR / ".env.example", override=False)

@dataclass
class RAGConfig:
    # Vector store settings
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "chroma").lower()
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "chroma_db"))
    faiss_persist_dir: str = os.getenv("FAISS_PERSIST_DIRECTORY", str(BASE_DIR / "data" / "faiss_index"))
    collection_name: str = os.getenv("COLLECTION_NAME", "rag_knowledge_base")

    # Document and chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    documents_dir: str = str(BASE_DIR / "data" / "documents")

    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "4"))
    search_type: str = os.getenv("SEARCH_TYPE", "similarity")  # "similarity" or "mmr"

    # Embedding provider settings ('local_minilm', 'openai', 'ollama')
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local_minilm").lower()
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # LLM provider settings ('openai', 'groq', 'ollama', 'mock')
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")

    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    def ensure_directories(self) -> None:
        """Create necessary data directories if they do not exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.faiss_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)

# Default global config instance
default_config = RAGConfig()
default_config.ensure_directories()
