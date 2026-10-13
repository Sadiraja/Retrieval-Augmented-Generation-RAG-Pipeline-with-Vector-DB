"""
Embeddings module for the RAG Chatbot Pipeline.
Provides both local zero-cost embeddings (MiniLM ONNX) and cloud embeddings (OpenAI).
"""

from __future__ import annotations
import os
import logging
from typing import List, Optional
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LocalMiniLMEmbeddings(Embeddings):
    """
    Local CPU-based text embedding generator using ChromaDB's embedded ONNX MiniLM model.
    Produces 384-dimensional dense vectors. Requires zero external API keys and runs offline.
    """

    def __init__(self):
        try:
            import chromadb.utils.embedding_functions as ef
            self._ef = ef.DefaultEmbeddingFunction()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize ChromaDB DefaultEmbeddingFunction: {exc}"
            ) from exc

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        # chromadb ef expects list of strings and returns numpy arrays or list of lists
        raw_embeddings = self._ef(texts)
        return [list(map(float, emb)) for emb in raw_embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        raw_embedding = self._ef([text])[0]
        return list(map(float, raw_embedding))


def get_embeddings(
    provider: str = "local_minilm",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Embeddings:
    """
    Factory function returning a LangChain Embeddings instance.

    Supported providers:
      - 'local_minilm': 100% offline, CPU fast, 384 dimensions.
      - 'openai': OpenAI embeddings (e.g. text-embedding-3-small).
      - 'ollama': Local Ollama embeddings (e.g. nomic-embed-text).
    """
    provider = provider.lower()

    if provider == "openai":
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            logger.warning(
                "OPENAI_API_KEY not found. Falling back to local MiniLM embeddings."
            )
            return LocalMiniLMEmbeddings()
        from langchain_community.embeddings import OpenAIEmbeddings
        emb_model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=emb_model, openai_api_key=resolved_key)

    elif provider == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            emb_model = model_name or "nomic-embed-text"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(model=emb_model, base_url=base_url)
        except Exception as exc:
            logger.warning("Ollama embeddings failed to initialize (%s). Falling back to local MiniLM.", exc)
            return LocalMiniLMEmbeddings()

    # Default: local zero-cost embeddings
    return LocalMiniLMEmbeddings()
