"""
Vector Store Manager module for RAG Chatbot Pipeline.
Provides a unified abstraction for both ChromaDB and FAISS vector stores.
"""

from __future__ import annotations
import os
import shutil
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.vectorstores import Chroma, FAISS

from config import RAGConfig, default_config
from embeddings import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Unified manager for ChromaDB and FAISS vector stores.
    Supports persistent storage, document ingestion, similarity/MMR retrieval,
    and index inspection.
    """

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        self.config = config or default_config
        self.config.ensure_directories()
        self.embeddings = embeddings or get_embeddings(self.config.embedding_provider)
        self._chroma_store: Optional[Chroma] = None
        self._faiss_store: Optional[FAISS] = None

    # -------------------------------------------------------------------------
    # ChromaDB Handling
    # -------------------------------------------------------------------------
    def get_chroma_store(self) -> Chroma:
        """Get or initialize the persistent ChromaDB vector store."""
        if self._chroma_store is None:
            self._chroma_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.chroma_persist_dir,
            )
        return self._chroma_store

    # -------------------------------------------------------------------------
    # FAISS Handling
    # -------------------------------------------------------------------------
    def get_faiss_store(self) -> Optional[FAISS]:
        """Get or load the persistent FAISS vector store from disk."""
        if self._faiss_store is not None:
            return self._faiss_store

        faiss_dir = Path(self.config.faiss_persist_dir)
        index_file = faiss_dir / "index.faiss"

        if index_file.exists():
            try:
                self._faiss_store = FAISS.load_local(
                    folder_path=str(faiss_dir),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", faiss_dir)
            except Exception as exc:
                logger.warning("Failed to load existing FAISS index: %s", exc)
                self._faiss_store = None
        return self._faiss_store

    def _save_faiss(self) -> None:
        """Persist the FAISS index to disk."""
        if self._faiss_store is not None:
            Path(self.config.faiss_persist_dir).mkdir(parents=True, exist_ok=True)
            self._faiss_store.save_local(self.config.faiss_persist_dir)
            logger.info("Saved FAISS index to %s", self.config.faiss_persist_dir)

    # -------------------------------------------------------------------------
    # Unified Store Access
    # -------------------------------------------------------------------------
    def get_store(self, store_type: Optional[str] = None) -> Optional[VectorStore]:
        """Retrieve either Chroma or FAISS store by name."""
        target = (store_type or self.config.vector_store_type).lower()
        if target == "chroma":
            return self.get_chroma_store()
        elif target == "faiss":
            return self.get_faiss_store()
        else:
            raise ValueError(f"Unknown vector store type: '{target}'. Use 'chroma' or 'faiss'.")

    def add_documents(
        self, documents: List[Document], store_type: Optional[str] = None
    ) -> int:
        """
        Add documents to the chosen vector store (or 'both').
        Returns the number of added document chunks.
        """
        if not documents:
            return 0

        target = (store_type or self.config.vector_store_type).lower()

        if target in ("chroma", "all", "both"):
            chroma = self.get_chroma_store()
            chroma.add_documents(documents)
            logger.info("Added %d documents to ChromaDB", len(documents))

        if target in ("faiss", "all", "both"):
            if self._faiss_store is None:
                faiss_existing = self.get_faiss_store()
                if faiss_existing is None:
                    # Initialize first FAISS index from documents
                    self._faiss_store = FAISS.from_documents(documents, self.embeddings)
                else:
                    self._faiss_store.add_documents(documents)
            else:
                self._faiss_store.add_documents(documents)
            self._save_faiss()
            logger.info("Added %d documents to FAISS", len(documents))

        return len(documents)

    def get_retriever(
        self,
        store_type: Optional[str] = None,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> VectorStoreRetriever:
        """
        Build and return a LangChain VectorStoreRetriever.
        Supports 'similarity' and 'mmr' search types.
        """
        target = (store_type or self.config.vector_store_type).lower()
        top_k = k or self.config.top_k
        mode = search_type or self.config.search_type

        store = self.get_store(target)
        if store is None:
            raise RuntimeError(
                f"Vector store '{target}' has not been initialized or has no documents yet."
            )

        search_kwargs: Dict[str, Any] = {"k": top_k}
        if mode == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult

        return store.as_retriever(search_type=mode, search_kwargs=search_kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        store_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search and return documents alongside distance/score."""
        target = (store_type or self.config.vector_store_type).lower()
        top_k = k or self.config.top_k

        store = self.get_store(target)
        if store is None:
            return []

        return store.similarity_search_with_score(query, k=top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics on document count and indexing status for both stores."""
        stats = {
            "chroma": {"count": 0, "status": "empty"},
            "faiss": {"count": 0, "status": "empty"},
        }

        # Chroma stats
        try:
            chroma = self.get_chroma_store()
            chroma_count = chroma._collection.count()
            stats["chroma"]["count"] = chroma_count
            stats["chroma"]["status"] = "ready" if chroma_count > 0 else "empty"
        except Exception as exc:
            stats["chroma"]["error"] = str(exc)

        # FAISS stats
        try:
            faiss_store = self.get_faiss_store()
            if faiss_store is not None and hasattr(faiss_store, "index"):
                count = faiss_store.index.ntotal
                stats["faiss"]["count"] = count
                stats["faiss"]["status"] = "ready" if count > 0 else "empty"
        except Exception as exc:
            stats["faiss"]["error"] = str(exc)

        return stats

    def reset_store(self, store_type: str = "all") -> None:
        """Clear indexed vectors and reset storage on disk."""
        target = store_type.lower()

        if target in ("chroma", "all", "both"):
            try:
                if self._chroma_store is not None:
                    self._chroma_store.delete_collection()
                chroma_path = Path(self.config.chroma_persist_dir)
                if chroma_path.exists():
                    shutil.rmtree(chroma_path, ignore_errors=True)
                self._chroma_store = None
                logger.info("ChromaDB reset successfully.")
            except Exception as exc:
                logger.warning("Error resetting ChromaDB: %s", exc)

        if target in ("faiss", "all", "both"):
            try:
                faiss_path = Path(self.config.faiss_persist_dir)
                if faiss_path.exists():
                    shutil.rmtree(faiss_path, ignore_errors=True)
                self._faiss_store = None
                logger.info("FAISS index reset successfully.")
            except Exception as exc:
                logger.warning("Error resetting FAISS: %s", exc)
