"""
Unit and Integration Tests for the RAG Chatbot Pipeline.
Tests document loaders, chunking, embeddings, ChromaDB, FAISS, and the RAG chain.
"""

import os
import shutil
import pytest
from pathlib import Path

from langchain_core.documents import Document
from config import RAGConfig
from embeddings import LocalMiniLMEmbeddings, get_embeddings
from loaders import load_from_uploaded_file, load_txt_or_md
from chunking import split_documents
from vectorstore import VectorStoreManager
from rag_chain import RAGPipeline, LocalFallbackLLM, RAGResponse


@pytest.fixture
def temp_config(tmp_path):
    """Provides an isolated RAGConfig with temporary storage paths."""
    chroma_dir = str(tmp_path / "chroma_db")
    faiss_dir = str(tmp_path / "faiss_index")
    cfg = RAGConfig(
        vector_store_type="chroma",
        chroma_persist_dir=chroma_dir,
        faiss_persist_dir=faiss_dir,
        collection_name="test_collection",
        chunk_size=300,
        chunk_overlap=50,
        top_k=2,
        embedding_provider="local_minilm",
        llm_provider="mock",
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def sample_documents():
    """Provides sample documents covering specific topics for deterministic testing."""
    return [
        Document(
            page_content=(
                "ChromaDB is an open-source vector database built for AI applications. "
                "It uses SQLite and DuckDB for metadata and persistent storage. "
                "ChromaDB supports collections and cosine distance filtering."
            ),
            metadata={"source": "chroma_guide.md", "page": 1},
        ),
        Document(
            page_content=(
                "FAISS was created by Meta AI Research for fast similarity search. "
                "It features GPU and multi-core CPU indexing such as IndexFlatL2 and IndexHNSW. "
                "FAISS can process billions of vectors with low latency."
            ),
            metadata={"source": "faiss_guide.md", "page": 1},
        ),
    ]


# -----------------------------------------------------------------------------
# 1. Embeddings Tests
# -----------------------------------------------------------------------------
def test_local_minilm_embeddings():
    emb = LocalMiniLMEmbeddings()
    vector = emb.embed_query("Vector search testing")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)

    batch = emb.embed_documents(["First document", "Second document"])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    assert len(batch[1]) == 384


# -----------------------------------------------------------------------------
# 2. Loaders and Chunking Tests
# -----------------------------------------------------------------------------
def test_load_from_uploaded_bytes():
    raw_text = "This is a test uploaded document for RAG ingestion."
    docs = load_from_uploaded_file("sample.txt", raw_text.encode("utf-8"))
    assert len(docs) == 1
    assert docs[0].page_content == raw_text
    assert docs[0].metadata["source"] == "sample.txt"


def test_chunking_metadata():
    doc = Document(
        page_content="Paragraph one about vector search.\n\nParagraph two about generation.\n\nParagraph three.",
        metadata={"source": "manual.txt"},
    )
    chunks = split_documents([doc], chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "chunk_char_count" in chunk.metadata
        assert chunk.metadata["source"] == "manual.txt"


# -----------------------------------------------------------------------------
# 3. ChromaDB Vector Store Tests
# -----------------------------------------------------------------------------
def test_chroma_indexing_and_retrieval(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    added = vsm.add_documents(chunks, store_type="chroma")
    assert added == len(chunks)

    # Search for Chroma content
    results = vsm.similarity_search_with_score("What storage does Chroma use?", store_type="chroma", k=1)
    assert len(results) > 0
    top_doc, score = results[0]
    assert "sqlite" in top_doc.page_content.lower() or "duckdb" in top_doc.page_content.lower()

    # Verify stats
    stats = vsm.get_stats()
    assert stats["chroma"]["count"] >= 2


# -----------------------------------------------------------------------------
# 4. FAISS Vector Store Tests
# -----------------------------------------------------------------------------
def test_faiss_indexing_and_persistence(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    added = vsm.add_documents(chunks, store_type="faiss")
    assert added == len(chunks)

    # Search for FAISS content
    results = vsm.similarity_search_with_score("Who created FAISS?", store_type="faiss", k=1)
    assert len(results) > 0
    top_doc, score = results[0]
    assert "meta" in top_doc.page_content.lower() or "research" in top_doc.page_content.lower()

    # Re-instantiate manager and check loaded persistence
    vsm_reloaded = VectorStoreManager(config=temp_config)
    retriever = vsm_reloaded.get_retriever(store_type="faiss", k=1)
    retrieved = retriever.invoke("Meta AI Research")
    assert len(retrieved) > 0


# -----------------------------------------------------------------------------
# 5. RAG Pipeline End-to-End Tests
# -----------------------------------------------------------------------------
def test_rag_pipeline_ask_chroma(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    vsm.add_documents(chunks, store_type="chroma")

    pipeline = RAGPipeline(
        vector_store_manager=vsm,
        llm=LocalFallbackLLM(),
        config=temp_config,
    )

    response = pipeline.ask("What is ChromaDB?", store_type="chroma", k=2)
    assert isinstance(response, RAGResponse)
    assert len(response.source_documents) > 0
    assert len(response.answer) > 0
    assert "chroma" in response.answer.lower()
    citations = response.format_citations()
    assert "chroma_guide.md" in citations


def test_rag_pipeline_ask_faiss(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    vsm.add_documents(chunks, store_type="faiss")

    pipeline = RAGPipeline(
        vector_store_manager=vsm,
        llm=LocalFallbackLLM(),
        config=temp_config,
    )

    response = pipeline.ask("What is FAISS used for?", store_type="faiss", k=2)
    assert isinstance(response, RAGResponse)
    assert len(response.source_documents) > 0
    assert "faiss" in response.answer.lower() or "meta" in response.answer.lower()


def test_rag_conversational_history(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    vsm.add_documents(chunks, store_type="chroma")

    pipeline = RAGPipeline(
        vector_store_manager=vsm,
        llm=LocalFallbackLLM(),
        config=temp_config,
    )

    history = [
        ("Tell me about ChromaDB", "ChromaDB is an open-source vector database using SQLite."),
    ]
    # Follow-up question referencing history
    response = pipeline.ask("What database does it use?", chat_history=history, store_type="chroma")
    assert response.answer is not None
    assert len(response.source_documents) > 0


def test_rag_streaming(temp_config, sample_documents):
    vsm = VectorStoreManager(config=temp_config)
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=30)
    vsm.add_documents(chunks, store_type="chroma")

    pipeline = RAGPipeline(
        vector_store_manager=vsm,
        llm=LocalFallbackLLM(),
        config=temp_config,
    )

    stream_chunks = list(pipeline.stream("What is ChromaDB?", store_type="chroma"))
    assert len(stream_chunks) > 0
    combined = "".join(stream_chunks)
    assert len(combined) > 0


def test_native_groq_initialization():
    from rag_chain import NativeGroqChatModel, get_llm
    llm = get_llm(provider="groq", api_key="gsk_dummy_test_key", model_name="llama-3.3-70b-versatile")
    assert isinstance(llm, NativeGroqChatModel)
    assert llm.model_name == "llama-3.3-70b-versatile"
    assert llm._llm_type == "groq"

