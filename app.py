"""
Streamlit Web Application for RAG Chatbot Pipeline.
Interactive UI supporting document uploading, dual vector stores (ChromaDB / FAISS),
multi-model selection, streaming responses, and source citation inspection.
"""

from __future__ import annotations
import os
import streamlit as st
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Ensure environment variables are loaded (with .env.example fallback)
load_dotenv(override=True)
_base_dir = Path(__file__).resolve().parent
if not os.getenv("GROQ_API_KEY") and (_base_dir / ".env.example").exists():
    load_dotenv(_base_dir / ".env.example", override=False)

from config import RAGConfig, default_config
from embeddings import get_embeddings
from loaders import load_from_uploaded_file, load_document
from chunking import split_documents
from vectorstore import VectorStoreManager
from rag_chain import RAGPipeline, get_llm, LocalFallbackLLM, NativeGroqChatModel, RAGResponse

# Page Setup
st.set_page_config(
    page_title="RAG Chatbot Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.55em;
        font-size: 0.85rem;
        font-weight: 600;
        border-radius: 0.35rem;
        margin-right: 0.5rem;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
    }
    .active-badge-groq {
        display: inline-block;
        padding: 0.35em 0.75em;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 0.4rem;
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        margin-bottom: 1rem;
    }
    .active-badge-offline {
        display: inline-block;
        padding: 0.35em 0.75em;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 0.4rem;
        background-color: #78350f;
        color: #fde68a;
        border: 1px solid #b45309;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []




# -----------------------------------------------------------------------------
# Sidebar: Settings & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ RAG Engine Settings")

    # Vector Store Choice
    vector_store = st.selectbox(
        "Vector Store Backend",
        options=["chroma", "faiss"],
        index=0,
        format_func=lambda x: "ChromaDB (Persistent SQLite)" if x == "chroma" else "FAISS (In-Memory / Fast Index)",
        help="Select which vector database indexes and retrieves your documents.",
    )

    # Embedding Provider Choice
    embedding_choice = st.selectbox(
        "Embedding Model",
        options=["local_minilm", "openai"],
        index=0,
        format_func=lambda x: "Local MiniLM-L6-v2 (100% Free / Offline)" if x == "local_minilm" else "OpenAI text-embedding-3-small",
        help="Local MiniLM runs 100% locally on CPU without requiring an API key.",
    )

    st.divider()

    # Retrieval Tuning
    st.subheader("🎯 Retrieval Parameters")
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=4)
    search_type = st.radio(
        "Search Mode",
        options=["similarity", "mmr"],
        index=0,
        format_func=lambda x: "Standard Similarity" if x == "similarity" else "MMR (Diverse Results)",
    )
    chunk_size = st.slider("Chunk Size (characters)", min_value=400, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=200, step=20)

    st.divider()

    # Vector Store Stats & Reset
    st.subheader("📊 Vector DB Status")
    vsm_stat = VectorStoreManager()
    stats = vsm_stat.get_stats()
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Chroma Chunks", stats["chroma"]["count"])
    with col_stat2:
        st.metric("FAISS Chunks", stats["faiss"]["count"])

    if st.button("🗑️ Reset All Vector Stores", use_container_width=True):
        vsm_stat.reset_store("all")
        st.session_state.messages = []
        st.session_state.uploaded_filenames = []
        st.success("Vector databases reset!")
        st.rerun()


# -----------------------------------------------------------------------------
# Main View: Header & Ingestion
# -----------------------------------------------------------------------------
st.title("🧠 Retrieval-Augmented Generation (RAG) Pipeline")

st.markdown("""
Ask questions grounded strictly in your documents. Choose between **ChromaDB** and **FAISS** vector stores, with conversational memory and verifiable citations.
""")

# Ingestion Expander
with st.expander("📁 Document Ingestion & Knowledge Base Management", expanded=False):
    tab1, tab2 = st.tabs(["Upload Custom Documents", "Load Pre-built Knowledge Base"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload knowledge base files (.pdf, .docx, .txt, .md)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if st.button("📥 Process & Index Uploaded Documents", type="primary"):
            if not uploaded_files:
                st.warning("Please upload one or more files first.")
            else:
                progress_bar = st.progress(0, text="Reading files...")
                all_raw_docs = []
                for i, uf in enumerate(uploaded_files):
                    file_bytes = uf.read()
                    docs = load_from_uploaded_file(uf.name, file_bytes)
                    all_raw_docs.extend(docs)
                    progress_bar.progress(int((i + 1) / len(uploaded_files) * 50), text=f"Read {uf.name}")

                progress_bar.progress(60, text="Splitting into semantic chunks...")
                chunks = split_documents(all_raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                progress_bar.progress(80, text=f"Generating embeddings and indexing into {vector_store.upper()}...")
                vsm_ingest = VectorStoreManager(
                    config=RAGConfig(
                        vector_store_type=vector_store,
                        embedding_provider=embedding_choice,
                    )
                )
                vsm_ingest.add_documents(chunks, store_type=vector_store)
                progress_bar.progress(100, text="Indexing complete!")
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s) into **{vector_store.upper()}**!")
                st.rerun()

    with tab2:
        st.markdown("Index the included sample knowledge base on **RAG Architecture & Vector Databases**:")
        sample_path = Path(__file__).resolve().parent / "sample_data" / "knowledge_base.md"
        if st.button("🚀 Index Sample Knowledge Base Now"):
            if sample_path.exists():
                docs = load_document(sample_path)
                chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                vsm_ingest = VectorStoreManager(
                    config=RAGConfig(
                        vector_store_type=vector_store,
                        embedding_provider=embedding_choice,
                    )
                )
                vsm_ingest.add_documents(chunks, store_type=vector_store)
                st.success(f"Indexed sample knowledge base ({len(chunks)} chunks) into **{vector_store.upper()}**!")
                st.rerun()
            else:
                st.error("Sample document not found at sample_data/knowledge_base.md")


# -----------------------------------------------------------------------------
# Chat Pipeline Setup
# -----------------------------------------------------------------------------
# Resolve active LLM automatically from .env
active_provider = os.getenv("LLM_PROVIDER", "groq")
active_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
active_model = os.getenv("GROQ_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")

llm_instance = get_llm(
    provider=active_provider,
    model_name=active_model,
    api_key=active_key,
    temperature=0.2,
)

active_config = RAGConfig(
    vector_store_type=vector_store,
    embedding_provider=embedding_choice,
    top_k=top_k,
    search_type=search_type,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
vsm = VectorStoreManager(
    config=active_config,
    embeddings=get_embeddings(provider=embedding_choice, api_key=active_key),
)
pipeline = RAGPipeline(
    vector_store_manager=vsm,
    llm=llm_instance,
    config=active_config,
)


# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
st.subheader("💬 Interactive RAG Conversation")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("🔍 Citations & Retrieved Context", expanded=False):
                st.markdown(msg["citations"])


# Handle new user query
if user_prompt := st.chat_input("Ask a question based on your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Format conversational history
    chat_history: List[Tuple[str, str]] = []
    for i in range(0, len(st.session_state.messages) - 1, 2):
        if i + 1 < len(st.session_state.messages):
            u_text = st.session_state.messages[i]["content"]
            a_text = st.session_state.messages[i + 1]["content"]
            chat_history.append((u_text, a_text))

    # Generate assistant answer with streaming
    with st.chat_message("assistant"):
        # Contextualize query and retrieve documents first
        standalone_q = pipeline.reformulate_question(user_prompt, chat_history)
        retrieved_docs = pipeline.retrieve_context(standalone_q, store_type=vector_store, k=top_k)

        # Stream generation
        full_answer = st.write_stream(
            pipeline.stream(
                question=user_prompt,
                chat_history=chat_history,
                store_type=vector_store,
                k=top_k,
            )
        )

        # Prepare Citations
        resp = RAGResponse(
            question=user_prompt,
            answer=full_answer or "",
            standalone_question=standalone_q,
            source_documents=retrieved_docs,
            vector_store=vector_store,
        )
        citations_text = resp.format_citations()

        if retrieved_docs:
            with st.expander("🔍 Citations & Retrieved Context", expanded=False):
                st.markdown(citations_text)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "citations": citations_text if retrieved_docs else None,
        })

# Chat Controls
col_ctrl1, col_ctrl2 = st.columns([1, 8])
with col_ctrl1:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
