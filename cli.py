"""
Command-Line Interface (CLI) for the RAG Chatbot Pipeline.
Supports ingestion, single query answering, interactive conversation,
database statistics, and storage resets.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

from config import default_config, RAGConfig
from embeddings import get_embeddings
from loaders import load_document, load_directory
from chunking import split_documents
from vectorstore import VectorStoreManager
from rag_chain import RAGPipeline


def print_banner():
    banner = r"""
===========================================================
  Retrieval-Augmented Generation (RAG) Pipeline CLI
  Powered by LangChain, ChromaDB & FAISS
===========================================================
    """
    print(banner)


def cmd_ingest(args):
    """Ingest a file or directory into the vector store."""
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"[Error] Path does not exist: {target_path}")
        sys.exit(1)

    print(f"[*] Ingesting documents from: {target_path}")
    if target_path.is_file():
        raw_docs = load_document(target_path)
    else:
        raw_docs = load_directory(target_path)

    if not raw_docs:
        print("[Warning] No valid documents found to ingest.")
        return

    print(f"[*] Loaded {len(raw_docs)} document segment(s). Splitting into chunks...")
    chunks = split_documents(
        raw_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    print(f"[*] Created {len(chunks)} chunks.")

    print(f"[*] Embedding and indexing into vector store: '{args.store}'...")
    cfg = RAGConfig(
        vector_store_type=args.store,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    vsm = VectorStoreManager(config=cfg)
    added = vsm.add_documents(chunks, store_type=args.store)
    print(f"[SUCCESS] Successfully indexed {added} chunks into '{args.store}'!")

    # Display updated stats
    stats = vsm.get_stats()
    print(f"[*] Current Vector Store Stats: {stats}")


def cmd_query(args):
    """Execute a single question against the RAG pipeline."""
    llm_prov = getattr(args, "llm", None) or default_config.llm_provider
    api_k = getattr(args, "api_key", None)
    cfg = RAGConfig(
        vector_store_type=args.store,
        top_k=args.top_k,
        llm_provider=llm_prov,
    )
    vsm = VectorStoreManager(config=cfg)
    from rag_chain import get_llm
    llm_obj = get_llm(provider=llm_prov, api_key=api_k)
    pipeline = RAGPipeline(vector_store_manager=vsm, llm=llm_obj, config=cfg)

    print(f"\n[Question]: {args.question}")
    print(f"[*] Searching vector store: '{args.store}' (top_k={args.top_k}) | LLM: {llm_obj._llm_type}...\n")

    response = pipeline.ask(args.question, store_type=args.store, k=args.top_k)

    print("-" * 60)
    print(f"[Answer]:\n{response.answer}")
    print("-" * 60)
    print(f"\n{response.format_citations()}\n")


def cmd_chat(args):
    """Start an interactive multi-turn chat session with conversational memory."""
    print_banner()
    llm_prov = getattr(args, "llm", None) or default_config.llm_provider
    api_k = getattr(args, "api_key", None)
    print(f"[*] Initializing chat using vector store: '{args.store}' | LLM: '{llm_prov}'")
    print("[*] Type 'exit', 'quit', or 'q' to end the session.")
    print("[*] Type 'clear' to reset chat conversation memory.\n")

    cfg = RAGConfig(
        vector_store_type=args.store,
        top_k=args.top_k,
        llm_provider=llm_prov,
    )
    vsm = VectorStoreManager(config=cfg)
    from rag_chain import get_llm
    llm_obj = get_llm(provider=llm_prov, api_key=api_k)
    pipeline = RAGPipeline(vector_store_manager=vsm, llm=llm_obj, config=cfg)

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Session closed. Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[*] Conversation memory cleared.")
            continue

        print("\nThinking...", end="\r")
        response = pipeline.ask(
            user_input,
            chat_history=history,
            store_type=args.store,
            k=args.top_k,
        )

        print(f"\nBot > {response.answer}\n")
        if response.source_documents:
            print(response.format_citations())

        history.append((user_input, response.answer))


def cmd_stats(args):
    """Display vector store statistics."""
    vsm = VectorStoreManager()
    stats = vsm.get_stats()
    print("\n=== Vector Store Status & Chunk Counts ===")
    print(f"ChromaDB : {stats['chroma']['count']} vectors (Status: {stats['chroma']['status']})")
    print(f"FAISS    : {stats['faiss']['count']} vectors (Status: {stats['faiss']['status']})")
    print("==========================================\n")


def cmd_reset(args):
    """Reset vector stores."""
    vsm = VectorStoreManager()
    confirm = input(f"Are you sure you want to reset store '{args.store}'? (y/N): ").strip().lower()
    if confirm in ("y", "yes"):
        vsm.reset_store(args.store)
        print(f"[SUCCESS] Vector store '{args.store}' was reset.")
    else:
        print("[*] Operation cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot Pipeline CLI with ChromaDB and FAISS"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into vector store")
    p_ingest.add_argument("--path", required=True, help="Path to file or directory to ingest")
    p_ingest.add_argument("--store", default="chroma", choices=["chroma", "faiss", "both"], help="Target vector store")
    p_ingest.add_argument("--chunk-size", type=int, default=1000, help="Chunk character length")
    p_ingest.add_argument("--overlap", type=int, default=200, help="Chunk overlap length")
    p_ingest.set_defaults(func=cmd_ingest)

    # Query command
    p_query = subparsers.add_parser("query", help="Ask a single question")
    p_query.add_argument("--question", "-q", required=True, help="Question to ask")
    p_query.add_argument("--store", default="chroma", choices=["chroma", "faiss"], help="Vector store to search")
    p_query.add_argument("--top-k", "-k", type=int, default=4, help="Number of retrieved chunks")
    p_query.add_argument("--llm", default=None, choices=["groq", "openai", "mock"], help="LLM provider (groq, openai, mock)")
    p_query.add_argument("--api-key", default=None, help="API key for chosen provider")
    p_query.set_defaults(func=cmd_query)

    # Chat command
    p_chat = subparsers.add_parser("chat", help="Start interactive multi-turn chat session")
    p_chat.add_argument("--store", default="chroma", choices=["chroma", "faiss"], help="Vector store to query")
    p_chat.add_argument("--top-k", "-k", type=int, default=4, help="Number of retrieved chunks")
    p_chat.add_argument("--llm", default=None, choices=["groq", "openai", "mock"], help="LLM provider (groq, openai, mock)")
    p_chat.add_argument("--api-key", default=None, help="API key for chosen provider")
    p_chat.set_defaults(func=cmd_chat)

    # Stats command
    p_stats = subparsers.add_parser("stats", help="Check vector store statistics")
    p_stats.set_defaults(func=cmd_stats)

    # Reset command
    p_reset = subparsers.add_parser("reset", help="Reset vector stores")
    p_reset.add_argument("--store", default="all", choices=["chroma", "faiss", "all"], help="Vector store to clear")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
