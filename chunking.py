"""
Text chunking module for the RAG Chatbot Pipeline.
Implements recursive character splitting with metadata enrichment.
"""

from __future__ import annotations
from typing import List, Optional
import hashlib
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def get_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> RecursiveCharacterTextSplitter:
    """Instantiate a RecursiveCharacterTextSplitter with customizable parameters."""
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split a list of documents into enriched chunks.
    Attaches chunk index, chunk ID, and character counts to metadata.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    enriched_chunks: List[Document] = []
    chunk_counters: dict[str, int] = {}

    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        src = str(chunk.metadata.get("source", "unknown"))
        idx = chunk_counters.get(src, 0)
        chunk_counters[src] = idx + 1

        meta = dict(chunk.metadata)
        meta["chunk_index"] = idx
        meta["chunk_id"] = f"{src}#chunk_{idx}"
        meta["chunk_char_count"] = len(text)
        # Content hash to identify identical chunks
        meta["chunk_hash"] = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

        enriched_chunks.append(Document(page_content=text, metadata=meta))

    logger.info(
        "Split %d documents into %d chunks (chunk_size=%d, overlap=%d)",
        len(documents),
        len(enriched_chunks),
        chunk_size,
        chunk_overlap,
    )
    return enriched_chunks
