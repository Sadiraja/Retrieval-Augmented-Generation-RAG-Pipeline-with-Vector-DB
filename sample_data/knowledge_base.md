# Enterprise Knowledge Base: Retrieval-Augmented Generation & Vector Databases

## 1. Overview of Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) is an architectural pattern that combines information retrieval systems with generative large language models (LLMs). Instead of relying solely on the static parametric memory stored in the neural network weights during training, RAG retrieves relevant document chunks dynamically from an external non-parametric knowledge base and provides them as grounding context within the prompt.

The RAG workflow consists of three primary stages:
1. **Ingestion & Indexing**: Raw documents (PDFs, Word documents, text files, Markdown) are extracted, preprocessed, split into manageable chunks, transformed into continuous numerical vectors via embedding models, and indexed in a vector database.
2. **Retrieval**: When a user submits a prompt, the user query is converted into a vector in the identical embedding space. The vector store conducts approximate nearest neighbor (ANN) or exact search to identify the top-k most semantically similar chunks.
3. **Generation**: The retrieved context chunks are prepended to the system prompt and sent to the LLM, which synthesizes a well-grounded, factual answer citing the provided sources.

## 2. Vector Databases: ChromaDB vs FAISS

### ChromaDB
ChromaDB is an open-source, AI-native embedding database designed for developer productivity and local-first or distributed deployments.
- **Key Characteristics**:
  - Embedded execution using DuckDB/SQLite for persistent metadata and vector storage.
  - Native collection management, automatic persistence to disk, and built-in filtering by metadata attributes.
  - Built-in embedding model support (e.g. ONNX MiniLM) enabling zero-dependency offline local setups.
  - Ideal for full-featured RAG applications where document lifecycle management, update/delete operations, and metadata filtering are essential.

### FAISS (Facebook AI Similarity Search)
FAISS is an industrial-strength vector similarity search library developed by Meta AI Research.
- **Key Characteristics**:
  - Written in optimized C++ with complete Python bindings for GPU and multi-core CPU execution.
  - Exceptional similarity search speed and memory efficiency for billion-scale vector collections.
  - Supports versatile indexing structures such as `IndexFlatL2` (exact brute-force L2 distance), `IndexFlatIP` (inner product / cosine similarity), and `IndexHNSWFlat` (Hierarchical Navigable Small World graphs for sub-millisecond retrieval).
  - Can serialize indexes directly to disk as binary `.faiss` files.

### Comparison Table
| Feature | ChromaDB | FAISS |
| :--- | :--- | :--- |
| **Primary Focus** | Complete Vector Database | High-Performance Indexing Library |
| **Storage Engine** | SQLite / DuckDB + HNSW | In-memory with binary serialization |
| **Metadata Filtering** | Built-in rich SQL/dict queries | Metadata managed in separate wrapper (e.g. LangChain) |
| **Hardware Acceleration** | CPU | Highly optimized CPU (AVX2/AVX512) & CUDA GPU |
| **Best For** | Production apps, developer velocity | High-throughput, large-scale search benchmarking |

## 3. Document Ingestion & Chunking Strategies
Effective chunking is critical to RAG performance. Chunks that are too small lack contextual meaning, while chunks that are too large dilute vector specificity and exceed model context windows.
- **Recursive Character Splitting**: The recommended approach splits by paragraphs (`\n\n`), then sentences (`\n`), and finally words (` `), maintaining semantic boundary coherence.
- **Chunk Size**: Typically 500 to 1,500 characters or 200 to 400 tokens for general text.
- **Chunk Overlap**: Typically 10% to 20% (e.g., 100 to 200 characters). Overlap ensures that sentences spanning the boundary between two adjacent chunks are not severed, preserving semantic flow.

## 4. Query Reformulation & Conversational Memory
In conversational RAG, follow-up queries (such as "How does it compare to the first one?") lack standalone clarity. The pipeline addresses this by:
1. Retaining recent conversational turns in memory.
2. Utilizing a query reformulation sub-chain to rewrite the query into an unambiguous standalone search string before querying the vector store.
3. Feeding the retrieved documents and complete conversation history to the synthesis LLM.

## 5. RAG Evaluation Metrics (The RAG Triad)
Enterprise RAG pipelines should be measured against three core pillars:
1. **Context Relevance**: Measures whether the retrieved documents are pertinent and strictly relevant to the user query, avoiding noisy distractors.
2. **Groundedness / Faithfulness**: Measures whether the generated claims in the LLM's response are fully supported and can be traced directly to the retrieved context chunks, preventing hallucination.
3. **Answer Relevance**: Measures whether the synthesized answer directly and comprehensively addresses the user's inquiry.
