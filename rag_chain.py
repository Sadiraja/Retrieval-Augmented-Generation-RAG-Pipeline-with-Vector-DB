"""
RAG Chain module for the RAG Chatbot Pipeline.
Implements history-aware query rewriting, retrieval-augmented QA chain,
source citation tracking, and streaming generation.
"""

from __future__ import annotations
import os
import re
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Iterator

from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    AIMessageChunk,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from config import RAGConfig, default_config
from vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Local Fallback LLM (Runs offline with 0 API keys for automated tests & demo)
# -----------------------------------------------------------------------------
class LocalFallbackLLM(BaseChatModel):
    """
    Offline extractive LLM fallback for local demonstrations and automated tests.
    Extracts high-relevance sentences from the retrieved context and formats citations.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_prompt = ""
        system_context = ""

        for m in messages:
            if isinstance(m, HumanMessage):
                user_prompt += f"\n{m.content}"
            elif isinstance(m, SystemMessage):
                system_context += f"\n{m.content}"
            else:
                user_prompt += f"\n{m.content}"

        # If this is a question reformulation step:
        if "formulate a standalone question" in user_prompt.lower() or "reformulate" in system_context.lower():
            # Extract the latest question
            lines = [line.strip() for line in user_prompt.strip().split("\n") if line.strip()]
            q = lines[-1] if lines else "What is the requested topic?"
            q = re.sub(r"^(Question:|User:)\s*", "", q, flags=re.IGNORECASE)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=q))])

        # If this is QA answer generation with context:
        # Check if context exists in prompt
        if "context:" in user_prompt.lower() or "context:" in system_context.lower():
            # Find context block
            full_text = f"{system_context}\n{user_prompt}"
            ctx_match = re.search(r"Context:\s*(.*?)(?:\n\nQuestion:|\nQuestion:|$)", full_text, re.DOTALL | re.IGNORECASE)
            context_body = ctx_match.group(1).strip() if ctx_match else full_text

            if not context_body or "no relevant documents" in context_body.lower():
                answer = "I do not have enough information in the provided knowledge base to answer this question."
            else:
                # Extract key sentences from the context
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context_body) if len(s.strip()) > 15]
                selected = sentences[:3] if sentences else [context_body[:200]]
                extracted_content = " ".join(selected)
                answer = (
                    f"Based on the provided documentation:\n\n"
                    f"{extracted_content}\n\n"
                    f"*(Answer synthesized by Local Extraction Engine)*"
                )
        else:
            answer = "I received your message. Please provide knowledge documents or ask a question based on indexed files."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    @property
    def _llm_type(self) -> str:
        return "local-extractive-fallback"


# -----------------------------------------------------------------------------
# Native Groq Chat Model (Direct High-Speed Groq SDK Integration)
# -----------------------------------------------------------------------------
class NativeGroqChatModel(BaseChatModel):
    """
    Native Groq Chat Model utilizing the official groq Python SDK.
    Supports ultra-low latency inference, full token streaming, and rich model options.
    """

    api_key: str
    model_name: str = "qwen/qwen3.8-27b"
    temperature: float = 0.2
    max_tokens: Optional[int] = 800

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        groq_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "user"
            groq_msgs.append({"role": role, "content": str(m.content)})
        return groq_msgs

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import groq

        client = groq.Groq(api_key=self.api_key)
        target_model = self.model_name

        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=self._convert_messages(messages),
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
            )
        except groq.NotFoundError:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=self._convert_messages(messages),
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
            )

        content = response.choices[0].message.content or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import groq

        client = groq.Groq(api_key=self.api_key)
        target_model = self.model_name

        try:
            stream_resp = client.chat.completions.create(
                model=target_model,
                messages=self._convert_messages(messages),
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except groq.NotFoundError:
            stream_resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=self._convert_messages(messages),
                temperature=self.temperature,
                stop=stop,
                max_tokens=self.max_tokens,
                stream=True,
            )

        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                cg_chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=cg_chunk)
                yield cg_chunk


# -----------------------------------------------------------------------------
# LLM Provider Factory
# -----------------------------------------------------------------------------
def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    streaming: bool = True,
) -> BaseChatModel:
    """Instantiate a ChatModel based on provider and API keys."""
    prov = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if prov == "groq":
        raw_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip().strip("'\"")
        if not raw_key:
            logger.warning("GROQ_API_KEY is empty or missing. Falling back to local extractor.")
            return LocalFallbackLLM()
        model = model_name or os.getenv("GROQ_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
        return NativeGroqChatModel(
            api_key=raw_key,
            model_name=model,
            temperature=temperature,
            max_tokens=800,
        )

    elif prov == "openai":
        raw_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip().strip("'\"")
        if not raw_key:
            logger.warning("OPENAI_API_KEY is empty or missing. Falling back to local extractor.")
            return LocalFallbackLLM()
        from langchain_community.chat_models import ChatOpenAI
        model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            openai_api_key=raw_key,
            temperature=temperature,
            streaming=streaming,
        )

    elif prov == "ollama":
        try:
            from langchain_community.chat_models import ChatOpenAI
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = model_name or os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
            return ChatOpenAI(
                model=model,
                openai_api_base=f"{base_url}/v1",
                openai_api_key="ollama",
                temperature=temperature,
                streaming=streaming,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Ollama: %s. Using LocalFallbackLLM.", exc)
            return LocalFallbackLLM()

    return LocalFallbackLLM()


# -----------------------------------------------------------------------------
# Response Container
# -----------------------------------------------------------------------------
@dataclass
class RAGResponse:
    question: str
    answer: str
    standalone_question: str
    source_documents: List[Document] = field(default_factory=list)
    vector_store: str = "chroma"

    def format_citations(self) -> str:
        """Format human-readable citations."""
        if not self.source_documents:
            return "No source documents retrieved."

        lines = ["### Citations & Retrieved Sources:"]
        seen = set()
        for i, doc in enumerate(self.source_documents, 1):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            chunk_idx = doc.metadata.get("chunk_index", 0)
            key = f"{src}-{page}-{chunk_idx}"
            if key in seen:
                continue
            seen.add(key)

            page_str = f" (Page {page})" if page else ""
            lines.append(f"- **[{i}] {src}{page_str}** [Chunk {chunk_idx}]")
            preview = doc.page_content.replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + "..."
            lines.append(f"  > *\"{preview}\"*")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# RAG Pipeline Class
# -----------------------------------------------------------------------------
class RAGPipeline:
    """
    End-to-End Conversational RAG Pipeline using LangChain LCEL.
    Features:
      - History-aware query contextualization
      - Multi-vectorstore retrieval (ChromaDB or FAISS)
      - Grounded QA answer generation
      - Source document extraction and citation pairing
    """

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    QA_SYSTEM_PROMPT = (
        "You are a helpful, professional, and accurate AI assistant powered by Retrieval-Augmented Generation (RAG).\n"
        "Answer the user's question using ONLY the provided context documents below.\n"
        "Strict Guidelines:\n"
        "1. If the context does not contain enough information to answer the question, explicitly state:\n"
        "   'I do not have enough information in the provided knowledge base to answer this question.'\n"
        "2. Do NOT invent or extrapolate facts beyond what is stated in the context.\n"
        "3. Explicitly cite your sources when referring to information, referencing the document name or page.\n\n"
        "Context:\n{context}"
    )

    QA_PROMPT = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm: Optional[BaseChatModel] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.config = config or default_config
        self.vsm = vector_store_manager or VectorStoreManager(config=self.config)
        self.llm = llm or get_llm(
            provider=self.config.llm_provider,
            model_name=self.config.openai_model_name,
            api_key=self.config.openai_api_key,
            temperature=self.config.temperature,
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents with metadata headers for the QA prompt."""
        if not docs:
            return "No relevant documents found in knowledge base."

        formatted_blocks = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", "N/A")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            header = f"[Document {i} | Source: {src} | Page: {page} | Chunk: {chunk_idx}]"
            formatted_blocks.append(f"{header}\n{doc.page_content.strip()}")

        return "\n\n".join(formatted_blocks)

    def _convert_history(
        self, chat_history: Optional[List[Tuple[str, str]]]
    ) -> List[BaseMessage]:
        """Convert a list of (user_msg, ai_msg) tuples into LangChain BaseMessage objects."""
        messages: List[BaseMessage] = []
        if not chat_history:
            return messages

        for user_text, ai_text in chat_history:
            if user_text:
                messages.append(HumanMessage(content=user_text))
            if ai_text:
                messages.append(AIMessage(content=ai_text))

        return messages

    def reformulate_question(
        self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """Reformulate conversational follow-up questions into standalone queries."""
        history_messages = self._convert_history(chat_history)
        if not history_messages:
            return question

        reformulator = self.REWRITE_PROMPT | self.llm | StrOutputParser()
        try:
            standalone = reformulator.invoke({
                "chat_history": history_messages,
                "question": question,
            })
            return standalone.strip()
        except Exception as exc:
            logger.warning("Error during query reformulation: %s. Using original question.", exc)
            return question

    def retrieve_context(
        self,
        query: str,
        store_type: Optional[str] = None,
        k: Optional[int] = None,
        search_type: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve relevant document chunks from the specified vector store."""
        target_store = store_type or self.config.vector_store_type
        top_k = k or self.config.top_k
        mode = search_type or self.config.search_type

        try:
            retriever = self.vsm.get_retriever(
                store_type=target_store,
                search_type=mode,
                k=top_k,
            )
            return retriever.invoke(query)
        except Exception as exc:
            logger.warning("Retrieval failed from %s: %s", target_store, exc)
            return []

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        store_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute full RAG generation:
        1. Contextualize query if chat history exists.
        2. Retrieve top-k chunks from vector store.
        3. Synthesize answer with context-grounded prompt.
        4. Return response with source documents.
        """
        target_store = store_type or self.config.vector_store_type
        standalone_q = self.reformulate_question(question, chat_history)
        retrieved_docs = self.retrieve_context(standalone_q, store_type=target_store, k=k)

        context_str = self._format_docs(retrieved_docs)
        history_messages = self._convert_history(chat_history)

        qa_chain = self.QA_PROMPT | self.llm | StrOutputParser()
        try:
            raw_answer = qa_chain.invoke({
                "context": context_str,
                "chat_history": history_messages,
                "question": standalone_q,
            })
            answer_text = raw_answer.strip()
        except Exception as exc:
            err_msg = str(exc)
            logger.error("LLM Generation Exception: %s", exc)
            if "invalid_api_key" in err_msg.lower() or "401" in err_msg or "authentication" in err_msg.lower():
                answer_text = (
                    f"❌ **Authentication Error**: The provided API key for `{self.llm._llm_type.upper()}` is invalid.\n\n"
                    f"Please check your API key in the Streamlit sidebar or `.env` file.\n"
                    f"- Get a free Groq API key: [https://console.groq.com/keys](https://console.groq.com/keys)\n\n"
                    f"*Technical details*: `{err_msg}`"
                )
            elif "rate_limit" in err_msg.lower() or "429" in err_msg:
                answer_text = f"⚠️ **Rate Limit**: API quota or rate limit reached for `{self.llm._llm_type.upper()}`. Please wait a moment or switch models."
            else:
                answer_text = f"⚠️ **Generation Error** ({self.llm._llm_type.upper()}): {err_msg}"

        return RAGResponse(
            question=question,
            answer=answer_text,
            standalone_question=standalone_q,
            source_documents=retrieved_docs,
            vector_store=target_store,
        )

    def stream(
        self,
        question: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        store_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream generation tokens for real-time interactive user interfaces."""
        target_store = store_type or self.config.vector_store_type
        standalone_q = self.reformulate_question(question, chat_history)
        retrieved_docs = self.retrieve_context(standalone_q, store_type=target_store, k=k)

        context_str = self._format_docs(retrieved_docs)
        history_messages = self._convert_history(chat_history)

        qa_chain = self.QA_PROMPT | self.llm | StrOutputParser()
        try:
            for chunk in qa_chain.stream({
                "context": context_str,
                "chat_history": history_messages,
                "question": standalone_q,
            }):
                yield chunk
        except Exception as exc:
            err_msg = str(exc)
            logger.error("LLM Streaming Exception: %s", exc)
            if "invalid_api_key" in err_msg.lower() or "401" in err_msg or "authentication" in err_msg.lower():
                yield (
                    f"\n\n❌ **Authentication Error**: The API key for `{self.llm._llm_type.upper()}` is invalid. "
                    f"Please check your key in the sidebar or `.env`. Get one at [console.groq.com/keys](https://console.groq.com/keys)."
                )
            else:
                yield f"\n\n⚠️ **Generation Error**: {err_msg}"

