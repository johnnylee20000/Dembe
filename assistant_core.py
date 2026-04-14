#!/usr/bin/env python3
"""
Shared backend utilities for the TTPS assistant.

This module provides:
- Vector DB path resolution
- Embeddings + LLM initialization (OpenAI or local Ollama)
- LangChain RetrievalQA chain construction (stuff chain)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import requests


DEFAULT_QA_PROMPT = """You are the TTPS Legal Assistant.
Answer only using the retrieved legal context.
If the answer is not present, say you are unsure and advise checking official revised laws.
When possible, cite the Act and Section from context.

Question:
{question}

Context:
{context}

Answer:"""


def resolve_vector_db_path(preferred_path: str = "/vector_db") -> Path:
    """
    Resolve vector DB path from expected deployment and local development locations.
    """
    candidates = [
        Path(preferred_path),
        Path.cwd() / "vector_db",
        Path("/workspace/vector_db"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Return preferred even if missing, so caller can show a clear error.
    return Path(preferred_path)


def build_embeddings(embedding_provider: Optional[str] = None):
    """
    Build embedding model.
    - OpenAI if OPENAI_API_KEY exists or provider=openai.
    - Local HuggingFace fallback otherwise.
    """
    provider = (embedding_provider or os.getenv("EMBEDDING_PROVIDER", "")).strip().lower()
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    return HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def build_llm(llm_provider: Optional[str] = None, model_name: Optional[str] = None):
    """
    Build chat model.
    - OpenAI GPT-4o by default when OPENAI_API_KEY exists.
    - Local Ollama fallback otherwise.
    """
    provider = (llm_provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set for OpenAI provider.")
        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
    if provider == "ollama":
        if not is_ollama_available():
            raise RuntimeError("Ollama provider selected but Ollama service is unavailable.")
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )

    # Auto mode: OpenAI first, then local Ollama.
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
    if is_ollama_available():
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "llama3.1"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
    raise RuntimeError(
        "No LLM runtime available. Set OPENAI_API_KEY for GPT-4o or start a local Ollama server."
    )


def is_ollama_available() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def retrieve_documents(
    query: str,
    *,
    vector_db_path: str = "/vector_db",
    embedding_provider: Optional[str] = None,
    k: int = 6,
):
    vector_store = load_vector_store(vector_db_path=vector_db_path, embedding_provider=embedding_provider)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def retrieval_only_search(
    query: str,
    *,
    vector_db_path: str = "/vector_db",
    embedding_provider: Optional[str] = None,
    k: int = 6,
) -> list[dict]:
    docs = retrieve_documents(
        query,
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    return format_source_documents(docs)


def build_ollama_chat_model(model_name: Optional[str] = None):
    return ChatOllama(
        model=model_name or os.getenv("OLLAMA_MODEL", "llama3.1"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    )


def load_vector_store(vector_db_path: str = "/vector_db", embedding_provider: Optional[str] = None) -> Chroma:
    resolved = resolve_vector_db_path(vector_db_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Vector DB not found at '{resolved}'. "
            "Build/populate the Chroma database first."
        )
    embeddings = build_embeddings(embedding_provider=embedding_provider)
    return Chroma(
        persist_directory=str(resolved),
        embedding_function=embeddings,
    )


def build_retriever(
    vector_db_path: str = "/vector_db",
    embedding_provider: Optional[str] = None,
    k: int = 6,
):
    vector_store = load_vector_store(vector_db_path=vector_db_path, embedding_provider=embedding_provider)
    return vector_store.as_retriever(search_kwargs={"k": k})


def build_retrieval_qa_chain(
    vector_db_path: str = "/vector_db",
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    k: int = 6,
):
    """
    Build a RetrievalQA chain using:
    - Chroma local store
    - Stuff chain type
    - return_source_documents=True
    """
    retriever = build_retriever(
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    llm = build_llm(llm_provider=llm_provider, model_name=model_name)
    prompt = PromptTemplate(
        template=DEFAULT_QA_PROMPT,
        input_variables=["question", "context"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def format_source_documents(source_documents) -> list[dict]:
    output: list[dict] = []
    for doc in source_documents or []:
        metadata = dict(doc.metadata or {})
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        output.append(
            {
                "metadata": metadata,
                "snippet": snippet[:320] + ("..." if len(snippet) > 320 else ""),
            }
        )
    return output
