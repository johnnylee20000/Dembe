"""Embedding generation and ChromaDB storage for T&T legal chunks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from .chunker import LegalChunk


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding backend and model configuration."""

    backend: str = "local"
    openai_model: str = "text-embedding-3-small"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: str | None = None

    def validate(self) -> None:
        if self.backend not in {"local", "openai"}:
            raise ValueError("backend must be one of: local, openai")
        if self.backend == "openai" and not (self.openai_api_key or os.getenv("OPENAI_API_KEY")):
            raise ValueError("OPENAI_API_KEY is required when backend is openai")


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk returned from semantic retrieval."""

    text: str
    metadata: dict[str, Any]
    distance: float


class TTVectorStore:
    """Chroma vector store wrapper for upsert and semantic search."""

    def __init__(
        self,
        persist_directory: str = ".chroma_tt_law",
        collection_name: str = "tt_law",
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedding_config.validate()
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self._local_model: SentenceTransformer | None = None
        self._openai_client: OpenAI | None = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_config.backend == "openai":
            api_key = self.embedding_config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if self._openai_client is None:
                self._openai_client = OpenAI(api_key=api_key)
            response = self._openai_client.embeddings.create(
                model=self.embedding_config.openai_model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        if self._local_model is None:
            self._local_model = SentenceTransformer(self.embedding_config.local_model)
        vectors = self._local_model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def upsert_chunks(self, chunks: list[LegalChunk]) -> int:
        """Embed chunk texts and upsert into Chroma with metadata."""
        if not chunks:
            return 0
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]
        embeddings = self._embed(texts)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve semantically similar legal chunks."""
        query_embedding = self._embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                text=str(text),
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]
