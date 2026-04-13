"""RAG orchestration for the Trinidad and Tobago legal assistant."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from openai import OpenAI

from .system_prompt import TT_LEGAL_SYSTEM_PROMPT, build_context_prompt
from .vector_store import RetrievedChunk, TTVectorStore


@dataclass(frozen=True)
class RAGAnswer:
    """Final answer and supporting context."""

    answer: str
    citations: list[dict[str, str]]
    retrieved: list[RetrievedChunk]


class TTRAGPipeline:
    """Retrieve relevant legal chunks and synthesize an answer."""

    def __init__(self, vector_store: TTVectorStore, model: str = "gpt-4o-mini") -> None:
        self.vector_store = vector_store
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def answer_question(self, question: str, top_k: int = 6) -> RAGAnswer:
        retrieved = self.vector_store.search(question, top_k=top_k)
        citations = [self._citation_from_metadata(chunk.metadata) for chunk in retrieved]
        context_blob = self._format_context(retrieved)

        if not self.client:
            return RAGAnswer(
                answer=(
                    "OPENAI_API_KEY is not configured. Retrieved legal context is available "
                    "below; configure the key to enable full synthesis.\n\n"
                    f"{context_blob}"
                ),
                citations=citations,
                retrieved=retrieved,
            )

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": TT_LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": build_context_prompt(question, context_blob)},
            ],
        )
        answer_text = completion.choices[0].message.content or ""
        return RAGAnswer(answer=answer_text.strip(), citations=citations, retrieved=retrieved)

    @staticmethod
    def _citation_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        return {
            "act_name": str(metadata.get("act_name", "Unknown Act")),
            "chapter": str(metadata.get("chapter", "Unknown Chapter")),
            "section": str(metadata.get("section", "Unknown Section")),
            "url": str(metadata.get("url", metadata.get("source_url", ""))),
        }

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            label = (
                f"[{idx}] Act: {meta.get('act_name', 'Unknown')} | "
                f"Chapter: {meta.get('chapter', 'N/A')} | "
                f"Section: {meta.get('section', 'N/A')} | "
                f"URL: {meta.get('url', meta.get('source_url', ''))}"
            )
            parts.append(f"{label}\n{chunk.text}")
        return "\n\n".join(parts)

