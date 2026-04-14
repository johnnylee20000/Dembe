"""Main retriever-driven TTPS assistant logic over /vector_db."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

from tt_legal_agent.system_prompt import TT_LEGAL_SYSTEM_PROMPT


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    sources: list[dict[str, str]]
    retrieved_docs: list[Document]


class TTPSAgent:
    """Retriever + synthesis agent that always queries /vector_db first."""

    def __init__(
        self,
        vector_db_dir: str = "vector_db",
        collection_name: str = "tt_law_chunks",
        embedding_provider: str = "huggingface",
        hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        openai_embedding_model: str = "text-embedding-3-small",
        answer_model: str = "gpt-4o-mini",
    ) -> None:
        self.vector_db_dir = Path(vector_db_dir)
        if not self.vector_db_dir.exists():
            raise FileNotFoundError(
                f"Vector DB directory not found: {self.vector_db_dir}. Run ingest.py first."
            )

        self.embedding_provider = embedding_provider
        self.embeddings = self._build_embeddings(
            embedding_provider=embedding_provider,
            hf_model=hf_model,
            openai_embedding_model=openai_embedding_model,
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.vector_db_dir),
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.answer_model = answer_model
        self.has_openai = bool(os.getenv("OPENAI_API_KEY", "").strip())

    @staticmethod
    def _build_embeddings(
        embedding_provider: str,
        hf_model: str,
        openai_embedding_model: str,
    ):
        if embedding_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY required for OpenAI embeddings.")
            return OpenAIEmbeddings(model=openai_embedding_model)
        return HuggingFaceEmbeddings(model_name=hf_model)

    def _format_context(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            md = doc.metadata
            parts.append(
                f"[{idx}] Act: {md.get('act_name','Unknown')} | "
                f"Chapter: {md.get('chapter','Unknown')} | "
                f"Section: {md.get('section','Unknown')} | "
                f"URL: {md.get('url','')}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _sources_from_docs(docs: list[Document]) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for doc in docs:
            md = doc.metadata
            output.append(
                {
                    "act_name": str(md.get("act_name", "Unknown")),
                    "chapter": str(md.get("chapter", "Unknown")),
                    "section": str(md.get("section", "Unknown")),
                    "url": str(md.get("url", "")),
                    "source_file": str(md.get("source_file", "")),
                }
            )
        return output

    def answer(self, question: str, top_k: int = 3) -> dict[str, Any]:
        docs = self.vector_store.similarity_search(question, k=top_k)
        sources = self._sources_from_docs(docs)
        if not docs:
            return {
                "answer": "I do not know based on the provided legal sources.",
                "citations": [],
                "retrieved_docs": [],
            }

        context = self._format_context(docs)
        if not self.has_openai:
            bullets = []
            for idx, src in enumerate(sources, start=1):
                bullets.append(
                    f"- [{idx}] {src['act_name']} Chapter {src['chapter']} Section {src['section']} ({src['url']})"
                )
            fallback = (
                "OPENAI_API_KEY is not configured, so this is retrieval-only output.\n\n"
                "Most relevant legal sections found:\n"
                + "\n".join(bullets)
            )
            return {
                "answer": fallback,
                "citations": sources,
                "retrieved_docs": docs,
            }

        llm = ChatOpenAI(model=self.answer_model, temperature=0)
        user_prompt = (
            "Question:\n"
            f"{question.strip()}\n\n"
            "Retrieved legal context:\n"
            f"{context}\n\n"
            "Answer using only this context and cite specific Chapter/Section."
        )
        response = llm.invoke(
            [
                {"role": "system", "content": TT_LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        return {
            "answer": text.strip(),
            "citations": sources,
            "retrieved_docs": docs,
        }

    def list_loaded_acts(self) -> list[str]:
        """Return distinct act names currently stored in the backing collection."""
        raw = self.vector_store.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []
        acts = sorted(
            {
                str(metadata.get("act_name", "Unknown"))
                for metadata in metadatas
                if isinstance(metadata, dict)
            }
        )
        return acts


def get_loaded_acts(
    vector_db_dir: str = "vector_db",
    collection_name: str = "tt_law_chunks",
    embedding_provider: str = "huggingface",
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    openai_embedding_model: str = "text-embedding-3-small",
) -> list[str]:
    """Return list of distinct acts loaded in the vector DB."""
    embeddings = (
        OpenAIEmbeddings(model=openai_embedding_model)
        if embedding_provider == "openai"
        else HuggingFaceEmbeddings(model_name=hf_model)
    )
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(Path(vector_db_dir)),
    )
    raw = db.get(include=["metadatas"])
    metadatas = raw.get("metadatas") or []
    acts = sorted({str(md.get("act_name", "Unknown")) for md in metadatas if isinstance(md, dict)})
    return acts

