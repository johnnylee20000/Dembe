"""Ingest chunked T&T law files into a local Chroma vector database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest /chunks files into local Chroma /vector_db."
    )
    parser.add_argument(
        "--chunks-dir",
        default="chunks",
        help="Directory containing chunked JSON/Markdown files.",
    )
    parser.add_argument(
        "--vector-db-dir",
        default="vector_db",
        help="Directory to persist local Chroma vector DB.",
    )
    parser.add_argument(
        "--collection-name",
        default="tt_law_chunks",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("openai", "huggingface"),
        default="huggingface",
        help="Embedding backend.",
    )
    parser.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when provider=openai.",
    )
    parser.add_argument(
        "--hf-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model when provider=huggingface.",
    )
    return parser.parse_args()


def _extract_markdown_metadata(text: str, source_file: Path) -> tuple[str, dict[str, Any]]:
    act = _extract_field(text, r"^\s*Act(?:\s+Name)?\s*:\s*(.+?)\s*$", source_file.stem)
    chapter = _extract_field(text, r"^\s*Chapter\s*:\s*(.+?)\s*$", "Unknown")
    section = _extract_field(text, r"^\s*Section\s*:\s*(.+?)\s*$", "Unknown")
    url = _extract_field(text, r"^\s*URL\s*:\s*(.+?)\s*$", "")
    metadata = {
        "act_name": act,
        "chapter": chapter,
        "section": section,
        "url": url,
        "source_file": source_file.name,
    }
    return text, metadata


def _extract_field(text: str, pattern: str, default: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else default


def _load_json_chunk(file_path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        text = str(payload.get("text") or payload.get("content") or "").strip()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        normalized = {
            "act_name": str(metadata.get("act_name") or payload.get("act_name") or file_path.stem),
            "chapter": str(metadata.get("chapter") or payload.get("chapter") or "Unknown"),
            "section": str(metadata.get("section") or payload.get("section") or "Unknown"),
            "url": str(metadata.get("url") or payload.get("url") or ""),
            "source_file": file_path.name,
        }
        return text, normalized
    raise ValueError(f"Unsupported JSON chunk shape in {file_path}")


def load_documents(chunks_dir: Path) -> list[Document]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    if not chunks_dir.is_dir():
        raise ValueError(f"Chunks path must be a directory: {chunks_dir}")

    documents: list[Document] = []
    for file_path in sorted(chunks_dir.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".markdown", ".json"}:
            continue

        if suffix == ".json":
            text, metadata = _load_json_chunk(file_path)
        else:
            text = file_path.read_text(encoding="utf-8").strip()
            text, metadata = _extract_markdown_metadata(text, file_path)

        if not text:
            continue
        documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        raise ValueError(f"No supported chunk files found in {chunks_dir}")
    return documents


def _build_embeddings(provider: str, openai_model: str, hf_model: str):
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings.")
        return OpenAIEmbeddings(model=openai_model)
    return HuggingFaceEmbeddings(model_name=hf_model)


def _stable_id(page_content: str, metadata: dict[str, Any]) -> str:
    raw = f"{metadata.get('source_file','')}|{metadata.get('act_name','')}|{metadata.get('section','')}|{page_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    vector_dir = Path(args.vector_db_dir)
    vector_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(chunks_dir)
    embeddings = _build_embeddings(args.embedding_provider, args.openai_model, args.hf_model)

    db = Chroma(
        collection_name=args.collection_name,
        embedding_function=embeddings,
        persist_directory=str(vector_dir),
    )

    ids = [_stable_id(doc.page_content, doc.metadata) for doc in docs]
    db.add_documents(documents=docs, ids=ids)

    unique_acts = sorted({doc.metadata.get("act_name", "Unknown") for doc in docs})
    print(f"Loaded chunks: {len(docs)}")
    print(f"Collection: {args.collection_name}")
    print(f"Vector DB dir: {vector_dir}")
    print("Acts loaded:")
    for act in unique_acts:
        print(f"- {act}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
