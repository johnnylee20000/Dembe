#!/usr/bin/env python3
"""
Build a local Chroma vector DB from chunk JSONL files.

Expected chunk JSONL format (from legal_chunker.py):
- text
- full_text
- header
- act_name
- section_number
- ... other metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from assistant_core import build_embeddings


def load_chunks_from_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            page_content = payload.get("full_text") or payload.get("text") or ""
            metadata = {
                "source_file": str(path),
                "line_number": idx,
                "act_name": payload.get("act_name", ""),
                "section_number": payload.get("section_number", ""),
                "header": payload.get("header", ""),
                "chunk_index": payload.get("chunk_index", ""),
                "token_estimate": payload.get("token_estimate", ""),
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def build_vector_db(
    chunks_glob: str,
    vector_db_path: str = "/workspace/vector_db",
    embedding_provider: str | None = None,
) -> tuple[int, str]:
    chunk_paths = sorted(Path("/workspace").glob(chunks_glob))
    if not chunk_paths:
        raise FileNotFoundError(
            f"No chunk files matched glob '{chunks_glob}' under /workspace."
        )

    all_docs: list[Document] = []
    for chunk_path in chunk_paths:
        all_docs.extend(load_chunks_from_jsonl(chunk_path))

    if not all_docs:
        raise RuntimeError("No chunk documents were loaded from JSONL files.")

    persist_path = Path(vector_db_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = build_embeddings(embedding_provider=embedding_provider)
    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    db.persist()
    return len(all_docs), str(persist_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Chroma vector DB from chunk JSONL files.")
    parser.add_argument(
        "--chunks-glob",
        default="data/*chunks*.jsonl",
        help="Glob under /workspace to discover chunk JSONL files.",
    )
    parser.add_argument(
        "--vector-db-path",
        default="/workspace/vector_db",
        help="Output directory for persisted Chroma DB.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Embedding provider override (openai or huggingface).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    count, path = build_vector_db(
        chunks_glob=args.chunks_glob,
        vector_db_path=args.vector_db_path,
        embedding_provider=args.embedding_provider,
    )
    print(f"indexed {count} documents into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
