"""Index markdown legal chunks into ChromaDB."""

from __future__ import annotations

import argparse

from tt_legal_agent.chunker import load_markdown_chunks
from tt_legal_agent.vector_store import TTVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index markdown chunks to ChromaDB.")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--persist-dir", default=".chroma_tt_law")
    parser.add_argument("--collection", default="tt_law")
    parser.add_argument("--embedding-backend", choices=("local", "openai"), default="local")
    parser.add_argument("--openai-api-key")
    parser.add_argument(
        "--local-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--openai-embedding-model", default="text-embedding-3-small")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_markdown_chunks(args.chunks_dir)

    store = TTVectorStore(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        embedding_backend=args.embedding_backend,
        local_embedding_model=args.local_embedding_model,
        openai_embedding_model=args.openai_embedding_model,
        openai_api_key=args.openai_api_key,
    )
    count = store.upsert_chunks(chunks)
    print(
        f"Indexed {count} chunk(s) into collection '{args.collection}' at {args.persist_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
