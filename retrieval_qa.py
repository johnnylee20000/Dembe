#!/usr/bin/env python3
"""
TTPS RetrievalQA / retrieval-only entrypoint.

Modes:
- Default: full RetrievalQA (LLM answer + sources)
- --retrieval-only: return top source chunks only (no LLM required)
"""

from __future__ import annotations

import argparse
import json

from assistant_core import (
    build_retrieval_qa_chain,
    retrieve_documents,
    format_source_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RetrievalQA against local ChromaDB.")
    parser.add_argument(
        "--vector-db-path",
        default="/vector_db",
        help="Path to persisted Chroma vector database.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Officer question to answer from legal context.",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        choices=["openai", "ollama"],
        help="LLM provider override.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name override (e.g., gpt-4o, llama3.1).",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        choices=["openai", "huggingface"],
        help="Embedding provider override.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of retrieved chunks.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM answer generation and return only top source chunks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.retrieval_only:
        docs = retrieve_documents(
            args.question,
            vector_db_path=args.vector_db_path,
            embedding_provider=args.embedding_provider,
            k=args.k,
        )
        payload = {
            "question": args.question,
            "answer": "",
            "mode": "retrieval_only",
            "sources": format_source_documents(docs),
        }
    else:
        chain = build_retrieval_qa_chain(
            vector_db_path=args.vector_db_path,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            embedding_provider=args.embedding_provider,
            k=args.k,
        )
        result = chain.invoke({"query": args.question})
        payload = {
            "question": args.question,
            "answer": result.get("result", ""),
            "mode": "retrieval_qa",
            "sources": format_source_documents(result.get("source_documents", [])),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
