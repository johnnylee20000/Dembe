#!/usr/bin/env python3
"""
TTPS RetrievalQA entrypoint.

Creates a LangChain RetrievalQA chain that:
- loads local ChromaDB from /vector_db (with local workspace fallback)
- uses GPT-4o (or local model) for generation
- uses a "stuff" chain
- returns source documents for citation
"""

from __future__ import annotations

import argparse
import json

from assistant_core import build_retrieval_qa_chain, format_source_documents


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
        "sources": format_source_documents(result.get("source_documents", [])),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
