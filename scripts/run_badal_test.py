"""Validation script for the Badal benchmark question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tt_legal_agent.rag_pipeline import TTRAGPipeline
from tt_legal_agent.vector_store import TTVectorStore


BADAL_QUESTION = (
    "If a suspect makes an utterance during an interview in Santa Cruz, "
    "what is the required police procedure before continuing?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Badal benchmark test.")
    parser.add_argument(
        "--chroma-dir",
        default=".chroma_tt_law",
        help="Directory for Chroma persistence (default: .chroma_tt_law)",
    )
    parser.add_argument(
        "--collection",
        default="tt_law",
        help="Chroma collection name (default: tt_law)",
    )
    parser.add_argument(
        "--output",
        default="badal_test_result.json",
        help="Path to write benchmark result JSON (default: badal_test_result.json)",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("local", "openai"),
        default="local",
        help="Embedding backend used for retrieval query embedding.",
    )
    parser.add_argument(
        "--local-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model for local embeddings.",
    )
    parser.add_argument(
        "--openai-embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when backend=openai.",
    )
    return parser.parse_args()


def evaluate_text(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {
        "mentions_statement_under_caution": "statement under caution" in lower,
        "mentions_justice_of_the_peace": "justice of the peace" in lower,
        "mentions_authentication": "authenticate" in lower
        or "authentication" in lower,
    }


def main() -> int:
    args = parse_args()
    store = TTVectorStore(
        persist_directory=args.chroma_dir,
        collection_name=args.collection,
        embedding_backend=args.embedding_backend,
        local_embedding_model=args.local_embedding_model,
        openai_embedding_model=args.openai_embedding_model,
    )
    pipeline = TTRAGPipeline(vector_store=store)
    result = pipeline.answer_question(BADAL_QUESTION, top_k=5)
    checks = evaluate_text(result.answer)

    payload = {
        "question": BADAL_QUESTION,
        "answer": result.answer,
        "citations": result.citations,
        "checks": checks,
        "pass": all(checks.values()),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

