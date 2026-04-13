"""Ask the T&T legal RAG assistant from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tt_legal_agent.rag_pipeline import TTRAGPipeline
from tt_legal_agent.vector_store import TTVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a Trinidad and Tobago legal assistant question."
    )
    parser.add_argument(
        "--persist-dir",
        default=".chroma_tt_law",
        help="Path to ChromaDB persistent directory.",
    )
    parser.add_argument(
        "--collection",
        default="tt_law",
        help="Chroma collection name.",
    )
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of relevant chunks to retrieve.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=("local", "openai"),
        default="local",
        help="Embedding backend used for retrieval query.",
    )
    parser.add_argument(
        "--local-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model when using local embeddings.",
    )
    parser.add_argument(
        "--openai-embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when using --embedding-backend openai.",
    )
    parser.add_argument(
        "--answer-model",
        default="gpt-4o-mini",
        help="LLM used for final synthesis when OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = TTVectorStore(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        embedding_backend=args.embedding_backend,
        local_embedding_model=args.local_embedding_model,
        openai_embedding_model=args.openai_embedding_model,
    )
    rag = TTRAGPipeline(vector_store=store, model=args.answer_model)
    result = rag.answer_question(args.question, top_k=args.k)
    print(result.answer)
    print("\nCitations:")
    for item in result.citations:
        print(
            "- "
            f"{item['act_name']} | Chapter {item['chapter']} | Section {item['section']} | {item['url']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
