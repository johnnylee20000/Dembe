#!/usr/bin/env python3
"""
Bootstrap prosecution-focused corpus and vector DB.

Pipeline:
1) Scrape/download a configurable number of Acts from laws.gov.tt.
2) Select priority Acts/procedural materials using keyword matching.
3) Chunk selected PDFs with legal structure-aware chunking.
4) Build a Chroma vector DB from resulting chunks.

This script is designed for autonomous "do the necessary" setup runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from pathlib import Path

from build_vector_db import build_vector_db
from laws_scraper import LawsTTScraper
from legal_chunker import chunk_legal_pdf, save_chunks_jsonl


DEFAULT_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "firearm",
    "firearms",
    "summary courts",
    "criminal",
    "evidence",
    "police",
    "drug",
    "dangerous drugs",
    "indictable",
    "procedure",
    "judge",
    "standing order",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _matches_priority(act_name: str, chapter: str, keywords: tuple[str, ...]) -> bool:
    haystack = f"{_normalize(act_name)} {_normalize(chapter)}"
    return any(keyword in haystack for keyword in keywords)


def _score_priority(act_name: str, chapter: str, keywords: tuple[str, ...]) -> int:
    haystack = f"{_normalize(act_name)} {_normalize(chapter)}"
    return sum(1 for keyword in keywords if keyword in haystack)


def _read_metadata_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(payloads: list[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(payloads)


def bootstrap_corpus(
    output_root: Path,
    *,
    scrape_limit: int = 120,
    min_priority_docs: int = 12,
    max_priority_docs: int = 50,
    priority_keywords: tuple[str, ...] = DEFAULT_PRIORITY_KEYWORDS,
    vector_db_path: Path = Path("/workspace/vector_db"),
) -> dict[str, int | str]:
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw_laws"
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    scraper = LawsTTScraper(output_dir=raw_dir, delay_seconds=0.6, timeout_seconds=45, max_retries=3)
    logging.info("Scraping laws corpus (limit=%s)...", scrape_limit)
    records = scraper.scrape(limit=scrape_limit)
    metadata_path = raw_dir / "metadata.csv"
    scraper.write_metadata(records, metadata_path)

    metadata_rows = _read_metadata_csv(metadata_path)
    downloaded_rows = [row for row in metadata_rows if row.get("download_status") == "downloaded"]
    ranked_rows = sorted(
        downloaded_rows,
        key=lambda row: (
            _score_priority(
                row.get("act_name", ""),
                row.get("chapter_number", ""),
                priority_keywords,
            ),
            row.get("act_name", ""),
        ),
        reverse=True,
    )
    selected_rows = [row for row in ranked_rows if _score_priority(row.get("act_name", ""), row.get("chapter_number", ""), priority_keywords) > 0]

    if len(selected_rows) < min_priority_docs:
        selected_rows = ranked_rows[: max(min_priority_docs, len(selected_rows))]
    else:
        selected_rows = selected_rows[: max(min_priority_docs, len(selected_rows))]
    selected_rows = selected_rows[: max_priority_docs]

    chunk_payloads: list[dict] = []
    chunked_docs = 0
    for row in selected_rows:
        local_file = row.get("local_file", "")
        if not local_file:
            continue
        pdf_path = Path(local_file)
        if not pdf_path.exists():
            continue
        act_name = row.get("act_name", "") or pdf_path.stem
        try:
            chunks = chunk_legal_pdf(
                pdf_path=pdf_path,
                act_name=act_name,
                target_chunk_tokens=1000,
                max_chunk_tokens=1200,
                overlap_ratio=0.18,
                atomic_threshold_tokens=1000,
            )
            for chunk in chunks:
                payload = {
                    "act_name": chunk.act_name,
                    "section_number": chunk.section_number,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks_in_section": chunk.total_chunks_in_section,
                    "token_estimate": chunk.token_estimate,
                    "header": chunk.header,
                    "text": chunk.text,
                    "full_text": chunk.full_text,
                    "chapter_number": row.get("chapter_number", ""),
                    "source_page_url": row.get("source_page_url", ""),
                    "selected_file_url": row.get("selected_file_url", ""),
                    "selected_file_category": row.get("selected_file_category", ""),
                }
                chunk_payloads.append(payload)
            chunked_docs += 1
        except Exception as exc:  # pragma: no cover - operational path
            logging.warning("Chunking failed for %s: %s", pdf_path, exc)

    chunks_file = chunks_dir / "prosecution_priority_chunks.jsonl"
    count = _write_jsonl(chunk_payloads, chunks_file)
    if count == 0:
        raise RuntimeError("No chunks produced during bootstrap; cannot build vector DB.")

    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)

    indexed, _ = build_vector_db(
        chunks_glob=str(chunks_file.relative_to(Path("/workspace"))),
        vector_db_path=str(vector_db_path),
        embedding_provider=None,
    )

    return {
        "scraped_records": len(records),
        "selected_priority_docs": len(selected_rows),
        "chunked_docs": chunked_docs,
        "chunk_count": count,
        "indexed_docs": indexed,
        "metadata_csv": str(metadata_path),
        "chunks_file": str(chunks_file),
        "vector_db_path": str(vector_db_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap prosecution-focused legal corpus and vector DB.")
    parser.add_argument(
        "--output-root",
        default="/workspace/data/prosecution_bootstrap",
        help="Root directory for bootstrap artifacts.",
    )
    parser.add_argument(
        "--scrape-limit",
        type=int,
        default=120,
        help="Max number of acts to scrape in this run.",
    )
    parser.add_argument(
        "--min-priority-docs",
        type=int,
        default=12,
        help="Minimum number of documents to include after prioritization.",
    )
    parser.add_argument(
        "--max-priority-docs",
        type=int,
        default=40,
        help="Maximum number of prioritized documents to include.",
    )
    parser.add_argument(
        "--priority-keywords",
        default=",".join(DEFAULT_PRIORITY_KEYWORDS),
        help="Comma-separated keyword list for prosecution/procedure prioritization.",
    )
    parser.add_argument(
        "--vector-db-path",
        default="/workspace/vector_db",
        help="Path for rebuilt vector database.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    keywords = tuple(_normalize(x) for x in args.priority_keywords.split(",") if _normalize(x))
    summary = bootstrap_corpus(
        output_root=Path(args.output_root),
        scrape_limit=args.scrape_limit,
        min_priority_docs=args.min_priority_docs,
        max_priority_docs=args.max_priority_docs,
        priority_keywords=keywords or DEFAULT_PRIORITY_KEYWORDS,
        vector_db_path=Path(args.vector_db_path),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
