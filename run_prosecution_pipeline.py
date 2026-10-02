#!/usr/bin/env python3
"""
One-command runner for prosecution assistant pipeline.

Steps:
1) Bootstrap prosecution corpus (scrape -> prioritize -> chunk -> vector DB)
2) Validate prosecution stack health
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap_prosecution_corpus import DEFAULT_PRIORITY_KEYWORDS, bootstrap_corpus
from validate_prosecution_stack import validate_stack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full prosecution pipeline bootstrap + validation.")
    parser.add_argument(
        "--output-root",
        default="/workspace/data/prosecution_bootstrap",
        help="Bootstrap output root directory.",
    )
    parser.add_argument(
        "--vector-db-path",
        default="/workspace/vector_db",
        help="Vector DB output path.",
    )
    parser.add_argument(
        "--scrape-limit",
        type=int,
        default=120,
        help="Maximum number of acts to scrape.",
    )
    parser.add_argument(
        "--min-priority-docs",
        type=int,
        default=20,
        help="Minimum prioritized docs after scoring.",
    )
    parser.add_argument(
        "--max-priority-docs",
        type=int,
        default=60,
        help="Maximum prioritized docs after scoring.",
    )
    parser.add_argument(
        "--priority-keywords",
        default=",".join(DEFAULT_PRIORITY_KEYWORDS),
        help="Comma-separated keyword list for priority scoring.",
    )
    parser.add_argument(
        "--strict-authority",
        action="store_true",
        help="Enable stricter authority-check gating during validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keywords = tuple(k.strip().lower() for k in args.priority_keywords.split(",") if k.strip())

    bootstrap_summary = bootstrap_corpus(
        output_root=Path(args.output_root),
        scrape_limit=args.scrape_limit,
        min_priority_docs=args.min_priority_docs,
        max_priority_docs=args.max_priority_docs,
        priority_keywords=keywords or DEFAULT_PRIORITY_KEYWORDS,
        vector_db_path=Path(args.vector_db_path),
    )

    validation_summary = validate_stack(
        vector_db_path=args.vector_db_path,
        strict_authority=args.strict_authority,
    )

    output = {
        "bootstrap": bootstrap_summary,
        "validation": validation_summary,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if validation_summary.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())

