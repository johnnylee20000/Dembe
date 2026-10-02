#!/usr/bin/env python3
"""
Structure-first chunking for Trinidad and Tobago legal PDFs.

Highlights:
- Uses layout-aware PDF extraction (pdfplumber, with PyMuPDF fallback).
- Detects legal structure headings (Section / Schedule).
- Keeps sections atomic when <= 1,000 tokens.
- Uses recursive character splitting for oversized sections, with preferred
  boundaries at double newlines and section-style breaks.
- Adds chunk headers in the format: "[Act Name] - Section [Number]".
- Applies 15-20% overlap (configurable; default 18%).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

SECTION_HEADING_RE = re.compile(
    r"^\s*(?:section|sec\.?|s\.)\s+(\d+[A-Za-z0-9().-]*)\b",
    flags=re.IGNORECASE,
)
SCHEDULE_HEADING_RE = re.compile(
    r"^\s*(?:(first|second|third|fourth|fifth)\s+)?schedule\b",
    flags=re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """
    Fast token approximation suitable for chunk sizing without heavy tokenizers.
    """
    words = re.findall(r"\S+", text)
    # Legal prose tends to average around 1.2-1.4 words/token depending on style.
    # Use a conservative midpoint to avoid oversized chunks.
    return max(1, int(len(words) / 0.75))


def _word_overlap_text(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = re.findall(r"\S+", text)
    if not words:
        return ""
    approx_overlap_words = max(1, int(overlap_tokens * 0.75))
    tail = words[-approx_overlap_words:]
    return " ".join(tail)


def recursive_character_split(
    text: str,
    max_tokens: int,
    separators: tuple[str, ...] = ("\n\n", "\n", "; ", ". ", " "),
) -> list[str]:
    """
    Recursively split text by progressively finer separators.
    """
    normalized = text.strip()
    if not normalized:
        return []
    if estimate_tokens(normalized) <= max_tokens:
        return [normalized]
    if not separators:
        words = re.findall(r"\S+", normalized)
        if not words:
            return []
        approx_words_per_chunk = max(1, int(max_tokens * 0.75))
        chunks: list[str] = []
        for i in range(0, len(words), approx_words_per_chunk):
            chunks.append(" ".join(words[i : i + approx_words_per_chunk]))
        return chunks

    separator = separators[0]
    parts = normalized.split(separator)
    if len(parts) == 1:
        return recursive_character_split(normalized, max_tokens, separators[1:])

    chunks: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}{separator}{part}" if buffer else part
        if estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer.strip())
        if estimate_tokens(part) <= max_tokens:
            buffer = part
        else:
            chunks.extend(recursive_character_split(part, max_tokens, separators[1:]))
            buffer = ""

    if buffer:
        chunks.append(buffer.strip())
    return [c for c in chunks if c]


@dataclass
class SectionBlock:
    section_number: str
    heading: str
    text: str


@dataclass
class LegalChunk:
    act_name: str
    section_number: str
    chunk_index: int
    total_chunks_in_section: int
    token_estimate: int
    header: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.header}\n\n{self.text}"


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text while preserving line breaks and rough page structure.
    """
    try:
        import pdfplumber  # type: ignore

        pages: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except Exception:
        # Fallback parser
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(pages)


def detect_sections(raw_text: str) -> list[SectionBlock]:
    """
    Split legal text into section/schedule blocks using heading detection.
    """
    lines = raw_text.splitlines()
    blocks: list[SectionBlock] = []

    current_number = "Preamble"
    current_heading = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append(
                SectionBlock(
                    section_number=current_number,
                    heading=current_heading,
                    text=text,
                )
            )

    for line in lines:
        stripped = line.strip()
        sec_match = SECTION_HEADING_RE.match(stripped)
        sch_match = SCHEDULE_HEADING_RE.match(stripped)

        if sec_match:
            flush()
            current_number = sec_match.group(1)
            current_heading = stripped
            current_lines = [line]
            continue

        if sch_match:
            flush()
            qualifier = sch_match.group(1) or ""
            current_number = f"{qualifier.title()} Schedule".strip()
            current_heading = stripped
            current_lines = [line]
            continue

        current_lines.append(line)

    flush()
    return blocks


def chunk_section(
    section: SectionBlock,
    act_name: str,
    *,
    target_chunk_tokens: int = 1000,
    max_chunk_tokens: int = 1200,
    overlap_ratio: float = 0.18,
    atomic_threshold_tokens: int = 1000,
) -> list[LegalChunk]:
    """
    Chunk one legal section while preserving small sections as atomic units.
    """
    section_tokens = estimate_tokens(section.text)
    header = f"[{act_name}] - Section {section.section_number}"

    if section_tokens <= atomic_threshold_tokens:
        return [
            LegalChunk(
                act_name=act_name,
                section_number=section.section_number,
                chunk_index=1,
                total_chunks_in_section=1,
                token_estimate=section_tokens,
                header=header,
                text=section.text.strip(),
            )
        ]

    split_parts = recursive_character_split(section.text, max_chunk_tokens)
    overlap_tokens = max(1, int(target_chunk_tokens * overlap_ratio))

    stitched: list[str] = []
    prev_part = ""
    for idx, part in enumerate(split_parts):
        part_clean = part.strip()
        if not part_clean:
            continue
        if idx == 0:
            stitched.append(part_clean)
            prev_part = part_clean
            continue

        overlap_text = _word_overlap_text(prev_part, overlap_tokens)
        merged = f"{overlap_text}\n\n{part_clean}" if overlap_text else part_clean

        # Keep final merged chunk bounded.
        if estimate_tokens(merged) > max_chunk_tokens:
            shrunk = recursive_character_split(merged, max_chunk_tokens)
            stitched.extend(shrunk)
            prev_part = shrunk[-1] if shrunk else part_clean
        else:
            stitched.append(merged)
            prev_part = part_clean

    chunks: list[LegalChunk] = []
    total = len(stitched)
    for i, text in enumerate(stitched, start=1):
        chunks.append(
            LegalChunk(
                act_name=act_name,
                section_number=section.section_number,
                chunk_index=i,
                total_chunks_in_section=total,
                token_estimate=estimate_tokens(text),
                header=header,
                text=text,
            )
        )
    return chunks


def chunk_legal_pdf(
    pdf_path: Path,
    act_name: str,
    *,
    target_chunk_tokens: int = 1000,
    max_chunk_tokens: int = 1200,
    overlap_ratio: float = 0.18,
    atomic_threshold_tokens: int = 1000,
) -> list[LegalChunk]:
    """
    End-to-end legal PDF chunking entrypoint.
    """
    raw_text = extract_pdf_text(pdf_path)
    sections = detect_sections(raw_text)
    if not sections:
        sections = [SectionBlock(section_number="Unknown", heading="", text=raw_text)]

    all_chunks: list[LegalChunk] = []
    for section in sections:
        all_chunks.extend(
            chunk_section(
                section,
                act_name,
                target_chunk_tokens=target_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
                overlap_ratio=overlap_ratio,
                atomic_threshold_tokens=atomic_threshold_tokens,
            )
        )
    return all_chunks


def save_chunks_jsonl(chunks: Iterable[LegalChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = asdict(chunk)
            payload["full_text"] = chunk.full_text
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk Trinidad and Tobago legal PDFs with section-aware splitting."
    )
    parser.add_argument("pdf_path", help="Path to legal PDF.")
    parser.add_argument(
        "--act-name",
        default=None,
        help="Act name used in chunk headers. Defaults to PDF filename stem.",
    )
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output JSONL file path.")
    parser.add_argument(
        "--target-chunk-tokens",
        type=int,
        default=1000,
        help="Target token size for chunking.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=1200,
        help="Hard upper bound for chunk token size.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.18,
        help="Overlap ratio between adjacent chunks (e.g., 0.18 for 18%%).",
    )
    parser.add_argument(
        "--atomic-threshold-tokens",
        type=int,
        default=1000,
        help="Keep a section atomic when token estimate is at/below this value.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    act_name = args.act_name or pdf_path.stem
    chunks = chunk_legal_pdf(
        pdf_path=pdf_path,
        act_name=act_name,
        target_chunk_tokens=args.target_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_ratio=args.overlap_ratio,
        atomic_threshold_tokens=args.atomic_threshold_tokens,
    )
    output_path = Path(args.output)
    save_chunks_jsonl(chunks, output_path)
    print(f"wrote {len(chunks)} chunks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
