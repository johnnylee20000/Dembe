"""Utilities for loading markdown chunks and extracting legal metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


@dataclass(frozen=True)
class LegalChunk:
    """Normalized chunk record to index in a vector store."""

    chunk_id: str
    text: str
    act_name: str
    chapter: str
    section: str
    url: str
    source_file: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "act_name": self.act_name,
            "chapter": self.chapter,
            "section": self.section,
            "url": self.url,
            # Backward compatibility for callers expecting source_url.
            "source_url": self.url,
            "source_file": self.source_file,
        }


ACT_RE = re.compile(r"^\s*Act(?:\s+Name)?\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
CHAPTER_RE = re.compile(r"^\s*Chapter\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
SECTION_RE = re.compile(r"^\s*Section\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
URL_RE = re.compile(r"^\s*URL\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _extract(pattern: re.Pattern[str], text: str, default: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else default


def parse_markdown_chunk(path: Path) -> LegalChunk:
    """Parse one markdown chunk into a LegalChunk object."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Chunk file is empty: {path}")

    title = _extract(TITLE_RE, text, path.stem.replace("_", " ").title())
    act_name = _extract(ACT_RE, text, title)
    chapter = _extract(CHAPTER_RE, text, "Unknown")
    section = _extract(SECTION_RE, text, "Unknown")
    url = _extract(URL_RE, text, "")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    chunk_id = f"{path.stem}-{digest}"

    return LegalChunk(
        chunk_id=chunk_id,
        text=text,
        act_name=act_name,
        chapter=chapter,
        section=section,
        url=url,
        source_file=path.name,
    )


def load_markdown_chunks(chunks_dir: str | Path) -> list[LegalChunk]:
    """Load and parse all markdown chunk files in a directory tree."""
    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")
    if not chunks_dir.is_dir():
        raise ValueError(f"Chunks path must be a directory: {chunks_dir}")

    chunks: list[LegalChunk] = []
    for file_path in sorted(chunks_dir.rglob("*.md")):
        chunks.append(parse_markdown_chunk(file_path))
    if not chunks:
        raise ValueError(f"No markdown chunks found in: {chunks_dir}")
    return chunks


def split_markdown_into_segments(
    source_file: Path,
    output_dir: Path,
    max_chars: int = 2200,
    overlap: int = 200,
) -> list[Path]:
    """Split a long markdown document into smaller markdown chunks.

    This helper is optional and useful when your source data is not already chunked.
    It preserves metadata headers if present and writes markdown chunk files.
    """
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")

    text = source_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Source markdown file is empty: {source_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    head = []
    for line in text.splitlines():
        if re.match(r"^\s*(Act(?:\s+Name)?|Chapter|Section|URL)\s*:", line, re.IGNORECASE):
            head.append(line)
    header = "\n".join(head).strip()

    body = text
    chunks: list[Path] = []
    start = 0
    idx = 1
    while start < len(body):
        end = min(len(body), start + max_chars)
        piece = body[start:end].strip()
        if header:
            payload = f"{header}\n\n{piece}\n"
        else:
            payload = f"{piece}\n"
        target = output_dir / f"{source_file.stem}_chunk_{idx:03d}.md"
        target.write_text(payload, encoding="utf-8")
        chunks.append(target)
        idx += 1
        if end >= len(body):
            break
        start = end - overlap
    return chunks
