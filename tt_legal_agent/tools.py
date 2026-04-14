"""Agentic helper tools for the T&T legal assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from urllib.parse import parse_qs, unquote, urlparse

from docx import Document
from pypdf import PdfReader
import requests


@dataclass(frozen=True)
class GazetteResult:
    title: str
    snippet: str
    url: str


def web_search_gazette(query: str, max_results: int = 5) -> list[GazetteResult]:
    """Simple web search over the Gazette website using DuckDuckGo HTML results.

    This keeps the implementation lightweight and avoids paid search APIs.
    """
    search_query = f"site:gazette.gov.tt {query}"
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": search_query},
        timeout=20,
    )
    response.raise_for_status()
    html = response.text

    # Small parser by regex for the DDG HTML endpoint.
    matches = re.findall(
        r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)">(?P<title>.*?)</a>.*?'
        r'<a class="result__snippet" href=".*?">(?P<snippet>.*?)</a>',
        html,
        flags=re.DOTALL,
    )

    results: list[GazetteResult] = []
    for url, title, snippet in matches[:max_results]:
        clean_title = re.sub("<.*?>", "", title)
        clean_snippet = re.sub("<.*?>", "", snippet)
        clean_url = _normalize_duckduckgo_result_url(url)
        results.append(
            GazetteResult(
                title=clean_title.strip(),
                snippet=clean_snippet.strip(),
                url=clean_url,
            )
        )
    return results


def _normalize_duckduckgo_result_url(raw_url: str) -> str:
    """Convert DuckDuckGo redirect URLs to direct target URLs."""
    parsed = urlparse(raw_url)
    if parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [])
        if uddg:
            return unquote(uddg[0])
    return raw_url


def _read_document_text(target: Path) -> str:
    suffix = target.suffix.lower()
    if suffix in {".txt", ".md"}:
        return target.read_text(encoding="utf-8")
    if suffix == ".pdf":
        reader = PdfReader(str(target))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = Document(str(target))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    raise ValueError("Supported formats are .txt, .md, .pdf, .docx")


def summarize_document(path: str, max_chars: int = 3500) -> dict[str, Any]:
    """Return a lightweight summary object for a local document.

    Reads plain text or markdown files and returns an extractive style summary.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    text = _read_document_text(target)
    compact = " ".join(text.split())
    preview = compact[:max_chars]

    # Heuristic highlights: sentence-like clauses with legal keywords.
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    key_terms = ("statement", "caution", "justice of the peace", "section", "chapter", "procedure")
    highlights = [
        s for s in sentences if any(term in s.lower() for term in key_terms)
    ][:6]

    return {
        "path": str(target),
        "char_count": len(text),
        "preview": preview,
        "highlights": highlights,
    }
