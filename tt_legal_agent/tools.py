"""Agentic helper tools for the T&T legal assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

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
        results.append(
            GazetteResult(
                title=clean_title.strip(),
                snippet=clean_snippet.strip(),
                url=url,
            )
        )
    return results


def summarize_document(path: str, max_chars: int = 3500) -> dict[str, Any]:
    """Return a lightweight summary object for a local document.

    Reads plain text or markdown files and returns an extractive style summary.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if target.suffix.lower() not in {".txt", ".md"}:
        raise ValueError("Only .txt and .md are supported in this scaffold.")

    text = target.read_text(encoding="utf-8")
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
