#!/usr/bin/env python3
"""
Scrape and chunk Trinidad & Tobago law/policing HTML pages.

Pipeline:
1) Fetch HTML from URL.
2) Extract main content container and remove site chrome.
3) Convert cleaned HTML to Markdown.
4) Split by Markdown headers.
5) Further split oversized chunks with recursive splitter.
6) Attach metadata for each final chunk.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from html2text import HTML2Text
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


MAIN_CONTENT_SELECTORS = [
    ".entry-content",
    "article",
    "#main",
    "main",
]

CHROME_SELECTORS = [
    "nav",
    "footer",
    "aside",
    ".sidebar",
    "#sidebar",
    ".site-navigation",
    ".navigation",
    ".menu",
    ".breadcrumbs",
    ".breadcrumb",
    ".related",
    ".related-posts",
    ".widget",
    ".comments",
    ".comment",
]

BREADCRUMB_SELECTORS = [
    ".breadcrumb",
    ".breadcrumbs",
    "[aria-label='breadcrumb']",
    "nav.breadcrumb",
    ".yoast-breadcrumb",
]

LAST_UPDATED_REGEXES = [
    re.compile(r"(last\s+updated\s*[:\-]\s*)(.+)", re.IGNORECASE),
    re.compile(r"(revised\s+date\s*[:\-]\s*)(.+)", re.IGNORECASE),
    re.compile(r"(date\s+revised\s*[:\-]\s*)(.+)", re.IGNORECASE),
]

LAW_REFERENCE_REGEX = re.compile(
    r"\bChapter\s+\d+(?::\d+)?\b",
    re.IGNORECASE,
)


@dataclass
class ChunkRecord:
    text: str
    metadata: Dict[str, Any]


def fetch_html(url: str, timeout: int = 30) -> str:
    """Fetch HTML from the source URL."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _remove_non_content_nodes(container: Tag) -> None:
    """Remove obvious chrome/non-content nodes from selected container."""
    for selector in CHROME_SELECTORS:
        for node in container.select(selector):
            node.decompose()

    # Remove script/style/noscript even if nested in content.
    for node in container.select("script, style, noscript"):
        node.decompose()


def select_main_container(soup: BeautifulSoup) -> Tag:
    """Pick main content container with fallback to body/document root."""
    for selector in MAIN_CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate:
            return candidate

    if soup.body:
        return soup.body

    # Extremely defensive fallback.
    return soup


def extract_breadcrumb(soup: BeautifulSoup) -> Optional[str]:
    """Extract breadcrumb navigation path when present."""
    for selector in BREADCRUMB_SELECTORS:
        node = soup.select_one(selector)
        if node:
            # Join visible breadcrumb items with a clear separator.
            items = [t.strip() for t in node.stripped_strings if t.strip()]
            if items:
                return " > ".join(items)

    # Heuristic fallback: any node with class/id containing breadcrumb.
    fallback = soup.select_one("[class*='breadcrumb'], [id*='breadcrumb']")
    if fallback:
        items = [t.strip() for t in fallback.stripped_strings if t.strip()]
        if items:
            return " > ".join(items)

    return None


def extract_last_updated(soup: BeautifulSoup) -> Optional[str]:
    """Extract 'Last Updated' / 'Revised Date' text from page."""
    all_text = "\n".join(t.strip() for t in soup.stripped_strings if t.strip())
    for pattern in LAST_UPDATED_REGEXES:
        match = pattern.search(all_text)
        if match:
            # Keep just the captured date/value portion.
            return match.group(2).split("\n")[0].strip()
    return None


def html_to_markdown(content_html: str) -> str:
    """Convert cleaned HTML to Markdown."""
    converter = HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0  # avoid hard-wrap
    converter.single_line_break = False
    return converter.handle(content_html).strip()


def estimate_token_count(text: str) -> int:
    """
    Lightweight token estimate.
    Rough heuristic for English prose: ~4 chars/token.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def split_markdown_structurally(markdown_text: str) -> List[Dict[str, Any]]:
    """Split markdown by hierarchical headers."""
    headers_to_split_on = [
        ("#", "Act/Title"),
        ("##", "Part/Chapter"),
        ("###", "Section"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    return splitter.split_text(markdown_text)


def split_oversized_chunk(text: str, max_tokens: int = 1000) -> List[str]:
    """Recursively split a chunk if it exceeds token budget."""
    if estimate_token_count(text) <= max_tokens:
        return [text]

    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=max_tokens * 4,  # convert token budget to char budget
        chunk_overlap=150 * 4,  # 150 token overlap
        length_function=len,
        is_separator_regex=False,
    )
    parts = recursive_splitter.split_text(text)
    return [part.strip() for part in parts if part.strip()]


def build_chunks(
    url: str,
    markdown_text: str,
    breadcrumb: Optional[str],
    last_updated: Optional[str],
) -> List[ChunkRecord]:
    """Build final chunk list with all metadata fields."""
    structural_docs = split_markdown_structurally(markdown_text)
    chunks: List[ChunkRecord] = []

    for doc in structural_docs:
        base_text = doc.page_content.strip()
        if not base_text:
            continue

        law_ref_match = LAW_REFERENCE_REGEX.search(base_text)
        law_reference = law_ref_match.group(0) if law_ref_match else None

        metadata: Dict[str, Any] = {
            "source_url": url,
            "breadcrumb": breadcrumb,
            "last_updated": last_updated,
            "law_reference": law_reference,
        }
        # Include structural header metadata from LangChain splitter.
        metadata.update(doc.metadata)

        secondary_parts = split_oversized_chunk(base_text, max_tokens=1000)
        for idx, piece in enumerate(secondary_parts):
            piece_meta = dict(metadata)
            piece_meta["subchunk_index"] = idx
            piece_meta["estimated_tokens"] = estimate_token_count(piece)
            chunks.append(ChunkRecord(text=piece, metadata=piece_meta))

    return chunks


def process_url(url: str) -> List[ChunkRecord]:
    """Run the complete scraping and chunking workflow for one URL."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    breadcrumb = extract_breadcrumb(soup)
    last_updated = extract_last_updated(soup)

    container = select_main_container(soup)
    # Work on a detached clone-like parse to avoid mutating breadcrumb context.
    content_soup = BeautifulSoup(str(container), "html.parser")
    content_root = content_soup.find()
    if content_root is None:
        return []

    _remove_non_content_nodes(content_root)
    markdown_text = html_to_markdown(str(content_root))
    return build_chunks(
        url=url,
        markdown_text=markdown_text,
        breadcrumb=breadcrumb,
        last_updated=last_updated,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape and chunk T&T law/policing HTML pages into markdown chunks."
    )
    parser.add_argument("url", help="Source page URL to scrape")
    parser.add_argument(
        "--output",
        default="chunks.json",
        help="Output JSON file path (default: chunks.json)",
    )
    args = parser.parse_args()

    chunks = process_url(args.url)
    payload = [
        {
            "text": c.text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(payload)} chunks to {args.output}")


if __name__ == "__main__":
    main()
