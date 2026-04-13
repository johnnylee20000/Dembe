#!/usr/bin/env python3
"""
Scrape Acts from the Trinidad and Tobago Digital Legislative Library.

Features:
- Navigates the alphabetical Acts pages.
- Resolves each listed Act to its detail page.
- Prefers the latest consolidated version (type=act) for download.
- Falls back to latest amending legislation if no consolidated version exists.
- Saves metadata:
  - Act Name
  - Chapter Number
  - Year of Commencement
  - Source URL
  - Selected file URL/category/label
- Handles broken links and transient request failures.
- Includes configurable request delay for polite rate limiting.

Optional:
- If Crawl4AI is installed and --crawl4ai is provided, it is used to fetch HTML.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler  # type: ignore

    HAS_CRAWL4AI = True
except Exception:  # pragma: no cover - optional dependency.
    HAS_CRAWL4AI = False

BASE_URL = "https://laws.gov.tt"
ALPHA_URL = f"{BASE_URL}/ttdll-web2/revision/bytitle"
DETAIL_URL = f"{BASE_URL}/ttdll-web/revision/list"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ALPHA_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")


@dataclass(frozen=True)
class DownloadCandidate:
    url: str
    label: str
    as_at_text: str
    category: str  # "consolidated" | "amending_legislation"


@dataclass
class ActRecord:
    act_name: str
    chapter_number: str
    year_of_commencement: Optional[int]
    source_page_url: str
    selected_file_url: str
    selected_file_label: str
    selected_file_as_at: str
    selected_file_category: str
    download_status: str
    local_file: str
    note: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "act_name": self.act_name,
            "chapter_number": self.chapter_number,
            "year_of_commencement": str(self.year_of_commencement or ""),
            "source_page_url": self.source_page_url,
            "selected_file_url": self.selected_file_url,
            "selected_file_label": self.selected_file_label,
            "selected_file_as_at": self.selected_file_as_at,
            "selected_file_category": self.selected_file_category,
            "download_status": self.download_status,
            "local_file": self.local_file,
            "note": self.note,
        }


class LawsTTScraper:
    def __init__(
        self,
        output_dir: Path,
        delay_seconds: float = 1.25,
        timeout_seconds: int = 45,
        max_retries: int = 3,
        use_crawl4ai: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.pdf_dir = output_dir / "pdfs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.use_crawl4ai = use_crawl4ai and HAS_CRAWL4AI

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _sleep(self) -> None:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def _request(self, url: str, *, stream: bool = False) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout_seconds,
                    stream=stream,
                    allow_redirects=True,
                )
                if response.status_code >= 400:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} for {url}",
                        response=response,
                    )
                return response
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                logging.warning(
                    "Request failed (%s/%s) for %s: %s",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Failed request after retries: {url}; last_error={last_error}")

    async def _fetch_with_crawl4ai(self, url: str) -> str:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            html = getattr(result, "html", None) or getattr(result, "cleaned_html", None)
            if not html:
                raise RuntimeError(f"Crawl4AI returned no HTML for {url}")
            return html

    def _fetch_html(self, url: str) -> str:
        if self.use_crawl4ai:
            try:
                html = asyncio.run(self._fetch_with_crawl4ai(url))
                self._sleep()
                return html
            except Exception as exc:  # pragma: no cover - optional path.
                logging.warning("Crawl4AI fetch failed, falling back to requests for %s: %s", url, exc)
        response = self._request(url)
        self._sleep()
        return response.text

    @staticmethod
    def _safe_filename_component(value: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
        return text.strip("_") or "unnamed"

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_chapter_number(text: str) -> str:
        match = re.search(r"(?:Chap(?:ter)?\.?\s*)(\d+[:.]\d+)", text, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_year(detail_soup: BeautifulSoup) -> Optional[int]:
        text = detail_soup.get_text(" ", strip=True)
        match = re.search(r"Commencement Date\s*-\s*[^0-9]*(\d{4})", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        fallback = re.search(r"Year\s*-\s*(\d{4})", text, flags=re.IGNORECASE)
        return int(fallback.group(1)) if fallback else None

    def _parse_alpha_page(
        self, letter: str, offset: int
    ) -> tuple[list[tuple[str, str, str]], Optional[int]]:
        url = f"{ALPHA_URL}?q={quote_plus(letter)}&max=30&offset={offset}"
        html = self._fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        rows: list[tuple[str, str, str]] = []
        for tr in soup.select("#list-acts tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            ref_text = self._normalize_whitespace(tds[1].get_text(" ", strip=True))
            title_text = self._normalize_whitespace(tds[2].get_text(" ", strip=True))
            download_link = tr.select_one("a[href*='/revision/download/']")
            href = download_link.get("href", "").strip() if download_link else ""
            if title_text:
                rows.append((ref_text, title_text, href))

        next_offset: Optional[int] = None
        next_link = soup.select_one("ul.pagination li.next a.step")
        if next_link:
            href = next_link.get("href", "")
            match = re.search(r"offset=(\d+)", href)
            if match:
                candidate = int(match.group(1))
                if candidate > offset:
                    next_offset = candidate

        return rows, next_offset

    def gather_alphabetical_acts(self, max_pages_per_letter: int = 400) -> list[dict[str, str]]:
        """
        Returns a deduplicated list of act hints from the alphabetical pages.
        """
        all_rows: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()

        for letter in ALPHA_LETTERS:
            offset = 0
            pages = 0
            while pages < max_pages_per_letter:
                pages += 1
                logging.info("Scanning alphabetical page letter=%s offset=%s", letter, offset)
                rows, next_offset = self._parse_alpha_page(letter, offset)
                if not rows:
                    break

                for ref, title, alpha_href in rows:
                    key = (title.lower(), ref.lower())
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_rows.append(
                        {
                            "reference": ref,
                            "title": title,
                            "alpha_download_href": alpha_href,
                        }
                    )

                if next_offset is None:
                    break
                offset = next_offset

        return all_rows

    def _detail_page_url_from_title(self, title: str) -> str:
        return f"{DETAIL_URL}?offset=0&q={quote_plus(title)}"

    @staticmethod
    def _extract_currentid_candidates(search_soup: BeautifulSoup) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []
        for anchor in search_soup.select("#law-list a[href*='currentid=']"):
            href = anchor.get("href", "")
            match = re.search(r"currentid=(\d+)", href)
            if not match:
                continue
            current_id = int(match.group(1))
            title = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
            candidates.append((current_id, title))
        return candidates

    def _resolve_detail_page(self, title: str) -> tuple[str, BeautifulSoup]:
        search_url = self._detail_page_url_from_title(title)
        search_html = self._fetch_html(search_url)
        search_soup = BeautifulSoup(search_html, "html.parser")

        candidates = self._extract_currentid_candidates(search_soup)
        if not candidates:
            return search_url, search_soup

        best_id = max(
            candidates,
            key=lambda item: SequenceMatcher(
                a=item[1].lower(),
                b=title.lower(),
            ).ratio(),
        )[0]
        detail_url = f"{DETAIL_URL}?offset=0&q={quote_plus(title)}&currentid={best_id}#email-content"
        detail_html = self._fetch_html(detail_url)
        return detail_url, BeautifulSoup(detail_html, "html.parser")

    @staticmethod
    def _extract_act_name(detail_soup: BeautifulSoup, fallback_title: str) -> str:
        heading = detail_soup.select_one("#law-detail h4 a") or detail_soup.select_one("h4 a")
        if heading:
            value = heading.get_text(" ", strip=True)
            value = re.sub(r"\s+Chap\.?\s*\d+[:.]\d+\s*$", "", value, flags=re.IGNORECASE).strip()
            if value:
                return value
        return fallback_title

    @staticmethod
    def _extract_candidates(detail_soup: BeautifulSoup) -> tuple[list[DownloadCandidate], list[DownloadCandidate]]:
        consolidated: list[DownloadCandidate] = []
        amendments: list[DownloadCandidate] = []

        for item in detail_soup.select("#activities li.list-group-item"):
            anchor = item.select_one("a[href*='/revision/download/']")
            if not anchor:
                continue
            href = anchor.get("href", "").strip()
            if not href:
                continue
            file_url = urljoin(BASE_URL, href)
            file_type = parse_qs(urlparse(file_url).query).get("type", [""])[0]
            label_node = item.select_one("strong.block")
            as_at_node = item.select_one("small.pull-right")
            candidate = DownloadCandidate(
                url=file_url,
                label=label_node.get_text(" ", strip=True) if label_node else "Unlabeled version",
                as_at_text=as_at_node.get_text(" ", strip=True) if as_at_node else "",
                category="consolidated" if file_type == "act" else "amending_legislation",
            )
            if candidate.category == "consolidated":
                consolidated.append(candidate)
            else:
                amendments.append(candidate)

        return consolidated, amendments

    @staticmethod
    def _choose_latest(candidates: list[DownloadCandidate]) -> Optional[DownloadCandidate]:
        # On this site, lists are ordered newest-first.
        return candidates[0] if candidates else None

    def _download_file(self, candidate: DownloadCandidate, act_name: str, chapter: str) -> tuple[str, str]:
        chapter_part = chapter or "nochapter"
        category_part = "consolidated" if candidate.category == "consolidated" else "amendment"
        filename = f"{self._safe_filename_component(act_name)}_{chapter_part}_{category_part}.pdf"
        path = self.pdf_dir / filename
        try:
            response = self._request(candidate.url, stream=True)
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)
            self._sleep()
            return "downloaded", str(path)
        except Exception as exc:
            logging.error("Broken link or download error for %s: %s", candidate.url, exc)
            return "broken_link", ""

    def scrape(self, limit: Optional[int] = None) -> list[ActRecord]:
        act_hints = self.gather_alphabetical_acts()
        logging.info("Discovered %s unique alphabetical entries", len(act_hints))
        if limit is not None:
            act_hints = act_hints[:limit]

        records: list[ActRecord] = []
        for idx, hint in enumerate(act_hints, start=1):
            title = hint["title"]
            ref = hint["reference"]
            logging.info("[%s/%s] Processing %s", idx, len(act_hints), title)

            try:
                detail_url, soup = self._resolve_detail_page(title)

                act_name = self._extract_act_name(soup, fallback_title=title)
                chapter = self._extract_chapter_number(soup.get_text(" ", strip=True)) or self._extract_chapter_number(ref)
                year = self._extract_year(soup)
                consolidated, amendments = self._extract_candidates(soup)

                chosen = self._choose_latest(consolidated)
                note = ""
                if not chosen:
                    chosen = self._choose_latest(amendments)
                    if chosen:
                        note = "No consolidated version found; used latest amending legislation."
                    else:
                        alpha_href = hint.get("alpha_download_href", "")
                        if alpha_href:
                            alpha_url = urljoin(BASE_URL, alpha_href)
                            alpha_type = parse_qs(urlparse(alpha_url).query).get("type", [""])[0]
                            chosen = DownloadCandidate(
                                url=alpha_url,
                                label="Alphabetical listing download",
                                as_at_text="",
                                category="consolidated"
                                if alpha_type == "act"
                                else "amending_legislation",
                            )
                            note = (
                                "No PDF listed on detail page; used alphabetical listing "
                                "download link fallback."
                            )
                        else:
                            note = "No PDF listed on detail page."

                if chosen:
                    status, local_file = self._download_file(chosen, act_name, chapter)
                    selected_url = chosen.url
                    selected_label = chosen.label
                    selected_as_at = chosen.as_at_text
                    selected_category = chosen.category
                else:
                    status = "missing_pdf"
                    local_file = ""
                    selected_url = ""
                    selected_label = ""
                    selected_as_at = ""
                    selected_category = ""

                records.append(
                    ActRecord(
                        act_name=act_name,
                        chapter_number=chapter,
                        year_of_commencement=year,
                        source_page_url=detail_url,
                        selected_file_url=selected_url,
                        selected_file_label=selected_label,
                        selected_file_as_at=selected_as_at,
                        selected_file_category=selected_category,
                        download_status=status,
                        local_file=local_file,
                        note=note,
                    )
                )
            except Exception as exc:
                logging.exception("Failed processing %s: %s", title, exc)
                records.append(
                    ActRecord(
                        act_name=title,
                        chapter_number="",
                        year_of_commencement=None,
                        source_page_url=detail_url,
                        selected_file_url="",
                        selected_file_label="",
                        selected_file_as_at="",
                        selected_file_category="",
                        download_status="error",
                        local_file="",
                        note=str(exc),
                    )
                )

        return records

    def write_metadata(self, records: list[ActRecord], metadata_path: Path) -> None:
        fieldnames = [
            "act_name",
            "chapter_number",
            "year_of_commencement",
            "source_page_url",
            "selected_file_url",
            "selected_file_label",
            "selected_file_as_at",
            "selected_file_category",
            "download_status",
            "local_file",
            "note",
        ]
        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.as_csv_row())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape laws.gov.tt alphabetical Acts list; download consolidated PDFs "
            "with amendment fallback and write metadata."
        )
    )
    parser.add_argument("--output-dir", default="data/laws_tt", help="Output directory.")
    parser.add_argument("--metadata-file", default="metadata.csv", help="Metadata CSV filename.")
    parser.add_argument("--delay-seconds", type=float, default=1.25, help="Rate-limit delay between requests.")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="HTTP timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="HTTP retry count.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of acts to process.")
    parser.add_argument(
        "--crawl4ai",
        action="store_true",
        help="Use Crawl4AI for page retrieval when installed.",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scraper = LawsTTScraper(
        output_dir=output_dir,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        use_crawl4ai=args.crawl4ai,
    )
    records = scraper.scrape(limit=args.limit)
    metadata_path = output_dir / args.metadata_file
    scraper.write_metadata(records, metadata_path)

    downloaded = sum(1 for r in records if r.download_status == "downloaded")
    fallback = sum(1 for r in records if r.selected_file_category == "amending_legislation")
    logging.info(
        "Completed %s records; downloaded=%s; amendment_fallbacks=%s",
        len(records),
        downloaded,
        fallback,
    )
    logging.info("Metadata CSV written to %s", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
