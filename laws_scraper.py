#!/usr/bin/env python3
"""
Scrape Trinidad and Tobago laws from the Digital Legislative Library.

Workflow:
1) Discover Act detail identifiers (currentid) from revised list pages.
2) Visit each Act detail page by currentid.
3) Prefer latest consolidated file (type=act).
4) Fallback to latest amending legislation when consolidated is unavailable.
5) Save metadata and download PDFs.

Notes:
- This site can return stale/default detail panels for some search-only URLs.
  The scraper avoids title-only resolution and uses currentid-based URLs.
- Crawl4AI is optional; requests+BeautifulSoup is the default path.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler  # type: ignore

    HAS_CRAWL4AI = True
except Exception:  # pragma: no cover
    HAS_CRAWL4AI = False


BASE_URL = "https://laws.gov.tt"
LIST_URL = f"{BASE_URL}/ttdll-web/revision/list"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class DownloadCandidate:
    url: str
    label: str
    as_at_text: str
    category: str  # "consolidated" | "amending_legislation"


@dataclass
class ActRecord:
    current_id: int
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
            "current_id": str(self.current_id),
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
            except Exception as exc:  # pragma: no cover
                logging.warning("Crawl4AI failed; falling back to requests for %s: %s", url, exc)
        response = self._request(url)
        self._sleep()
        return response.text

    @staticmethod
    def _safe_filename_component(value: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
        return text.strip("_") or "unnamed"

    @staticmethod
    def _normalize_space(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _extract_chapter(text: str) -> str:
        match = re.search(
            r"(?:Chap(?:ter)?\.?\s*)(\d+[:.]\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(r"\bCHAPTER\s+(\d+:\d+)\b", text, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_year(detail_soup: BeautifulSoup) -> Optional[int]:
        text = detail_soup.get_text(" ", strip=True)
        match = re.search(r"Commencement Date\s*-\s*[^0-9]*(\d{4})", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        fallback = re.search(r"Year\s*-\s*(\d{4})", text, flags=re.IGNORECASE)
        return int(fallback.group(1)) if fallback else None

    @staticmethod
    def _extract_act_name(detail_soup: BeautifulSoup) -> str:
        heading = detail_soup.select_one("#law-detail h4 a") or detail_soup.select_one("h4 a")
        if not heading:
            return "Unknown Act"
        value = heading.get_text(" ", strip=True)
        value = re.sub(r"\s+Chap\.?\s*\d+[:.]\d+\s*$", "", value, flags=re.IGNORECASE).strip()
        return value or "Unknown Act"

    @staticmethod
    def _extract_heading_text(detail_soup: BeautifulSoup) -> str:
        heading = detail_soup.select_one("#law-detail h4 a") or detail_soup.select_one("h4 a")
        return heading.get_text(" ", strip=True) if heading else ""

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
        # Lists on these pages are newest-first.
        return candidates[0] if candidates else None

    def _download_file(self, candidate: DownloadCandidate, act_name: str, chapter: str) -> tuple[str, str]:
        chapter_part = chapter or "nochapter"
        kind = "consolidated" if candidate.category == "consolidated" else "amendment"
        filename = f"{self._safe_filename_component(act_name)}_{chapter_part}_{kind}.pdf"
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

    def discover_current_ids(
        self,
        *,
        max_offset: int = 5000,
        step: int = 10,
        target_count: Optional[int] = None,
    ) -> list[int]:
        """
        Crawl revised list offsets and collect currentid values.
        """
        all_ids: set[int] = set()
        stale_pages = 0
        consecutive_errors = 0

        for offset in range(0, max_offset + 1, step):
            page_url = f"{LIST_URL}?offset={offset}"
            logging.info("Scanning revised list offset=%s", offset)
            try:
                html = self._fetch_html(page_url)
            except Exception as exc:
                consecutive_errors += 1
                logging.warning(
                    "Skipping offset=%s due to repeated server error (%s/%s): %s",
                    offset,
                    consecutive_errors,
                    3,
                    exc,
                )
                if consecutive_errors >= 3:
                    break
                continue
            consecutive_errors = 0

            page_ids = {int(v) for v in re.findall(r"currentid=(\d+)", html)}
            previous_count = len(all_ids)
            all_ids.update(page_ids)
            added = len(all_ids) - previous_count

            if target_count is not None and len(all_ids) >= target_count:
                break

            if added == 0:
                stale_pages += 1
            else:
                stale_pages = 0

            # After enough empty-growth pages, stop probing.
            if offset >= 500 and stale_pages >= 20:
                break

        return sorted(all_ids)

    def _detail_url(self, current_id: int) -> str:
        # currentid is the reliable selector for the active Act panel.
        return f"{LIST_URL}?offset=0&q=&currentid={current_id}#email-content"

    def scrape(self, limit: Optional[int] = None) -> list[ActRecord]:
        current_ids = self.discover_current_ids(target_count=limit)
        logging.info("Discovered %s unique currentid values", len(current_ids))
        if limit is not None:
            current_ids = current_ids[:limit]

        records: list[ActRecord] = []
        for idx, current_id in enumerate(current_ids, start=1):
            detail_url = self._detail_url(current_id)
            logging.info("[%s/%s] Processing currentid=%s", idx, len(current_ids), current_id)

            try:
                html = self._fetch_html(detail_url)
                soup = BeautifulSoup(html, "html.parser")

                act_name = self._extract_act_name(soup)
                heading_text = self._extract_heading_text(soup)
                chapter = self._extract_chapter(heading_text)
                if not chapter:
                    all_text = soup.get_text(" ", strip=True)
                    chapter = self._extract_chapter(all_text)
                year = self._extract_year(soup)
                consolidated, amendments = self._extract_candidates(soup)

                chosen = self._choose_latest(consolidated)
                note = ""
                if not chosen:
                    chosen = self._choose_latest(amendments)
                    if chosen:
                        note = "No consolidated version found; used latest amending legislation."
                    else:
                        note = "No downloadable PDF candidates found."

                if chosen:
                    status, local_file = self._download_file(chosen, act_name, chapter)
                    selected_url = chosen.url
                    selected_label = self._normalize_space(chosen.label)
                    selected_as_at = self._normalize_space(chosen.as_at_text)
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
                        current_id=current_id,
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
                logging.exception("Failed processing currentid=%s: %s", current_id, exc)
                records.append(
                    ActRecord(
                        current_id=current_id,
                        act_name="",
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
            "current_id",
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
            "Scrape laws.gov.tt revised Acts with consolidated-first and amendment fallback."
        )
    )
    parser.add_argument("--output-dir", default="data/laws_tt", help="Output directory.")
    parser.add_argument("--metadata-file", default="metadata.csv", help="Metadata CSV filename.")
    parser.add_argument("--delay-seconds", type=float, default=1.25, help="Delay between requests.")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="HTTP timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="HTTP retry count.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on records processed.")
    parser.add_argument(
        "--crawl4ai",
        action="store_true",
        help="Use Crawl4AI for HTML retrieval if installed.",
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
