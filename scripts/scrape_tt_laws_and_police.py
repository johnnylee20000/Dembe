#!/usr/bin/env python3
"""Scrape Trinidad and Tobago laws and police procedure inventories.

This script pulls data from official/public sources and writes structured
datasets under ./data.
"""

from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency fallback
    PdfReader = None

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass


BASE_LAWS_URL = "https://laws.gov.tt"
AJAX_INDEX_URL = f"{BASE_LAWS_URL}/ttdll-web/revision/Ajaxlist"
TTPS_SITE_URL = "https://www.ttps.gov.tt/"
MNS_PUBLIC_STATEMENT_URL = (
    "https://nationalsecurity.gov.tt/wp-content/uploads/2025/01/"
    "Updated-Public-Statements-2024.pdf"
)
JUDGES_RULES_URL = (
    "https://www.ttlawcourts.org/attachments/article/11951/Judges%20Rules%201965.pdf"
)
POLICE_SERVICE_REGULATIONS_URL = (
    "https://www.ttlawcourts.org/attachments/article/1438/LN145_07.pdf"
)


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
REQUEST_HEADERS = {"User-Agent": USER_AGENT}


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LAWS_DIR = DATA_DIR / "trinidad_tobago_laws"
POLICE_DIR = DATA_DIR / "trinidad_tobago_police"
RAW_DIR = DATA_DIR / "raw"


def normalize_whitespace(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split()).strip()


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment, flags=re.S)
    return normalize_whitespace(html.unescape(text))


def ensure_dirs() -> None:
    for directory in (DATA_DIR, LAWS_DIR, POLICE_DIR, RAW_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    verify: bool = True,
) -> str:
    response = session.get(url, params=params, timeout=120, headers=REQUEST_HEADERS, verify=verify)
    response.raise_for_status()
    return response.text


def fetch_bytes(
    session: requests.Session,
    url: str,
    *,
    verify: bool = True,
) -> bytes:
    response = session.get(url, timeout=180, headers=REQUEST_HEADERS, verify=verify)
    response.raise_for_status()
    return response.content


def extract_table_rows(page_html: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, flags=re.S | re.I):
        link_match = re.search(
            r'href="(/ttdll-web/revision/download/(\d+)\?type=([a-z]+))"',
            row_html,
            flags=re.I,
        )
        if not link_match:
            continue

        cols = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S | re.I)
        reference = strip_tags(cols[1]) if len(cols) > 1 else ""
        title = strip_tags(cols[2]) if len(cols) > 2 else ""

        if not reference:
            strong_match = re.search(r"<strong[^>]*>(.*?)</strong>", row_html, flags=re.S | re.I)
            if strong_match:
                reference = strip_tags(strong_match.group(1))

        rows.append(
            {
                "download_path": link_match.group(1),
                "download_id": link_match.group(2),
                "document_type": link_match.group(3),
                "reference": reference,
                "title": title,
            }
        )
    return rows


def scrape_ajax_law_index(session: requests.Session) -> List[Dict[str, object]]:
    response = session.get(AJAX_INDEX_URL, timeout=120, headers=REQUEST_HEADERS)
    response.raise_for_status()
    payload = response.json()
    output: List[Dict[str, object]] = []
    for item in payload:
        law_id = item.get("id")
        output.append(
            {
                "law_id": law_id,
                "description": item.get("description", ""),
                "chapter_number": item.get("chapternumber", ""),
                "detail_url": (
                    f"{BASE_LAWS_URL}/ttdll-web/revision/list?offset=0&currentid={law_id}"
                ),
            }
        )
    return output


def scrape_paged_revision_table(
    session: requests.Session, listing_path: str, listing_name: str
) -> List[Dict[str, object]]:
    offset = 0
    rows_out: List[Dict[str, object]] = []
    while True:
        url = f"{BASE_LAWS_URL}/ttdll-web/revision/{listing_path}"
        page_html = fetch_text(session, url, params={"offset": offset})
        parsed = extract_table_rows(page_html)
        if not parsed:
            if offset == 0:
                break
            break

        for row in parsed:
            rows_out.append(
                {
                    "listing": listing_name,
                    "offset": offset,
                    "reference": row["reference"],
                    "title": row["title"],
                    "document_type": row["document_type"],
                    "download_id": row["download_id"],
                    "download_url": urljoin(BASE_LAWS_URL, row["download_path"]),
                }
            )
        offset += 30
        if offset > 12000:
            break
    return rows_out


def extract_year_options(page_html: str) -> List[int]:
    select_match = re.search(
        r'<select[^>]*name="year"[^>]*>(.*?)</select>',
        page_html,
        flags=re.S | re.I,
    )
    if not select_match:
        return []
    years = []
    for option_value in re.findall(r'<option[^>]*value="(\d{4})"', select_match.group(1)):
        years.append(int(option_value))
    return sorted(set(years), reverse=True)


def scrape_year_listing(
    session: requests.Session, listing_path: str, listing_name: str
) -> List[Dict[str, object]]:
    base_url = f"{BASE_LAWS_URL}/ttdll-web/revision/{listing_path}"
    initial_html = fetch_text(session, base_url)
    years = extract_year_options(initial_html)
    rows_out: List[Dict[str, object]] = []

    for year in years:
        page_html = fetch_text(session, base_url, params={"year": year, "apply": "Apply"})
        parsed = extract_table_rows(page_html)
        for row in parsed:
            rows_out.append(
                {
                    "listing": listing_name,
                    "year": year,
                    "reference": row["reference"],
                    "title": row["title"],
                    "document_type": row["document_type"],
                    "download_id": row["download_id"],
                    "download_url": urljoin(BASE_LAWS_URL, row["download_path"]),
                }
            )
    return rows_out


@dataclass
class PoliceProcedureRecord:
    procedure_or_policy: str
    availability: str
    source_name: str
    source_url: str
    notes: str

    def to_row(self) -> Dict[str, str]:
        return {
            "procedure_or_policy": self.procedure_or_policy,
            "availability": self.availability,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "notes": self.notes,
        }


def extract_bullet_items(text_block: str) -> List[str]:
    items: List[str] = []
    current = ""
    for raw_line in text_block.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            continue
        if line.startswith("•") or line.startswith("▪") or line.startswith("-"):
            if current:
                items.append(current)
            current = line.lstrip("•▪- ").strip()
        else:
            if current:
                current = f"{current} {line}".strip()
    if current:
        items.append(current)
    return items


def parse_tpps_policy_list_from_public_statement(pdf_path: Path) -> List[str]:
    if PdfReader is None:
        return []

    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        text = page.extract_text() or ""
        if (
            "TRINIDAD AND TOBAGO POLICE SERVICE" in text
            and "Use of Force Policy" in text
            and "IMMIGRATION DIVISION" in text
        ):
            start = text.find("TRINIDAD AND TOBAGO POLICE SERVICE")
            end = text.find("IMMIGRATION DIVISION", start)
            block = text[start:end]
            # Keep only the explicit bullet section.
            block = block.split("TRINIDAD AND TOBAGO POLICE SERVICE", 1)[-1]
            return [item for item in extract_bullet_items(block) if item]
    return []


def parse_orders_list_from_public_statement(pdf_path: Path) -> List[str]:
    if PdfReader is None:
        return []

    reader = PdfReader(str(pdf_path))
    discovered: List[str] = []
    keywords = ("Departmental Orders", "Standing Orders", "Branch Orders")
    for page in reader.pages:
        text = page.extract_text() or ""
        if all(keyword in text for keyword in keywords):
            for keyword in keywords:
                if keyword not in discovered:
                    discovered.append(keyword)
            break
    return discovered


def scrape_police_law_documents(session: requests.Session) -> List[Dict[str, object]]:
    response = session.get(AJAX_INDEX_URL, params={"q": "police"}, timeout=120, headers=REQUEST_HEADERS)
    response.raise_for_status()
    entries = response.json()
    rows: List[Dict[str, object]] = []
    for entry in entries:
        law_id = entry["id"]
        detail_url = f"{BASE_LAWS_URL}/ttdll-web/revision/list?offset=0&currentid={law_id}"
        page_html = fetch_text(session, detail_url)
        title_match = re.search(r"<h4[^>]*>(.*?)</h4>", page_html, flags=re.S | re.I)
        law_heading = strip_tags(title_match.group(1)) if title_match else entry.get("description", "")
        for row in extract_table_rows(page_html):
            rows.append(
                {
                    "law_id": law_id,
                    "law_heading": law_heading,
                    "reference": row["reference"],
                    "title": row["title"],
                    "document_type": row["document_type"],
                    "download_id": row["download_id"],
                    "download_url": urljoin(BASE_LAWS_URL, row["download_path"]),
                }
            )
    return rows


def build_police_procedure_inventory(
    tpps_public_statement_items: List[str],
    orders_items: List[str],
) -> List[PoliceProcedureRecord]:
    records: List[PoliceProcedureRecord] = []

    for item in tpps_public_statement_items:
        records.append(
            PoliceProcedureRecord(
                procedure_or_policy=item,
                availability="Listed publicly; full text generally requested via FOIA process",
                source_name="Ministry of National Security Updated Public Statement 2024",
                source_url=MNS_PUBLIC_STATEMENT_URL,
                notes="Appears in section 8(1)(a)(ii) inventory for Trinidad and Tobago Police Service.",
            )
        )

    for item in orders_items:
        records.append(
            PoliceProcedureRecord(
                procedure_or_policy=item,
                availability="Listed publicly; full text not directly downloadable from the statement",
                source_name="Ministry of National Security Updated Public Statement 2024",
                source_url=MNS_PUBLIC_STATEMENT_URL,
                notes="Document class referenced under TTPS records on interpretation/enforcement guidance.",
            )
        )

    records.extend(
        [
            PoliceProcedureRecord(
                procedure_or_policy="Police Service Regulations, 2007 (Legal Notice No. 145 of 2007)",
                availability="Direct public PDF download",
                source_name="Judiciary of Trinidad and Tobago (Law Library)",
                source_url=POLICE_SERVICE_REGULATIONS_URL,
                notes="Primary procedural/regulatory framework for the Police Service.",
            ),
            PoliceProcedureRecord(
                procedure_or_policy="Judges' Rules and Administrative Directions to the Police (1965)",
                availability="Direct public PDF download",
                source_name="Judiciary of Trinidad and Tobago (Law Library)",
                source_url=JUDGES_RULES_URL,
                notes="Published procedural guidance for police interviews/statements.",
            ),
            PoliceProcedureRecord(
                procedure_or_policy="Trinidad and Tobago Police Service official website",
                availability="Public website; no comprehensive SOP manual download index found",
                source_name="Trinidad and Tobago Police Service",
                source_url=TTPS_SITE_URL,
                notes="Used to verify publicly visible police service resources and links.",
            ),
        ]
    )
    return records


def main() -> None:
    ensure_dirs()
    session = requests.Session()

    revised_laws = scrape_ajax_law_index(session)
    byyear_rows = scrape_year_listing(session, "byyear", "Chronological list by year")
    legal_notice_rows = scrape_year_listing(
        session, "bylegalnotice", "Chronological legal notices by year"
    )
    listed_category_rows: List[Dict[str, object]] = []
    for path, label in (
        ("bytitle", "Alphabetical list"),
        ("byrepealed", "Repealed Acts"),
        ("byomitted", "Omitted Acts"),
        ("byunprocliamed", "Acts not in operation"),
    ):
        listed_category_rows.extend(scrape_paged_revision_table(session, path, label))

    police_law_docs = scrape_police_law_documents(session)

    mns_pdf_path = RAW_DIR / "Updated-Public-Statements-2024.pdf"
    if not mns_pdf_path.exists():
        mns_pdf_path.write_bytes(fetch_bytes(session, MNS_PUBLIC_STATEMENT_URL))

    # This host has certificate-chain issues in this environment.
    judges_rules_path = RAW_DIR / "Judges_Rules_1965.pdf"
    if not judges_rules_path.exists():
        judges_rules_path.write_bytes(fetch_bytes(session, JUDGES_RULES_URL, verify=False))

    police_regs_path = RAW_DIR / "Police_Service_Regulations_2007.pdf"
    if not police_regs_path.exists():
        police_regs_path.write_bytes(fetch_bytes(session, POLICE_SERVICE_REGULATIONS_URL, verify=False))

    tpps_items = parse_tpps_policy_list_from_public_statement(mns_pdf_path)
    orders_items = parse_orders_list_from_public_statement(mns_pdf_path)
    police_inventory = build_police_procedure_inventory(tpps_items, orders_items)

    write_json(LAWS_DIR / "revised_laws_index.json", revised_laws)
    write_csv(
        LAWS_DIR / "revised_laws_index.csv",
        revised_laws,
        ["law_id", "description", "chapter_number", "detail_url"],
    )

    write_csv(
        LAWS_DIR / "laws_by_year.csv",
        byyear_rows,
        ["listing", "year", "reference", "title", "document_type", "download_id", "download_url"],
    )
    write_csv(
        LAWS_DIR / "legal_notices_by_year.csv",
        legal_notice_rows,
        ["listing", "year", "reference", "title", "document_type", "download_id", "download_url"],
    )
    write_csv(
        LAWS_DIR / "revision_category_documents.csv",
        listed_category_rows,
        ["listing", "offset", "reference", "title", "document_type", "download_id", "download_url"],
    )
    write_csv(
        POLICE_DIR / "police_related_laws.csv",
        police_law_docs,
        ["law_id", "law_heading", "reference", "title", "document_type", "download_id", "download_url"],
    )

    inventory_rows = [record.to_row() for record in police_inventory]
    write_csv(
        POLICE_DIR / "police_procedures_public_inventory.csv",
        inventory_rows,
        ["procedure_or_policy", "availability", "source_name", "source_url", "notes"],
    )
    write_json(POLICE_DIR / "police_procedures_public_inventory.json", inventory_rows)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "laws_portal": BASE_LAWS_URL,
            "laws_ajax_index": AJAX_INDEX_URL,
            "mns_public_statement_pdf": MNS_PUBLIC_STATEMENT_URL,
            "judges_rules_pdf": JUDGES_RULES_URL,
            "police_service_regulations_pdf": POLICE_SERVICE_REGULATIONS_URL,
            "ttps_website": TTPS_SITE_URL,
        },
        "counts": {
            "revised_laws_index_items": len(revised_laws),
            "laws_by_year_rows": len(byyear_rows),
            "legal_notice_rows": len(legal_notice_rows),
            "revision_category_rows": len(listed_category_rows),
            "police_related_law_documents": len(police_law_docs),
            "police_procedure_inventory_rows": len(inventory_rows),
            "parsed_ttps_policy_items_from_mns_statement": len(tpps_items),
        },
        "notes": [
            "The laws.gov.tt library explicitly labels itself as providing unofficial versions online.",
            "Some TT police policy/procedure documents are listed publicly in FOIA statements but not directly downloadable.",
            "SSL verification for www.ttlawcourts.org fails in this environment; those two downloads use verify=False.",
        ],
    }
    write_json(DATA_DIR / "scrape_summary.json", summary)

    print("Scrape complete.")
    for key, value in summary["counts"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
