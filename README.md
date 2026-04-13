# Trinidad & Tobago Laws Scraping + Legal Chunking

This repository provides two Python utilities:

1. `laws_scraper.py`
   - Scrapes the Digital Legislative Library revised Acts pages.
   - Discovers and crawls Act detail records by `currentid`.
   - Prioritizes the latest **Consolidated** file (`type=act`).
   - Falls back to **Amending Legislation** when consolidated is unavailable.
   - Downloads PDFs and writes metadata CSV.

2. `legal_chunker.py`
   - Chunks legal PDFs using structure-first logic.
   - Detects `Section` and `Schedule` headings.
   - Keeps sections atomic when section token estimate is <= 1000.
   - Uses recursive character splitting for oversized sections.
   - Adds chunk headers in the format: `[Act Name] - Section [Number]`.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Scraper Usage

```bash
python3 laws_scraper.py \
  --output-dir data/laws_tt \
  --delay-seconds 1.25 \
  --limit 25
```

Key options:
- `--crawl4ai`: use Crawl4AI for HTML fetch (if installed).
- `--limit`: process only the first N discovered `currentid` records.
- `--metadata-file`: output CSV file name inside output directory.
- `--max-retries`: retries for transient request failures.

Output:
- PDFs in `data/laws_tt/pdfs/`
- Metadata CSV in `data/laws_tt/metadata.csv`

## Chunker Usage

```bash
python3 legal_chunker.py "data/laws_tt/pdfs/example.pdf" \
  --act-name "Example Act" \
  --output data/chunks/example.jsonl \
  --target-chunk-tokens 1000 \
  --max-chunk-tokens 1200 \
  --overlap-ratio 0.18 \
  --atomic-threshold-tokens 1000
```

Output format is JSONL with one chunk per line, including:
- `act_name`
- `section_number`
- `chunk_index`
- `token_estimate`
- `header`
- `text`
- `full_text`

## Legal Validity Notice

Digital Legislative Library files are published as online reference material and may be unofficial. For official legal use, consult the authoritative printed revised laws and relevant legal professionals.
