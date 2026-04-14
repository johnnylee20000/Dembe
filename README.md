# Trinidad & Tobago Laws Scraping + Legal Chunking

This repository provides five Python utilities:

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

3. `retrieval_qa.py`
   - Builds a LangChain `RetrievalQA` ("stuff") chain over local Chroma.
   - Loads vector DB from `/vector_db` (with local fallback path checks).
   - Returns source documents for answer traceability.

4. `complaint_drafter.py`
   - Exposes specialized `draft_complaint` generation.
   - Uses one-shot Badal-style drafting prompt.
   - Retrieves legal context (ICCS / Act / Section) from vector store before drafting.

5. `app.py`
   - Streamlit interface with:
     - Tab 1: Legal Advisor
     - Tab 2: File Drafter
     - Tab 3: ICCS Lookup

6. `build_vector_db.py`
   - Builds a persisted local Chroma DB from chunk JSONL files.
   - Defaults to indexing `data/*chunks*.jsonl` into `/workspace/vector_db`.

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

## RetrievalQA Usage

```bash
python3 retrieval_qa.py \
  --vector-db-path /vector_db \
  --question "What are my powers under the Dangerous Drugs Act?" \
  --model-name gpt-4o
```

Notes:
- Uses `chain_type="stuff"`.
- Returns source documents in JSON output.
- If `OPENAI_API_KEY` is absent, falls back to local Ollama model config.

## Build Vector DB (if not already present)

```bash
python3 build_vector_db.py \
  --chunks-glob "data/*chunks*.jsonl" \
  --vector-db-path /workspace/vector_db
```

## Complaint Drafter Usage

```bash
python3 complaint_drafter.py \
  --vector-db-path /vector_db \
  --officer-notes "Stopped man in Santa Cruz with a rifle and no licence."
```

Output includes:
- ICCS Code line
- Act/Section line
- Statement of Offence
- Particulars of Offence
- source-document metadata/snippets

## Streamlit App

```bash
streamlit run app.py
```

Expected environment:
- `VECTOR_DB_PATH` (default `/vector_db`)
- `OPENAI_API_KEY` for GPT-4o **or** local Ollama configured via `OLLAMA_MODEL`

## Legal Validity Notice

Digital Legislative Library files are published as online reference material and may be unofficial. For official legal use, consult the authoritative printed revised laws and relevant legal professionals.
