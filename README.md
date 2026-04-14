# Dembe

Trinidad and Tobago legal AI assistant scaffold with RAG, local vector search, and officer-focused UI.

## Required file structure

This project now follows the exact structure requested:

- `/data` - raw scraped files
- `/chunks` - processed, chunked files ready for indexing
- `/vector_db` - local Chroma vector database (created by `ingest.py`)
- `ingest.py` - LangChain ingestion script (`/chunks` -> `/vector_db`)
- `agent.py` - retriever + answer layer for officer questions
- `.cursorrules` - project behavior constraints for TTPS assistant behavior
- `app_streamlit.py` - police officer chat interface

## Core deliverables in this repo

- `ingest.py`: LangChain ingestion script for `/chunks` to local Chroma `/vector_db`.
- `agent.py`: retriever-based assistant that queries `/vector_db` before answering.
- `app_streamlit.py`: officer UI with loaded-laws sidebar, chat window, and source citation box.
- `.cursorrules`: TTPS-specific behavior instructions.
- `chunks/*.md`: sample chunk data seeded from existing T&T examples.
- `vector_db/`: created by ingest script.
- `tt_legal_agent/`: reusable package from earlier iterations.
- `requirements.txt`: Python dependencies.

## Architecture implemented

1. **Generate embeddings and upsert to local vector DB**
   - `ingest.py` initializes local ChromaDB in `/vector_db`
   - Supports embeddings:
     - OpenAI (`text-embedding-3-small`)
     - HuggingFace local (`sentence-transformers/all-MiniLM-L6-v2`)
   - Stores metadata with each chunk:
     - `act_name`, `chapter`, `section`, `url`, `source_file`

2. **Retriever-connected agent**
   - `agent.py` loads `/vector_db`
   - converts officer question to vector
   - retrieves top matches (default top 3)
   - answers with citation-grounded legal text

3. **Cursor behavior rules**
   - `.cursorrules` enforces:
     - query `/vector_db` before answering
     - prioritize relevant Laws of Trinidad and Tobago sections
     - use State v. Allister Badal complaint formatting standard

4. **Police officer UI**
   - sidebar shows loaded Laws/Acts currently in DB
   - main chat window for officer Q&A
   - source citation box under each answer

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingest chunks into `/vector_db`

HuggingFace/local embeddings:

```bash
python3 ingest.py \
  --chunks-dir chunks \
  --vector-db-dir vector_db \
  --collection-name tt_law_chunks \
  --embedding-provider huggingface
```

OpenAI embeddings:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python3 ingest.py \
  --chunks-dir chunks \
  --vector-db-dir vector_db \
  --collection-name tt_law_chunks \
  --embedding-provider openai
```

## Ask a question (CLI agent)

```bash
python3 agent.py \
  --question "What is the ICCS code for 7.62 ammo?" \
  --vector-db-dir vector_db \
  --collection-name tt_law_chunks \
  --k 3
```

## Run Streamlit app

```bash
streamlit run app_streamlit.py
```

## Environment variables

- `OPENAI_API_KEY`: required for OpenAI embeddings and full LLM synthesis in `agent.py`.
- Without this key, retrieval still works and returns a citation-grounded retrieval summary.

## Optional artifacts from prior work

- `ai_agent.py`, `build_agent_cli.py`, `sample_task.json`: generic agent blueprint generator.
- `legal_ai_english_law_report.md`, `legal_ai_english_law_spec.json`: legal AI benchmarking outputs.
