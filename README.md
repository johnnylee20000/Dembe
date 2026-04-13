# Dembe

Trinidad and Tobago legal AI assistant scaffold with RAG, vector search, agentic tools, and validation workflow.

## Core deliverables in this repo

- `tt_legal_agent/`: reusable package for chunk loading, embeddings + ChromaDB storage, retrieval, and answer synthesis.
- `scripts/index_chunks.py`: embed and upsert markdown chunks into ChromaDB.
- `scripts/ask_agent.py`: ask legal questions over indexed chunks.
- `scripts/run_badal_test.py`: benchmark script for the "Badal test" style check.
- `app_streamlit.py`: Streamlit chat UI for quick deployment.
- `data/chunks/*.md`: sample T&T law/procedure chunks with citation metadata.
- `requirements.txt`: Python dependencies.

## Architecture implemented

1. **Generate embeddings and upsert to vector DB**
   - Supports embedding backends:
     - OpenAI (`text-embedding-3-small`)
     - Local SentenceTransformers (`sentence-transformers/all-MiniLM-L6-v2`)
   - Stores vectors in **ChromaDB**.
   - Stores metadata with each chunk:
     - `act_name`
     - `chapter`
     - `section`
     - `url`
     - `source_file`

2. **RAG pipeline**
   - Semantic retrieval from ChromaDB.
   - Context injection into strict legal system prompt.
   - Synthesis with LLM (when `OPENAI_API_KEY` is configured).
   - Safe fallback mode if key is absent (returns retrieved context and citations).

3. **Strict system prompt**
   - Located in `tt_legal_agent/system_prompt.py`.
   - Enforces:
     - context-only answers
     - explicit unknown response when missing
     - mandatory citations
     - professional legal tone

4. **Agentic tools**
   - Gazette web search: `web_search_gazette(...)`
   - Document summarizer: `summarize_document(...)` for `.txt`, `.md`, `.pdf`, and `.docx`

5. **Validation ("Badal test")**
   - Script checks for key procedural concepts in answer:
     - statement under caution
     - justice of the peace
     - authentication

6. **Deployment surface**
   - Streamlit UI included.
   - Suitable for deployment on Render or similar Python app hosts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Index legal markdown chunks

Local embeddings:

```bash
python3 scripts/index_chunks.py \
  --chunks-dir data/chunks \
  --persist-dir .chroma_tt_law \
  --collection tt_law \
  --embedding-backend local
```

OpenAI embeddings:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python3 scripts/index_chunks.py \
  --chunks-dir data/chunks \
  --persist-dir .chroma_tt_law \
  --collection tt_law \
  --embedding-backend openai
```

## Ask a question (CLI)

```bash
python3 scripts/ask_agent.py \
  --persist-dir .chroma_tt_law \
  --collection tt_law \
  --embedding-backend local \
  --question "If a suspect makes an utterance during interview, what should happen before continuing?"
```

## Run Badal benchmark

```bash
python3 scripts/run_badal_test.py \
  --chroma-dir .chroma_tt_law \
  --collection tt_law \
  --embedding-backend local
```

## Run Streamlit app

```bash
streamlit run app_streamlit.py
```

## Environment variables

- `OPENAI_API_KEY`: required for OpenAI embeddings and full LLM synthesis.
- Without this key, retrieval still works and the assistant returns a grounded
  context-based fallback response with citations.

## Optional artifacts from prior work

- `ai_agent.py`, `build_agent_cli.py`, `sample_task.json`: generic agent blueprint generator.
- `legal_ai_english_law_report.md`, `legal_ai_english_law_spec.json`: legal AI benchmarking outputs.
