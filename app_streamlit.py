"""Streamlit UI for the Trinidad and Tobago legal RAG assistant."""

from __future__ import annotations

import streamlit as st

from tt_legal_agent.rag_pipeline import TTRAGPipeline
from tt_legal_agent.vector_store import TTVectorStore
from tt_legal_agent.tools import summarize_document, web_search_gazette


st.set_page_config(page_title="T&T Legal Assistant", page_icon="⚖️")
st.title("Trinidad & Tobago Legal Assistant")
st.caption("RAG + citations over indexed legal and TTPS chunks")

with st.sidebar:
    st.header("Vector DB settings")
    persist_dir = st.text_input("Persist directory", value=".chroma_tt_law")
    collection = st.text_input("Collection", value="tt_law")
    embedding_backend = st.selectbox("Embedding backend", ["local", "openai"], index=0)
    local_embedding_model = st.text_input(
        "Local embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )
    openai_embedding_model = st.text_input(
        "OpenAI embedding model",
        value="text-embedding-3-small",
    )
    answer_model = st.text_input("Answer model", value="gpt-4o-mini")
    top_k = st.slider("Retrieved chunks (k)", 2, 12, 6)


@st.cache_resource(show_spinner=False)
def get_pipeline(
    persist_directory: str,
    collection_name: str,
    embed_backend: str,
    local_model: str,
    openai_model: str,
    llm_model: str,
) -> TTRAGPipeline:
    store = TTVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_backend=embed_backend,
        local_embedding_model=local_model,
        openai_embedding_model=openai_model,
    )
    return TTRAGPipeline(vector_store=store, model=llm_model)


pipeline = get_pipeline(
    persist_dir,
    collection,
    embedding_backend,
    local_embedding_model,
    openai_embedding_model,
    answer_model,
)

question = st.text_area(
    "Ask a legal question",
    placeholder="Example: If a suspect makes an utterance during an interview in Santa Cruz, what is required before continuing?",
)
if st.button("Ask") and question.strip():
    with st.spinner("Retrieving legal chunks and generating answer..."):
        result = pipeline.answer_question(question.strip(), top_k=top_k)
    st.subheader("Answer")
    st.write(result.answer)

    st.subheader("Citations")
    for item in result.citations:
        st.markdown(
            f"- **{item['act_name']}** | Chapter {item['chapter']} | "
            f"Section {item['section']} | {item['url']}"
        )

    with st.expander("Retrieved context"):
        for idx, chunk in enumerate(result.retrieved, start=1):
            meta = chunk.metadata
            st.markdown(
                f"**[{idx}] {meta.get('act_name', 'Unknown')} - "
                f"Ch {meta.get('chapter', 'N/A')} Sec {meta.get('section', 'N/A')}**"
            )
            st.code(chunk.text[:1800])

st.divider()
st.subheader("Agentic tool: Gazette search")
gazette_query = st.text_input("Search Gazette", value="police standing orders statement under caution")
if st.button("Run Gazette search"):
    try:
        results = web_search_gazette(gazette_query, max_results=5)
        for res in results:
            st.markdown(f"- [{res.title}]({res.url})")
            st.caption(res.snippet)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gazette search failed: {exc}")

st.divider()
st.subheader("Agentic tool: Document summarizer")
uploaded = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
if uploaded is not None:
    tmp_path = f"/tmp/{uploaded.name}"
    with open(tmp_path, "wb") as handle:
        handle.write(uploaded.getvalue())
    try:
        summary = summarize_document(tmp_path)
        st.json(summary)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Summarization failed: {exc}")
