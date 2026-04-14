"""Streamlit UI for a TTPS-style legal assistant."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import TTPSAgent


st.set_page_config(page_title="TTPS Assistant", page_icon="🚓")
st.title("TTPS Assistant")
st.caption("Field-ready legal chat with source-grounded citations")

DEFAULT_VECTOR_DB = "vector_db"
DEFAULT_COLLECTION = "tt_law_chunks"
DEFAULT_EMBED_PROVIDER = "huggingface"


@st.cache_resource(show_spinner=False)
def build_agent(
    vector_db_dir: str,
    collection_name: str,
    embedding_provider: str,
    answer_model: str,
    hf_model: str,
    openai_embedding_model: str,
) -> TTPSAgent:
    return TTPSAgent(
        vector_db_dir=vector_db_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        answer_model=answer_model,
        hf_model=hf_model,
        openai_embedding_model=openai_embedding_model,
    )


with st.sidebar:
    st.header("Assistant settings")
    vector_db_dir = st.text_input("Vector DB directory", value=DEFAULT_VECTOR_DB)
    collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)
    embedding_provider = st.selectbox(
        "Embedding provider",
        options=["huggingface", "openai"],
        index=0,
    )
    hf_model = st.text_input(
        "HuggingFace embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )
    openai_embedding_model = st.text_input(
        "OpenAI embedding model",
        value="text-embedding-3-small",
    )
    answer_model = st.text_input("Answer model", value="gpt-4o-mini")
    top_k = st.slider("Top matches", min_value=1, max_value=10, value=3, step=1)

agent = build_agent(
    vector_db_dir=vector_db_dir,
    collection_name=collection_name,
    embedding_provider=embedding_provider,
    answer_model=answer_model,
    hf_model=hf_model,
    openai_embedding_model=openai_embedding_model,
)

with st.sidebar:
    st.subheader("Loaded Laws/Acts")
    loaded_acts = agent.list_loaded_acts()
    if loaded_acts:
        for act in loaded_acts:
            st.markdown(f"- {act}")
    else:
        st.info("No acts found in the selected vector database.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            st.markdown("**Source Citation**")
            for citation in message["citations"]:
                st.markdown(
                    "- "
                    f"**{citation['act_name']}** | "
                    f"Chapter {citation['chapter']} | "
                    f"Section {citation['section']} | "
                    f"{citation['url']}"
                )

question = st.chat_input("Ask a legal or police-procedure question...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking vector_db and preparing answer..."):
            result = agent.answer(question=question, top_k=top_k)
        st.markdown(result["answer"])
        st.markdown("**Source Citation**")
        for citation in result["citations"]:
            st.markdown(
                "- "
                f"**{citation['act_name']}** | "
                f"Chapter {citation['chapter']} | "
                f"Section {citation['section']} | "
                f"{citation['url']}"
            )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
        }
    )
