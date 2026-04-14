#!/usr/bin/env python3
"""
Streamlit interface for the TTPS Assistant.

Tabs:
1) Legal Advisor (RAG chat)
2) File Drafter (Complaint on Oath generation)
3) ICCS Lookup (quick offence/legal lookup)
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from assistant_core import build_retrieval_qa_chain, format_source_documents
from complaint_drafter import draft_complaint


def _safe_chain_invoke(chain, query: str) -> dict[str, Any]:
    result = chain.invoke({"query": query})
    return {
        "answer": result.get("result", ""),
        "sources": format_source_documents(result.get("source_documents", [])),
    }


@st.cache_resource(show_spinner=False)
def get_chain():
    vector_db_path = os.getenv("VECTOR_DB_PATH", "/vector_db")
    llm_provider = os.getenv("LLM_PROVIDER", None)
    model_name = os.getenv("OPENAI_MODEL", None) or os.getenv("OLLAMA_MODEL", None)
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", None)
    return build_retrieval_qa_chain(
        vector_db_path=vector_db_path,
        llm_provider=llm_provider,
        model_name=model_name,
        embedding_provider=embedding_provider,
        k=int(os.getenv("RAG_K", "6")),
    )


def render_legal_advisor_tab():
    st.subheader("Legal Advisor")
    st.caption("Ask legal powers/procedural questions and see source Act/Section chunks.")
    query = st.text_input(
        "Officer question",
        placeholder="What are my powers under the Dangerous Drugs Act?",
        key="legal_query",
    )
    if st.button("Get legal answer", key="legal_btn"):
        if not query.strip():
            st.warning("Enter a question first.")
            return
        with st.spinner("Retrieving legal context..."):
            try:
                chain = get_chain()
                payload = _safe_chain_invoke(chain, query.strip())
                st.markdown("### Answer")
                st.write(payload["answer"])
                st.markdown("### Source Documents")
                for idx, src in enumerate(payload["sources"], start=1):
                    st.markdown(f"**Source {idx}**")
                    st.json(src["metadata"])
                    st.write(src["snippet"])
            except Exception as exc:
                st.error(f"Legal advisor failed: {exc}")


def render_file_drafter_tab():
    st.subheader("File Drafter")
    st.caption("Generate Complaint on Oath style Statement + Particulars from incident notes.")
    officer_notes = st.text_area(
        "Incident details",
        placeholder="Stopped man in Santa Cruz with a rifle in waistband, no licence produced.",
        height=180,
        key="drafter_notes",
    )
    if st.button("Draft Complaint on Oath", key="drafter_btn"):
        if not officer_notes.strip():
            st.warning("Enter incident details first.")
            return
        with st.spinner("Drafting complaint..."):
            try:
                payload = draft_complaint(
                    officer_notes=officer_notes.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    llm_provider=os.getenv("LLM_PROVIDER", None),
                    model_name=os.getenv("OPENAI_MODEL", None) or os.getenv("OLLAMA_MODEL", None),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("DRAFTER_K", "8")),
                )
                st.markdown("### Draft Output")
                st.code(payload["draft"], language="markdown")
                st.markdown("### Retrieved Legal Sources")
                for idx, src in enumerate(payload["sources"], start=1):
                    st.markdown(f"**Source {idx}**")
                    st.json(src["metadata"])
                    st.write(src["snippet"])
            except Exception as exc:
                st.error(f"Drafting failed: {exc}")


def render_iccs_lookup_tab():
    st.subheader("ICCS Lookup")
    st.caption("Quick-search offences and likely legal references/penalties.")
    query = st.text_input(
        "Offence keyword",
        placeholder="firearm possession without licence",
        key="iccs_query",
    )
    if st.button("Lookup ICCS / Offence", key="iccs_btn"):
        if not query.strip():
            st.warning("Enter an offence query first.")
            return
        with st.spinner("Searching ICCS/legal references..."):
            try:
                chain = get_chain()
                lookup_prompt = (
                    "Return ICCS code, offence name, Act/Section, and penalty summary if present "
                    f"in context. Query: {query.strip()}"
                )
                payload = _safe_chain_invoke(chain, lookup_prompt)
                st.markdown("### Lookup Result")
                st.write(payload["answer"])
                st.markdown("### Source Documents")
                for idx, src in enumerate(payload["sources"], start=1):
                    st.markdown(f"**Source {idx}**")
                    st.json(src["metadata"])
                    st.write(src["snippet"])
            except Exception as exc:
                st.error(f"Lookup failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="TTPS Assistant", page_icon="⚖️", layout="wide")
    st.title("TTPS Assistant")
    st.caption("Legal Advisor + Complaint Drafter + ICCS Lookup")

    tab1, tab2, tab3 = st.tabs(["Legal Advisor", "File Drafter", "ICCS Lookup"])
    with tab1:
        render_legal_advisor_tab()
    with tab2:
        render_file_drafter_tab()
    with tab3:
        render_iccs_lookup_tab()

    st.markdown("---")
    st.caption(
        "Disclaimer: Digital Legislative Library content is generally unofficial online material. "
        "Always verify against official revised laws and current prosecutorial guidance."
    )


if __name__ == "__main__":
    main()
