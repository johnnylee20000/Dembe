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

from assistant_core import (
    build_retrieval_qa_chain,
    format_source_documents,
    retrieval_only_search,
)
from complaint_drafter import draft_complaint
from prosecution_audit import run_defense_anticipation, run_prosecution_audit


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
                st.warning(
                    "LLM response unavailable; showing retrieved legal context only. "
                    f"Reason: {exc}"
                )
                sources = retrieval_only_search(
                    query=query.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("RAG_K", "6")),
                )
                st.markdown("### Retrieved Sources (No Generated Answer)")
                for idx, src in enumerate(sources, start=1):
                    st.markdown(f"**Source {idx}**")
                    st.json(src["metadata"])
                    st.write(src["snippet"])


def render_file_drafter_tab():
    st.subheader("File Drafter")
    st.caption("Generate a structurally standard TT Complaint on Oath from incident notes.")
    officer_notes = st.text_area(
        "Incident details",
        placeholder="Observed suspect in possession of a firearm without visible lawful authority.",
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
                st.warning(
                    "LLM response unavailable; showing retrieved legal context only. "
                    f"Reason: {exc}"
                )
                sources = retrieval_only_search(
                    query=query.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("RAG_K", "6")),
                )
                st.markdown("### Retrieved Sources (No Generated Answer)")
                for idx, src in enumerate(sources, start=1):
                    st.markdown(f"**Source {idx}**")
                    st.json(src["metadata"])
                    st.write(src["snippet"])


def render_prosecution_audit_tab():
    st.subheader("Prosecution Audit")
    st.caption(
        "Audit officer statements for essential elements, mens rea sufficiency, "
        "and admissibility-focused weaknesses."
    )
    statement = st.text_area(
        "Officer statement / notes for audit",
        placeholder=(
            "Describe facts, sequence, seizure details, suspect conduct, and any caution/interview steps."
        ),
        height=220,
        key="audit_statement",
    )
    if st.button("Run Prosecution Audit", key="audit_btn"):
        if not statement.strip():
            st.warning("Enter statement details first.")
            return
        with st.spinner("Running prosecution audit..."):
            try:
                payload = run_prosecution_audit(
                    officer_statement=statement.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    llm_provider=os.getenv("LLM_PROVIDER", None),
                    model_name=os.getenv("OPENAI_MODEL", None) or os.getenv("OLLAMA_MODEL", None),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("AUDIT_K", "10")),
                )
            except Exception as exc:
                st.warning(
                    "LLM or advanced analysis unavailable; running retrieval-only prosecution audit. "
                    f"Reason: {exc}"
                )
                payload = run_prosecution_audit(
                    officer_statement=statement.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("AUDIT_K", "10")),
                    retrieval_only=True,
                )

            st.markdown("### Audit Findings")
            st.code(payload["analysis"], language="markdown")
            st.markdown("### Source Documents")
            for idx, src in enumerate(payload["sources"], start=1):
                st.markdown(f"**Source {idx}**")
                st.json(src["metadata"])
                st.write(src["snippet"])


def render_defense_strategy_tab():
    st.subheader("Defense Anticipation")
    st.caption(
        "Identify defense loopholes for search/arrest/interview steps and generate prosecution rebuttals."
    )
    statement = st.text_area(
        "Procedure notes for loophole detection",
        placeholder=(
            "Include search grounds, warrant status, caution timing, utterances, arrest chronology."
        ),
        height=220,
        key="defense_statement",
    )
    if st.button("Run Defense Anticipation", key="defense_btn"):
        if not statement.strip():
            st.warning("Enter procedure notes first.")
            return
        with st.spinner("Running loophole detection and rebuttal strategy..."):
            try:
                payload = run_defense_anticipation(
                    officer_statement=statement.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    llm_provider=os.getenv("LLM_PROVIDER", None),
                    model_name=os.getenv("OPENAI_MODEL", None) or os.getenv("OLLAMA_MODEL", None),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("AUDIT_K", "10")),
                )
            except Exception as exc:
                st.warning(
                    "LLM or advanced analysis unavailable; running retrieval-only defense audit. "
                    f"Reason: {exc}"
                )
                payload = run_defense_anticipation(
                    officer_statement=statement.strip(),
                    vector_db_path=os.getenv("VECTOR_DB_PATH", "/vector_db"),
                    embedding_provider=os.getenv("EMBEDDING_PROVIDER", None),
                    k=int(os.getenv("AUDIT_K", "10")),
                    retrieval_only=True,
                )

            st.markdown("### Defense / Rebuttal Analysis")
            st.code(payload["analysis"], language="markdown")
            st.markdown("### Source Documents")
            for idx, src in enumerate(payload["sources"], start=1):
                st.markdown(f"**Source {idx}**")
                st.json(src["metadata"])
                st.write(src["snippet"])


def main() -> None:
    st.set_page_config(page_title="TTPS Assistant", page_icon="⚖️", layout="wide")
    st.title("TTPS Assistant")
    st.caption("Legal Advisor + Complaint Drafter + ICCS Lookup + Prosecution Audit")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Legal Advisor",
            "File Drafter",
            "ICCS Lookup",
            "Prosecution Audit",
            "Defense Anticipation",
        ]
    )
    with tab1:
        render_legal_advisor_tab()
    with tab2:
        render_file_drafter_tab()
    with tab3:
        render_iccs_lookup_tab()
    with tab4:
        render_prosecution_audit_tab()
    with tab5:
        render_defense_strategy_tab()

    st.markdown("---")
    st.caption(
        "Disclaimer: Digital Legislative Library content is generally unofficial online material. "
        "Always verify against official revised laws and current prosecutorial guidance."
    )


if __name__ == "__main__":
    main()
