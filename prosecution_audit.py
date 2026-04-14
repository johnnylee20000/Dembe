#!/usr/bin/env python3
"""
Prosecution-oriented auditing tools for TTPS operational/legal review.

Capabilities:
1) Prosecution Audit (elements-of-offence coverage)
2) Defense Anticipation (procedural loopholes + prosecution rebuttals)

Design principles:
- Prioritize admissibility and procedural legality.
- Distinguish standards (reasonable suspicion vs prima facie thresholds).
- Ground analysis with retrieved legal context where available.
"""

from __future__ import annotations

import argparse
import json
import re
from textwrap import dedent

from langchain_core.prompts import PromptTemplate

from assistant_core import (
    build_llm,
    format_source_documents,
    retrieve_documents,
)


PROSECUTION_AUDIT_PROMPT = PromptTemplate(
    template=dedent(
        """
        You are the TTPS Digital Prosecution Council.
        You must prioritize admissibility, legal sufficiency, and procedural integrity.
        Use ONLY the retrieved legal context and the officer statement.
        If uncertain, write REVIEW REQUIRED instead of guessing.

        Officer Statement / Notes:
        {officer_statement}

        Retrieved Legal Context:
        {context}

        Required output format:

        Charge Identification:
        - Act and Section: <value or REVIEW REQUIRED>
        - Offence Label: <value or REVIEW REQUIRED>

        Essential Elements (Points to Prove):
        1. <Element 1>
           - Evidence Found in Statement: <specific excerpt or MISSING>
           - Assessment: STRONG / WEAK / MISSING
        2. <Element 2>
           - Evidence Found in Statement: <specific excerpt or MISSING>
           - Assessment: STRONG / WEAK / MISSING
        3. <Element 3>
           - Evidence Found in Statement: <specific excerpt or MISSING>
           - Assessment: STRONG / WEAK / MISSING

        Mens Rea / Knowledge Analysis:
        - Finding: <present / weak / missing + why>

        Standard of Proof Guidance:
        - Reasonable Suspicion Issues: <if any>
        - Prima Facie Charge Sufficiency: <if any>

        High-Risk Gaps:
        - <bullet list of material deficiencies that could collapse prosecution>

        Recommended Additions for Officer Statement:
        - <concrete factual details to add>
        """
    ).strip(),
    input_variables=["officer_statement", "context"],
)


DEFENSE_ANTICIPATION_PROMPT = PromptTemplate(
    template=dedent(
        """
        You are acting as both:
        1) Defense Attorney (to identify procedural loopholes), and
        2) Prosecution Counsel (to craft lawful rebuttal).

        Prioritize procedures for:
        - Search
        - Arrest
        - Interview/Utterances/Caution

        Officer Statement / Procedure Notes:
        {officer_statement}

        Retrieved Legal/Standing Orders Context:
        {context}

        Output format:

        Defense Loophole Analysis:
        1. Procedure: <Search / Arrest / Interview / Evidence handling>
           - Potential Loophole: <description>
           - Risk Rating: LOW / MEDIUM / HIGH
           - Why Defense May Succeed: <reason>

        Procedural Integrity Flags:
        - Caution Timing: <compliant / unclear / non-compliant + reason>
        - Utterance Handling: <spontaneous / elicited / unclear + consequence>
        - Search Authority Basis: <warrant / warrantless exception / unclear>

        Prosecution Rebuttal Strategy:
        - Rebuttal 1: <legal argument tied to context>
        - Rebuttal 2: <legal argument tied to context>

        Reasonable Suspicion Justification Draft:
        - <short paragraph officer can adapt to justify grounds lawfully>

        Admissibility Verdict:
        - Overall Admissibility Risk: LOW / MEDIUM / HIGH
        - Immediate Corrective Actions: <specific actions>
        """
    ).strip(),
    input_variables=["officer_statement", "context"],
)


def _extract_context(source_docs, char_limit: int = 26000) -> str:
    return "\n\n".join(doc.page_content for doc in source_docs)[:char_limit]


def _fallback_prosecution_audit(statement: str, context: str) -> str:
    statement_lower = statement.lower()
    has_possession = any(word in statement_lower for word in ["possession", "found with", "had in his possession"])
    has_knowledge = any(word in statement_lower for word in ["knew", "knowingly", "admitted", "aware"])
    has_control = any(word in statement_lower for word in ["on his person", "waistband", "in his hand", "within reach"])

    mens_rea = "WEAK/MISSING" if not has_knowledge else "PRESENT"
    control = "PRESENT" if has_control else "MISSING"
    possession_assessment = "WEAK" if has_possession and not has_control else ("MISSING" if not has_possession else "STRONG")

    return dedent(
        f"""
        Charge Identification:
        - Act and Section: REVIEW REQUIRED
        - Offence Label: REVIEW REQUIRED

        Essential Elements (Points to Prove):
        1. Possession/Custody
           - Evidence Found in Statement: {"present" if has_possession else "MISSING"}
           - Assessment: {possession_assessment}
        2. Control over item
           - Evidence Found in Statement: {"present" if has_control else "MISSING"}
           - Assessment: {control}
        3. Knowledge / Mens Rea
           - Evidence Found in Statement: {"present" if has_knowledge else "MISSING"}
           - Assessment: {"STRONG" if has_knowledge else "MISSING"}

        Mens Rea / Knowledge Analysis:
        - Finding: {mens_rea}. Add facts proving awareness of nature/character of item.

        Standard of Proof Guidance:
        - Reasonable Suspicion Issues: REVIEW REQUIRED from full legal context.
        - Prima Facie Charge Sufficiency: potentially vulnerable where knowledge/control facts are thin.

        High-Risk Gaps:
        - Missing explicit facts proving knowledge.
        - Missing explicit chain of custody / control details.

        Recommended Additions for Officer Statement:
        - State exact location of item relative to accused and visibility/access.
        - Record any admissions, conduct, or circumstances indicating knowledge.
        - Record seizure continuity and witness/officer observations chronologically.

        Note: Retrieval-only fallback mode used (no LLM runtime available).
        Context excerpt:
        {context[:1200]}
        """
    ).strip()


def _fallback_defense_anticipation(statement: str, context: str) -> str:
    text = statement.lower()
    mentions_utterance = any(word in text for word in ["utterance", "said", "stated", "admitted", "confessed"])
    mentions_caution = "caution" in text
    mentions_warrant = "warrant" in text
    mentions_search = "search" in text

    caution_flag = (
        "non-compliant/unclear (utterance present with no caution timing)"
        if mentions_utterance and not mentions_caution
        else ("compliant/mentioned" if mentions_caution else "unclear")
    )
    search_basis = (
        "warrant"
        if mentions_warrant
        else ("warrantless exception or unclear" if mentions_search else "unclear")
    )
    risk = "HIGH" if mentions_utterance and not mentions_caution else ("MEDIUM" if mentions_search else "LOW")

    return dedent(
        f"""
        Defense Loophole Analysis:
        1. Procedure: Interview/Utterance
           - Potential Loophole: {"No immediate caution recorded after utterance." if mentions_utterance and not mentions_caution else "No obvious utterance-related defect identified from provided text."}
           - Risk Rating: {risk}
           - Why Defense May Succeed: Uncautioned elicited statements may be challenged for inadmissibility.

        Procedural Integrity Flags:
        - Caution Timing: {caution_flag}
        - Utterance Handling: {"spontaneous vs elicited unclear" if mentions_utterance else "no utterance detail provided"}
        - Search Authority Basis: {search_basis}

        Prosecution Rebuttal Strategy:
        - Rebuttal 1: Demonstrate lawful grounds and contemporaneous officer notes for each step.
        - Rebuttal 2: Distinguish spontaneous utterance from prompted questioning where supported by facts.

        Reasonable Suspicion Justification Draft:
        - Based on objective observations available at the scene, the officer formed reasonable suspicion
          and acted proportionately, documenting grounds, timing, and continuity of action.

        Admissibility Verdict:
        - Overall Admissibility Risk: {risk}
        - Immediate Corrective Actions: add caution timing, grounds for search/arrest, and full chronology.

        Note: Retrieval-only fallback mode used (no LLM runtime available).
        Context excerpt:
        {context[:1200]}
        """
    ).strip()


def run_prosecution_audit(
    officer_statement: str,
    *,
    vector_db_path: str = "/vector_db",
    llm_provider: str | None = None,
    model_name: str | None = None,
    embedding_provider: str | None = None,
    k: int = 10,
    retrieval_only: bool = False,
) -> dict:
    source_docs = retrieve_documents(
        query=(
            "Identify charge, Act/Section, essential offence elements, mens rea requirements, "
            f"and admissibility risks for: {officer_statement}"
        ),
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    context = _extract_context(source_docs)

    if retrieval_only:
        analysis = _fallback_prosecution_audit(officer_statement, context)
        return {
            "mode": "retrieval_only",
            "analysis": analysis,
            "sources": format_source_documents(source_docs),
        }

    llm = build_llm(llm_provider=llm_provider, model_name=model_name)
    prompt = PROSECUTION_AUDIT_PROMPT.format(
        officer_statement=officer_statement,
        context=context,
    )
    completion = llm.invoke(prompt)
    text = getattr(completion, "content", str(completion))
    return {
        "mode": "llm",
        "analysis": text,
        "sources": format_source_documents(source_docs),
    }


def run_defense_anticipation(
    officer_statement: str,
    *,
    vector_db_path: str = "/vector_db",
    llm_provider: str | None = None,
    model_name: str | None = None,
    embedding_provider: str | None = None,
    k: int = 10,
    retrieval_only: bool = False,
) -> dict:
    source_docs = retrieve_documents(
        query=(
            "Identify procedural rules for search, arrest, caution, utterances/interviews, "
            f"defense loopholes, and prosecution rebuttals for: {officer_statement}"
        ),
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    context = _extract_context(source_docs)

    if retrieval_only:
        analysis = _fallback_defense_anticipation(officer_statement, context)
        return {
            "mode": "retrieval_only",
            "analysis": analysis,
            "sources": format_source_documents(source_docs),
        }

    llm = build_llm(llm_provider=llm_provider, model_name=model_name)
    prompt = DEFENSE_ANTICIPATION_PROMPT.format(
        officer_statement=officer_statement,
        context=context,
    )
    completion = llm.invoke(prompt)
    text = getattr(completion, "content", str(completion))
    return {
        "mode": "llm",
        "analysis": text,
        "sources": format_source_documents(source_docs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prosecution/legal audit workflows.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prosecution_audit", "defense_anticipation"],
        help="Audit mode to run.",
    )
    parser.add_argument("--officer-statement", required=True, help="Officer statement/notes text.")
    parser.add_argument("--vector-db-path", default="/vector_db", help="Path to persisted Chroma DB.")
    parser.add_argument("--llm-provider", default=None, help="openai or ollama (optional).")
    parser.add_argument("--model-name", default=None, help="Model override (gpt-4o, llama3.1, etc.).")
    parser.add_argument("--embedding-provider", default=None, help="openai or huggingface (optional).")
    parser.add_argument("--k", type=int, default=10, help="Retriever top-k.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM and run deterministic retrieval-based audit scaffolding.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "prosecution_audit":
        payload = run_prosecution_audit(
            officer_statement=args.officer_statement,
            vector_db_path=args.vector_db_path,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            embedding_provider=args.embedding_provider,
            k=args.k,
            retrieval_only=args.retrieval_only,
        )
    else:
        payload = run_defense_anticipation(
            officer_statement=args.officer_statement,
            vector_db_path=args.vector_db_path,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            embedding_provider=args.embedding_provider,
            k=args.k,
            retrieval_only=args.retrieval_only,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
