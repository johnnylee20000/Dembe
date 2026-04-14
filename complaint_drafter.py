#!/usr/bin/env python3
"""
Complaint drafter for TTPS workflow.

Implements a generalized `draft_complaint` function using a structural framework:
- Header: Court jurisdiction
- Complainant/Accused block
- Offence block: ICCS, Statement, Particulars
- Legal authority closing line:
  "Contrary to [Section] of the [Act Name] Chapter [Number]"
- Evidence summary placeholder ("Appendix A") from chronological notes
"""

from __future__ import annotations

import argparse
import json
from textwrap import dedent

from langchain_core.prompts import PromptTemplate

from assistant_core import (
    build_retrieval_qa_chain,
    format_source_documents,
    retrieve_documents,
)


COMPLAINT_PROMPT = PromptTemplate(
    template=dedent(
        """
        You are the TTPS Operational Assistant.
        Draft a Complaint on Oath for Trinidad and Tobago using only the retrieved legal context.
        Do not use any case-specific names or facts unless they are in the officer notes.
        Never invent legal citations. If missing, mark REVIEW REQUIRED.

        Officer Notes:
        {officer_notes}

        Court Jurisdiction:
        {court_jurisdiction}

        Retrieved Legal Context:
        {context}

        Produce output in this strict structure:

        Complaint on Oath
        Court: <Court Jurisdiction>

        Complainant:
        Rank: <REVIEW REQUIRED if unknown>
        Regimental Number: <REVIEW REQUIRED if unknown>
        Station: <REVIEW REQUIRED if unknown>

        Accused:
        Name: <REVIEW REQUIRED if unknown>
        Address: <REVIEW REQUIRED if unknown>
        Other Identifiers: <REVIEW REQUIRED if unknown>

        Offence Block:
        ICCS Code: <value or REVIEW REQUIRED>
        Statement of Offence: <formal statement>
        Particulars of Offence: <formal particulars paragraph based on notes>

        Legal Authority:
        Contrary to [Section] of the [Act Name] Chapter [Number].

        Evidence Summary (Appendix A):
        <chronologically ordered bullet points derived from the officer notes>

        Safety/Procedure Reminder:
        <include cautioning protocol reminder and officer safety reminder>
        """
    ).strip(),
    input_variables=["officer_notes", "court_jurisdiction", "context"],
)


def _ordered_notes_block(officer_notes: str) -> str:
    pieces = [p.strip() for p in officer_notes.replace("\n", " ").split(".") if p.strip()]
    if not pieces:
        pieces = [officer_notes.strip()] if officer_notes.strip() else ["REVIEW REQUIRED"]
    lines = []
    for i, piece in enumerate(pieces, start=1):
        lines.append(f"- Step {i}: {piece}.")
    return "\n".join(lines)


def _fallback_structural_template(officer_notes: str, court_jurisdiction: str, context: str) -> str:
    return dedent(
        f"""
        Complaint on Oath
        Court: {court_jurisdiction}

        Complainant:
        Rank: REVIEW REQUIRED
        Regimental Number: REVIEW REQUIRED
        Station: REVIEW REQUIRED

        Accused:
        Name: REVIEW REQUIRED
        Address: REVIEW REQUIRED
        Other Identifiers: REVIEW REQUIRED

        Offence Block:
        ICCS Code: REVIEW REQUIRED
        Statement of Offence: REVIEW REQUIRED
        Particulars of Offence: Based on officer notes: "{officer_notes}".

        Legal Authority:
        Contrary to [Section] of the [Act Name] Chapter [Number].

        Evidence Summary (Appendix A):
        {_ordered_notes_block(officer_notes)}

        Safety/Procedure Reminder:
        Ensure the suspect is cautioned in accordance with applicable TT law and procedure, and
        maintain officer safety controls during arrest/search.

        Context excerpt:
        {context[:1400]}
        """
    ).strip()


def draft_complaint(
    officer_notes: str,
    *,
    court_jurisdiction: str = "REVIEW REQUIRED",
    vector_db_path: str = "/vector_db",
    llm_provider: str | None = None,
    model_name: str | None = None,
    embedding_provider: str | None = None,
    k: int = 8,
    retrieval_only: bool = False,
) -> dict:
    """
    Draft a generalized Complaint on Oath from officer notes.
    """
    source_docs = retrieve_documents(
        query=(
            "Identify most relevant ICCS, Act, chapter, section, statement of offence and "
            f"particulars guidance for: {officer_notes}"
        ),
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    context = "\n\n".join(doc.page_content for doc in source_docs)[:22000]

    if retrieval_only:
        return {
            "officer_notes": officer_notes,
            "draft": _fallback_structural_template(officer_notes, court_jurisdiction, context),
            "sources": format_source_documents(source_docs),
            "mode": "retrieval_only",
        }

    qa_chain = build_retrieval_qa_chain(
        vector_db_path=vector_db_path,
        llm_provider=llm_provider,
        model_name=model_name,
        embedding_provider=embedding_provider,
        k=k,
    )
    llm = qa_chain.combine_documents_chain.llm_chain.llm
    prompt = COMPLAINT_PROMPT.format(
        officer_notes=officer_notes,
        court_jurisdiction=court_jurisdiction,
        context=context,
    )
    completion = llm.invoke(prompt)
    text = getattr(completion, "content", str(completion))
    return {
        "officer_notes": officer_notes,
        "draft": text,
        "sources": format_source_documents(source_docs),
        "mode": "llm",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft TTPS Complaint on Oath from officer notes.")
    parser.add_argument("--officer-notes", required=True, help="Raw incident notes from officer.")
    parser.add_argument(
        "--court-jurisdiction",
        default="REVIEW REQUIRED",
        help="Court jurisdiction label (e.g., North, South, Tobago).",
    )
    parser.add_argument("--vector-db-path", default="/vector_db", help="Path to persisted Chroma DB.")
    parser.add_argument("--llm-provider", default=None, help="LLM provider override (openai or ollama).")
    parser.add_argument("--model-name", default=None, help="Model override (e.g., gpt-4o).")
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Embedding provider override (openai or huggingface).",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieved chunk count for grounding.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation and return retrieval-grounded structural scaffold.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = draft_complaint(
        officer_notes=args.officer_notes,
        court_jurisdiction=args.court_jurisdiction,
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
