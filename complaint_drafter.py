#!/usr/bin/env python3
"""
Complaint drafter for TTPS workflow.

Implements a specialized function `draft_complaint` that:
- uses a one-shot complaint template inspired by State v. Allister Badal format
- retrieves legally grounded context (ICCS, Act, Section) from vector DB
- returns draft Statement of Offence + Particulars of Offence
"""

from __future__ import annotations

import argparse
import json
from textwrap import dedent

from langchain_core.prompts import PromptTemplate

from assistant_core import (
    build_llm,
    build_retrieval_qa_chain,
    format_source_documents,
    retrieve_documents,
)

BADAL_STYLE_EXAMPLE = dedent(
    """
    Example (style reference):
    Statement of Offence:
    Possession of firearm and ammunition.

    Particulars of Offence:
    John Doe, on the 14th day of March 2025 at St. James in the said district,
    had in his possession one firearm, to wit a .38 revolver, and five rounds of
    ammunition without being the holder of a Firearm User's Licence, contrary to
    section [X] of the Firearms Act, Chap. 16:01.
    """
).strip()


COMPLAINT_PROMPT = PromptTemplate(
    template=dedent(
        """
        You are drafting a TTPS Complaint on Oath.
        Use the legal context to identify the most suitable offence with:
        - ICCS code (if available in context)
        - Act name
        - Chapter
        - Section

        Never invent law references. If uncertain, say "REVIEW REQUIRED" and explain what is missing.

        Style example:
        {badal_example}

        Officer Notes:
        {officer_notes}

        Retrieved Legal Context:
        {context}

        Output format (strict):
        ICCS Code: <value or REVIEW REQUIRED>
        Act/Section: <Act, Chapter, Section or REVIEW REQUIRED>

        Statement of Offence:
        <formal one-sentence offence title>

        Particulars of Offence:
        <formal particulars paragraph suitable for charge drafting>
        """
    ).strip(),
    input_variables=["officer_notes", "context", "badal_example"],
)


def _fallback_template(officer_notes: str, context: str) -> str:
    return dedent(
        f"""
        ICCS Code: REVIEW REQUIRED
        Act/Section: REVIEW REQUIRED

        Statement of Offence:
        REVIEW REQUIRED — identify precise offence title from retrieved legal references.

        Particulars of Offence:
        Based on officer notes, the defendant was allegedly involved in the following incident:
        "{officer_notes}".
        Retrieved legal context was found, but no LLM is configured in this runtime. Use the cited
        source excerpts to finalize ICCS code, Act/Section, and formal particulars.

        Context excerpt:
        {context[:1400]}
        """
    ).strip()


def draft_complaint(
    officer_notes: str,
    *,
    vector_db_path: str = "/vector_db",
    llm_provider: str | None = None,
    model_name: str | None = None,
    embedding_provider: str | None = None,
    k: int = 8,
    retrieval_only: bool = False,
) -> dict:
    """
    Draft a Complaint on Oath section from officer notes.
    """
    source_docs = retrieve_documents(
        query=(
            "Identify the most appropriate offence, ICCS code, Act and section for: "
            f"{officer_notes}"
        ),
        vector_db_path=vector_db_path,
        embedding_provider=embedding_provider,
        k=k,
    )
    context = "\n\n".join(doc.page_content for doc in source_docs)[:22000]

    if retrieval_only:
        text = _fallback_template(officer_notes, context)
        return {
            "officer_notes": officer_notes,
            "draft": text,
            "sources": format_source_documents(source_docs),
            "mode": "retrieval_only",
        }

    # Build/reuse QA chain for validated LLM path + shared config.
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
        context=context,
        badal_example=BADAL_STYLE_EXAMPLE,
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
    parser = argparse.ArgumentParser(description="Draft TTPS complaint text from officer notes.")
    parser.add_argument("--officer-notes", required=True, help="Raw incident notes from officer.")
    parser.add_argument("--vector-db-path", default="/vector_db", help="Path to persisted Chroma DB.")
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider override (openai or ollama).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model override (e.g., gpt-4o or local ollama model name).",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        help="Embedding provider override (openai or huggingface).",
    )
    parser.add_argument("--k", type=int, default=8, help="Retrieved chunk count for grounding.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation and return retrieval-grounded drafting scaffold.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = draft_complaint(
        officer_notes=args.officer_notes,
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
