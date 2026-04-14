#!/usr/bin/env python3
"""
Automated validation suite for prosecution-focused assistant capabilities.

Checks:
1) Legal advisor retrieval for caution/search prompts.
2) Prosecution audit and defense anticipation deterministic behavior.
3) Cleanliness check for banned case-specific artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prosecution_audit import run_defense_anticipation, run_prosecution_audit
from assistant_core import retrieval_only_search


BANNED_TERMS = ("jaggan", "santa cruz", "draco", "allister badal", "badal")


def _contains_banned(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in BANNED_TERMS if term in lowered]


def validate_stack(vector_db_path: str) -> dict:
    results: dict[str, object] = {}

    # 1) Legal advisor retrieval test prompts.
    prompt_1 = "What is the standard caution I must give a suspect upon arrest in Trinidad and Tobago?"
    prompt_2 = "What are the requirements for a search warrant under the Summary Courts Act?"
    sources_1 = retrieval_only_search(prompt_1, vector_db_path=vector_db_path, k=8)
    sources_2 = retrieval_only_search(prompt_2, vector_db_path=vector_db_path, k=8)
    results["legal_advisor"] = {
        "prompt_1_sources": len(sources_1),
        "prompt_2_sources": len(sources_2),
        "prompt_1_preview": sources_1[0]["snippet"] if sources_1 else "",
        "prompt_2_preview": sources_2[0]["snippet"] if sources_2 else "",
    }

    # 2) Prosecution audit deterministic checks.
    prosecution_statement = (
        "I searched the accused and found one pistol in his waistband. "
        "He said it was not his. I arrested him."
    )
    audit = run_prosecution_audit(
        officer_statement=prosecution_statement,
        vector_db_path=vector_db_path,
        retrieval_only=True,
        k=10,
    )
    defense_statement = (
        "During transport the suspect said the gun was his. I cautioned him later at station. "
        "I searched him without warrant based on suspicious bulge."
    )
    defense = run_defense_anticipation(
        officer_statement=defense_statement,
        vector_db_path=vector_db_path,
        retrieval_only=True,
        k=10,
    )

    audit_text = str(audit.get("analysis", ""))
    defense_text = str(defense.get("analysis", ""))
    results["prosecution_audit"] = {
        "has_mens_rea_section": "Mens Rea / Knowledge Analysis" in audit_text,
        "has_high_risk_gaps": "High-Risk Gaps" in audit_text,
        "sources": len(audit.get("sources", [])),
    }
    results["defense_anticipation"] = {
        "has_admissibility_verdict": "Admissibility Verdict" in defense_text,
        "has_rebuttal_strategy": "Prosecution Rebuttal Strategy" in defense_text,
        "sources": len(defense.get("sources", [])),
    }

    # 3) Cleanliness scan across python and markdown files.
    banned_hits: dict[str, list[str]] = {}
    self_file = str(Path(__file__).resolve())
    for path in Path("/workspace").glob("*"):
        if path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        if str(path.resolve()) == self_file:
            # Avoid false positives from the validator's own banned-term list.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        hits = _contains_banned(text)
        if hits:
            banned_hits[str(path)] = hits
    results["cleanliness"] = {
        "banned_hits": banned_hits,
        "clean": not banned_hits,
    }

    # Overall pass/fail conditions.
    source1_acts = " ".join(
        (
            str(item.get("metadata", {}).get("act_name", ""))
            + " "
            + str(item.get("metadata", {}).get("header", ""))
        ).lower()
        for item in sources_1
    )
    source2_acts = " ".join(
        (
            str(item.get("metadata", {}).get("act_name", ""))
            + " "
            + str(item.get("metadata", {}).get("header", ""))
        ).lower()
        for item in sources_2
    )
    authority_hits = {
        "caution_prompt_has_police_or_judges": (
            ("police" in source1_acts)
            or ("judge" in source1_acts)
            or ("evidence" in source1_acts)
            or ("criminal" in source1_acts)
        ),
        "warrant_prompt_has_summary_or_procedure": (
            ("summary courts" in source2_acts)
            or ("summary" in source2_acts)
            or ("procedure" in source2_acts)
            or ("search" in source2_acts)
        ),
    }
    results["authority_checks"] = authority_hits

    pass_conditions = [
        bool(sources_1),
        bool(sources_2),
        results["prosecution_audit"]["has_mens_rea_section"],  # type: ignore[index]
        results["defense_anticipation"]["has_admissibility_verdict"],  # type: ignore[index]
        results["cleanliness"]["clean"],  # type: ignore[index]
        authority_hits["caution_prompt_has_police_or_judges"],
        authority_hits["warrant_prompt_has_summary_or_procedure"],
    ]
    results["overall_pass"] = all(pass_conditions)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate prosecution stack end-to-end.")
    parser.add_argument(
        "--vector-db-path",
        default="/workspace/vector_db",
        help="Path to vector DB used in validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = validate_stack(vector_db_path=args.vector_db_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary.get("overall_pass"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
