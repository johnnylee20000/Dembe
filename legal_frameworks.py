#!/usr/bin/env python3
"""
Deterministic legal/procedural frameworks for prosecution-oriented auditing.

These frameworks are used as a fallback when no LLM runtime is available and
to stabilize analysis quality for core offence/procedure checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OffenceElement:
    name: str
    evidence_keywords: Sequence[str]
    importance: str


@dataclass(frozen=True)
class OffenceFramework:
    offence_key: str
    offence_label: str
    act_name: str
    chapter: str
    likely_sections: Sequence[str]
    trigger_keywords: Sequence[str]
    elements: Sequence[OffenceElement]


OFFENCE_FRAMEWORKS: tuple[OffenceFramework, ...] = (
    OffenceFramework(
        offence_key="firearm_possession",
        offence_label="Possession of Firearm and/or Ammunition Without Lawful Authority",
        act_name="Firearms Act",
        chapter="16:01",
        likely_sections=("5", "6", "7", "22"),
        trigger_keywords=(
            "firearm",
            "gun",
            "pistol",
            "rifle",
            "shotgun",
            "ammunition",
            "cartridge",
            "f.u.l",
            "licence",
            "license",
        ),
        elements=(
            OffenceElement(
                name="Possession/Custody of the firearm or ammunition",
                evidence_keywords=(
                    "in his possession",
                    "found with",
                    "on his person",
                    "waistband",
                    "in his hand",
                    "in his bag",
                    "in his vehicle",
                    "recovered from",
                ),
                importance="critical",
            ),
            OffenceElement(
                name="Knowledge and control over the item (mens rea)",
                evidence_keywords=(
                    "admitted",
                    "said it was his",
                    "aware",
                    "knew",
                    "attempted to conceal",
                    "threw",
                    "hid",
                    "fled",
                ),
                importance="critical",
            ),
            OffenceElement(
                name="Absence of lawful authority/licence",
                evidence_keywords=(
                    "no licence",
                    "no license",
                    "unable to produce",
                    "not holder",
                    "without lawful authority",
                    "failed to produce",
                ),
                importance="critical",
            ),
        ),
    ),
    OffenceFramework(
        offence_key="dangerous_drug_possession",
        offence_label="Possession of Dangerous Drug",
        act_name="Dangerous Drugs Act",
        chapter="11:25",
        likely_sections=("4", "5", "6"),
        trigger_keywords=(
            "dangerous drug",
            "cannabis",
            "ganja",
            "cocaine",
            "heroin",
            "narcotic",
            "drug",
        ),
        elements=(
            OffenceElement(
                name="Custody or control of dangerous drug",
                evidence_keywords=(
                    "found with",
                    "in his possession",
                    "on his person",
                    "in his bag",
                    "in vehicle",
                    "recovered from",
                ),
                importance="critical",
            ),
            OffenceElement(
                name="Knowledge of nature of substance",
                evidence_keywords=(
                    "admitted",
                    "knew",
                    "aware",
                    "attempted to hide",
                    "attempted to discard",
                    "utered",
                    "uttered",
                ),
                importance="critical",
            ),
            OffenceElement(
                name="Nature/identity of substance supported",
                evidence_keywords=(
                    "field test",
                    "analyst",
                    "forensic",
                    "exhibit",
                    "sealed",
                    "labeled",
                    "certificate",
                ),
                importance="high",
            ),
        ),
    ),
)


PROCEDURAL_RISK_GUIDES: dict[str, tuple[str, ...]] = {
    "caution_protocol": (
        "Check whether caution was administered before or immediately after any elicited statement.",
        "Differentiate spontaneous utterance from responses to police questioning.",
    ),
    "search_authority": (
        "Identify whether search was warrant-based or justified by lawful warrantless grounds.",
        "Record objective facts forming reasonable suspicion at time of search.",
    ),
    "arrest_integrity": (
        "Document grounds for arrest at time of intervention.",
        "Maintain chronology from stop/search/arrest to exhibit handling.",
    ),
    "interview_integrity": (
        "Document location/timing of utterance and who initiated conversation.",
        "Record caution timing and witness presence where possible.",
    ),
}

