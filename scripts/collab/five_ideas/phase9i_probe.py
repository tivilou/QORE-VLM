"""Observation-only candidate and verifier probes for Phase 9I."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Mapping, Sequence

try:
    from scripts.collab.five_ideas.phase9h_probe import (
        ProbeResult,
        build_extractive_prompt,
        generate_from_prompt,
    )
except ImportError:  # pragma: no cover
    from phase9h_probe import ProbeResult, build_extractive_prompt, generate_from_prompt


EXTRACTIVE_SPAN_PROFILE = "extractive_span_v1"
EVIDENCE_CONSTRAINED_PROFILE = "evidence_constrained_v1"
VERIFIER_PROFILE = "grounded_candidate_choice_v1"
ORDER_AUDIT_PROFILE = "fixed_permutation_v1"
ABSTENTION_TOKEN = "<ABSTAIN>"


@dataclass(frozen=True)
class CandidateProbeResult:
    mode: str
    text: str
    parse_status: str
    generation_time_ms: float


@dataclass(frozen=True)
class VerifierProbeResult:
    choice_mode: str | None
    choice_index: int | None
    parse_status: str
    generation_time_ms: float


def _chat_or_plain_prompt(
    tokenizer: Any, *, system: str, user: str, fallback_prefix: str
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{fallback_prefix}\n\n{user}\nAnswer:"


def _context(passages: Sequence[str]) -> str:
    return "\n\n".join(
        f"Passage {index + 1}: {passage}" for index, passage in enumerate(passages)
    )


def build_evidence_constrained_prompt(
    tokenizer: Any, question: str, passages: Sequence[str]
) -> str:
    system = (
        "Answer the question using only the supplied passages. Return the "
        "shortest directly supported answer, with no explanation or label. "
        f"If the passages do not directly support an answer, return exactly {ABSTENTION_TOKEN}."
    )
    user = (
        f"Context:\n{_context(passages)}\n\nQuestion: {question}\n\n"
        "Short supported answer:"
    )
    return _chat_or_plain_prompt(
        tokenizer, system=system, user=user, fallback_prefix=system
    )


def build_verifier_prompt(
    tokenizer: Any,
    question: str,
    passages: Sequence[str],
    candidates: Sequence[tuple[str, str]],
) -> str:
    system = (
        "Choose the candidate answer that is most directly supported by the "
        "provided context and answers the question. Return exactly one token "
        "from the candidate IDs shown below, such as CANDIDATE_0. Do not explain."
    )
    candidate_block = "\n".join(
        f"CANDIDATE_{index} ({mode}): {text}"
        for index, (mode, text) in enumerate(candidates)
    )
    user = (
        f"Context:\n{_context(passages)}\n\nQuestion: {question}\n\n"
        f"Candidates:\n{candidate_block}\n\nSelected candidate ID:"
    )
    return _chat_or_plain_prompt(
        tokenizer, system=system, user=user, fallback_prefix=system
    )


def normalize_candidate(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def candidate_parse_status(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return "empty"
    if normalize_candidate(stripped) == normalize_candidate(ABSTENTION_TOKEN):
        return "abstain"
    return "ok"


def fixed_candidate_permutation(
    candidates: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    return list(reversed(list(candidates)))


def parse_candidate_id(text: str, candidate_count: int) -> int | None:
    matches = re.findall(r"\bCANDIDATE[_ -]?(\d+)\b", str(text).upper())
    if len(matches) != 1:
        return None
    index = int(matches[0])
    if index < 0 or index >= candidate_count:
        return None
    return index


def run_evidence_constrained_probe(
    generator: Any, question: str, passages: Sequence[str]
) -> CandidateProbeResult:
    result: ProbeResult = generate_from_prompt(
        generator,
        build_evidence_constrained_prompt(generator.tokenizer, question, passages),
    )
    return CandidateProbeResult(
        mode=EVIDENCE_CONSTRAINED_PROFILE,
        text=result.prediction,
        parse_status=candidate_parse_status(result.prediction),
        generation_time_ms=result.generation_time_ms,
    )


def run_extractive_candidate_probe(
    generator: Any, question: str, passages: Sequence[str]
) -> CandidateProbeResult:
    result: ProbeResult = generate_from_prompt(
        generator,
        build_extractive_prompt(generator.tokenizer, question, passages),
    )
    return CandidateProbeResult(
        mode=EXTRACTIVE_SPAN_PROFILE,
        text=result.prediction,
        parse_status=candidate_parse_status(result.prediction),
        generation_time_ms=result.generation_time_ms,
    )


def run_verifier_probe(
    generator: Any,
    question: str,
    passages: Sequence[str],
    candidates: Sequence[tuple[str, str]],
) -> VerifierProbeResult:
    if not candidates:
        return VerifierProbeResult(None, None, "no_candidates", 0.0)
    result: ProbeResult = generate_from_prompt(
        generator,
        build_verifier_prompt(
            generator.tokenizer, question, passages, candidates
        ),
    )
    choice_index = parse_candidate_id(result.prediction, len(candidates))
    choice_mode = (
        candidates[choice_index][0] if choice_index is not None else None
    )
    return VerifierProbeResult(
        choice_mode=choice_mode,
        choice_index=choice_index,
        parse_status="ok" if choice_index is not None else "invalid_choice",
        generation_time_ms=result.generation_time_ms,
    )


def candidate_pairs(
    baseline_text: str,
    extractive: CandidateProbeResult,
    constrained: CandidateProbeResult,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if candidate_parse_status(baseline_text) in {"ok", "abstain"}:
        pairs.append(("baseline_v1", str(baseline_text)))
    for result in (extractive, constrained):
        if result.parse_status in {"ok", "abstain"} and result.text.strip():
            pairs.append((result.mode, result.text))
    return pairs


__all__ = [
    "ABSTENTION_TOKEN",
    "CandidateProbeResult",
    "EVIDENCE_CONSTRAINED_PROFILE",
    "EXTRACTIVE_SPAN_PROFILE",
    "ORDER_AUDIT_PROFILE",
    "VERIFIER_PROFILE",
    "VerifierProbeResult",
    "build_evidence_constrained_prompt",
    "build_verifier_prompt",
    "candidate_pairs",
    "candidate_parse_status",
    "fixed_candidate_permutation",
    "normalize_candidate",
    "parse_candidate_id",
    "run_evidence_constrained_probe",
    "run_extractive_candidate_probe",
    "run_verifier_probe",
]
