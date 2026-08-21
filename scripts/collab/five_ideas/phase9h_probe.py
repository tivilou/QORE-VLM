"""Observation-only answerability and generation probes for Phase 9H."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Sequence

EXTRACTIVE_PROMPT_PROFILE = "extractive_span_v1"
SUPPORT_JUDGE_PROFILE = "selected_context_support_judge_v1"
GOLD_ANSWER_COPY_PROFILE = "gold_answer_copy_v1"
SUPPORT_LABELS = ("supported", "unsupported", "uncertain")


@dataclass(frozen=True)
class ProbeResult:
    prediction: str
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
    return f"{fallback_prefix}\n\n{user}\nShort Answer:"


def _context(passages: Sequence[str]) -> str:
    return "\n\n".join(
        f"Passage {index + 1}: {passage}" for index, passage in enumerate(passages)
    )


def build_extractive_prompt(
    tokenizer: Any, question: str, passages: Sequence[str]
) -> str:
    system = (
        "Extract the shortest answer span that directly answers the question. "
        "Copy the answer words verbatim from the supplied context. Return only "
        "that span, with no explanation, label, or surrounding sentence."
    )
    user = (
        f"Context:\n{_context(passages)}\n\nQuestion: {question}\n\n"
        "Extract the exact answer span:"
    )
    return _chat_or_plain_prompt(
        tokenizer, system=system, user=user, fallback_prefix=system
    )


def build_support_judge_prompt(
    tokenizer: Any,
    question: str,
    passages: Sequence[str],
    gold_answer: str,
) -> str:
    system = (
        "You are an evidence verifier. Decide whether the candidate answer is "
        "directly supported by the passages for the question. Return exactly "
        "SUPPORTED, UNSUPPORTED, or UNCERTAIN. Do not explain."
    )
    user = (
        f"Context:\n{_context(passages)}\n\nQuestion: {question}\n"
        f"Candidate answer: {gold_answer}\n\nVerdict:"
    )
    return _chat_or_plain_prompt(
        tokenizer, system=system, user=user, fallback_prefix=system
    )


def build_gold_answer_copy_prompt(
    tokenizer: Any, question: str, gold_answer: str
) -> str:
    system = (
        "Copy the supplied candidate answer exactly. Return only the candidate "
        "answer, with no explanation or label."
    )
    user = (
        f"Question: {question}\nCandidate answer: {gold_answer}\n\n"
        "Copy the candidate answer:"
    )
    return _chat_or_plain_prompt(
        tokenizer, system=system, user=user, fallback_prefix=system
    )


def parse_support_label(prediction: str) -> str:
    """Parse only an explicit verdict; ambiguous generations become uncertain."""
    text = str(prediction).upper()
    unsupported = bool(re.search(r"\bUNSUPPORTED\b", text))
    supported = bool(re.search(r"\bSUPPORTED\b", text))
    uncertain = bool(re.search(r"\bUNCERTAIN\b", text))
    if unsupported and not supported and not uncertain:
        return "unsupported"
    if supported and not unsupported and not uncertain:
        return "supported"
    if uncertain and not supported and not unsupported:
        return "uncertain"
    return "uncertain"


def generate_from_prompt(generator: Any, prompt: str) -> ProbeResult:
    import torch

    inputs = generator.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096
    )
    inputs = {key: value.to(generator.model.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            max_new_tokens=generator.max_new_tokens,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    input_length = inputs["input_ids"].shape[1]
    answer = generator.tokenizer.decode(
        outputs[0][input_length:], skip_special_tokens=True
    ).strip()
    return ProbeResult(prediction=answer, generation_time_ms=elapsed)


def run_context_probe(
    generator: Any, question: str, passages: Sequence[str]
) -> ProbeResult:
    started = time.perf_counter()
    forwarded = passages if isinstance(passages, list) else list(passages)
    prediction = generator.generate(question, forwarded)
    return ProbeResult(
        prediction=prediction,
        generation_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def run_extractive_probe(
    generator: Any, question: str, passages: Sequence[str]
) -> ProbeResult:
    return generate_from_prompt(
        generator, build_extractive_prompt(generator.tokenizer, question, passages)
    )


def run_support_judge_probe(
    generator: Any,
    question: str,
    passages: Sequence[str],
    gold_answer: str,
) -> tuple[str, float]:
    result = generate_from_prompt(
        generator,
        build_support_judge_prompt(
            generator.tokenizer, question, passages, gold_answer
        ),
    )
    return parse_support_label(result.prediction), result.generation_time_ms


def run_gold_answer_copy_probe(
    generator: Any, question: str, gold_answer: str
) -> ProbeResult:
    return generate_from_prompt(
        generator,
        build_gold_answer_copy_prompt(generator.tokenizer, question, gold_answer),
    )


__all__ = [
    "EXTRACTIVE_PROMPT_PROFILE",
    "SUPPORT_JUDGE_PROFILE",
    "GOLD_ANSWER_COPY_PROFILE",
    "SUPPORT_LABELS",
    "ProbeResult",
    "build_extractive_prompt",
    "build_support_judge_prompt",
    "build_gold_answer_copy_prompt",
    "parse_support_label",
    "generate_from_prompt",
    "run_context_probe",
    "run_extractive_probe",
    "run_support_judge_probe",
    "run_gold_answer_copy_probe",
]
