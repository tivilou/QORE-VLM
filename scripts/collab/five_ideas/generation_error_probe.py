"""Observation-only generation probes for the Phase 9E diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Sequence


EXTRACTIVE_PROMPT_PROFILE = "extractive_span_v1"
GOLD_ANSWER_COPY_PROFILE = "gold_answer_copy_v1"


@dataclass(frozen=True)
class ProbeResult:
    prediction: str
    generation_time_ms: float


def extract_gold_matched_sentences(
    passages: Sequence[str], gold_answers: Sequence[str]
) -> tuple[str, ...]:
    """Return deterministic sentence windows containing a token-boundary gold match."""
    matches: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        text = str(passage)
        lowered = text.lower()
        for answer in gold_answers:
            answer_norm = str(answer).lower().strip()
            if not answer_norm:
                continue
            pattern = re.compile(r"\b" + re.escape(answer_norm) + r"\b")
            for hit in pattern.finditer(lowered):
                left = max(
                    text.rfind(".", 0, hit.start()),
                    text.rfind("?", 0, hit.start()),
                    text.rfind("!", 0, hit.start()),
                    text.rfind("\n", 0, hit.start()),
                )
                right_candidates = [
                    position
                    for marker in (".", "?", "!", "\n")
                    if (position := text.find(marker, hit.end())) >= 0
                ]
                right = min(right_candidates) + 1 if right_candidates else len(text)
                sentence = text[left + 1:right].strip()
                if sentence and sentence not in seen:
                    seen.add(sentence)
                    matches.append(sentence)
    return tuple(matches)


def _chat_or_plain_prompt(
    tokenizer: Any,
    *,
    system: str,
    user: str,
    fallback_prefix: str,
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


def build_extractive_prompt(tokenizer: Any, question: str, passages: Sequence[str]) -> str:
    context = "\n\n".join(
        f"Passage {index + 1}: {passage}" for index, passage in enumerate(passages)
    )
    system = (
        "Extract the shortest answer span that directly answers the question. "
        "Copy the answer words verbatim from the supplied context. Return only "
        "that span, with no explanation, label, or surrounding sentence."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nExtract the exact answer span:"
    return _chat_or_plain_prompt(
        tokenizer,
        system=system,
        user=user,
        fallback_prefix=system,
    )


def build_gold_answer_copy_prompt(tokenizer: Any, question: str, gold_answer: str) -> str:
    system = (
        "Copy the supplied candidate answer exactly. Return only the candidate "
        "answer, with no explanation or label."
    )
    user = f"Question: {question}\nCandidate answer: {gold_answer}\n\nCopy the candidate answer:"
    return _chat_or_plain_prompt(
        tokenizer,
        system=system,
        user=user,
        fallback_prefix=system,
    )


def generate_from_prompt(generator: Any, prompt: str) -> ProbeResult:
    """Generate with the frozen model settings while varying only prompt text."""
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


def run_frozen_baseline_probe(
    generator: Any, question: str, passages: list[str]
) -> ProbeResult:
    """Call the stable Generator.generate entry point without altering inputs."""
    started = time.perf_counter()
    prediction = generator.generate(question, passages)
    return ProbeResult(
        prediction=prediction,
        generation_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def run_extractive_probe(
    generator: Any, question: str, passages: Sequence[str]
) -> ProbeResult:
    return generate_from_prompt(
        generator,
        build_extractive_prompt(generator.tokenizer, question, passages),
    )


def run_gold_answer_copy_probe(
    generator: Any, question: str, gold_answer: str
) -> ProbeResult:
    return generate_from_prompt(
        generator,
        build_gold_answer_copy_prompt(generator.tokenizer, question, gold_answer),
    )


__all__ = [
    "EXTRACTIVE_PROMPT_PROFILE",
    "GOLD_ANSWER_COPY_PROFILE",
    "ProbeResult",
    "build_extractive_prompt",
    "build_gold_answer_copy_prompt",
    "extract_gold_matched_sentences",
    "generate_from_prompt",
    "run_frozen_baseline_probe",
    "run_extractive_probe",
    "run_gold_answer_copy_probe",
]
