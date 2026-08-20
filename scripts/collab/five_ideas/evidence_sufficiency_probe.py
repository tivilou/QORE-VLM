"""Observation-only evidence variants for Phase 9F."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Sequence


EVIDENCE_VARIANT_PROFILE = "gold_evidence_2x2_v1"
GOLD_ANSWER_COPY_PROFILE = "gold_answer_copy_v1"


@dataclass(frozen=True)
class EvidenceVariants:
    sentence_plain: tuple[str, ...]
    sentence_highlight: tuple[str, ...]
    window_plain: tuple[str, ...]
    window_highlight: tuple[str, ...]
    match_count: int


@dataclass(frozen=True)
class ProbeResult:
    prediction: str
    generation_time_ms: float


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char in ".?!\n":
            end = index + 1
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _highlight(text: str, start: int, end: int) -> str:
    return text[:start] + "[[" + text[start:end] + "]]" + text[end:]


def build_evidence_variants(
    passages: Sequence[str], gold_answers: Sequence[str]
) -> EvidenceVariants:
    """Build deterministic sentence/window variants from selected text only."""
    sentence_plain: list[str] = []
    sentence_highlight: list[str] = []
    window_plain: list[str] = []
    window_highlight: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    match_count = 0
    for passage in passages:
        text = str(passage)
        spans = _sentence_spans(text)
        lowered = text.lower()
        for answer in gold_answers:
            answer_norm = str(answer).lower().strip()
            if not answer_norm:
                continue
            pattern = re.compile(r"\b" + re.escape(answer_norm) + r"\b")
            for hit in pattern.finditer(lowered):
                sentence_index = next(
                    (idx for idx, (left, right) in enumerate(spans)
                     if left <= hit.start() < right),
                    None,
                )
                if sentence_index is None:
                    continue
                left, right = spans[sentence_index]
                sentence = text[left:right].strip()
                highlighted_sentence = _highlight(
                    text[left:right], hit.start() - left, hit.end() - left
                ).strip()
                window_left = spans[max(0, sentence_index - 1)][0]
                window_right = spans[min(len(spans) - 1, sentence_index + 1)][1]
                window = text[window_left:window_right].strip()
                highlighted_window = _highlight(
                    text[window_left:window_right],
                    hit.start() - window_left,
                    hit.end() - window_left,
                ).strip()
                key = (sentence, highlighted_sentence, window, highlighted_window)
                if key in seen:
                    continue
                seen.add(key)
                sentence_plain.append(sentence)
                sentence_highlight.append(highlighted_sentence)
                window_plain.append(window)
                window_highlight.append(highlighted_window)
                match_count += 1
    return EvidenceVariants(
        sentence_plain=tuple(sentence_plain),
        sentence_highlight=tuple(sentence_highlight),
        window_plain=tuple(window_plain),
        window_highlight=tuple(window_highlight),
        match_count=match_count,
    )


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


def build_gold_answer_copy_prompt(tokenizer: Any, question: str, gold_answer: str) -> str:
    system = (
        "Copy the supplied candidate answer exactly. Return only the candidate "
        "answer, with no explanation or label."
    )
    user = f"Question: {question}\nCandidate answer: {gold_answer}\n\nCopy the candidate answer:"
    return _chat_or_plain_prompt(tokenizer, system=system, user=user, fallback_prefix=system)


def generate_from_prompt(generator: Any, prompt: str) -> ProbeResult:
    import torch

    inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
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


def run_context_probe(generator: Any, question: str, passages: Sequence[str]) -> ProbeResult:
    """Call the stable Generator while varying only diagnostic context text."""
    started = time.perf_counter()
    forwarded = passages if isinstance(passages, list) else list(passages)
    prediction = generator.generate(question, forwarded)
    return ProbeResult(
        prediction=prediction,
        generation_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def run_gold_answer_copy_probe(generator: Any, question: str, gold_answer: str) -> ProbeResult:
    return generate_from_prompt(
        generator, build_gold_answer_copy_prompt(generator.tokenizer, question, gold_answer)
    )


__all__ = [
    "EVIDENCE_VARIANT_PROFILE",
    "GOLD_ANSWER_COPY_PROFILE",
    "EvidenceVariants",
    "ProbeResult",
    "build_evidence_variants",
    "build_gold_answer_copy_prompt",
    "generate_from_prompt",
    "run_context_probe",
    "run_gold_answer_copy_probe",
]
