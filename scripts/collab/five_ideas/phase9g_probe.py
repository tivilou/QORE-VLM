"""Observation-only context variants for Phase 9G."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Sequence

CONTEXT_VARIANT_PROFILE = "full_context_3way_v1"
GOLD_ANSWER_COPY_PROFILE = "gold_answer_copy_v1"


@dataclass(frozen=True)
class ContextVariants:
    full_plain: tuple[str, ...]
    full_highlight: tuple[str, ...]
    answer_first: tuple[str, ...]
    retrieved_answer_oracle: tuple[str, ...]
    selected_match_count: int
    oracle_match_count: int
    answer_first_changed: bool


@dataclass(frozen=True)
class ProbeResult:
    prediction: str
    generation_time_ms: float


def _matches(text: str, answers: Sequence[str]) -> list[re.Match[str]]:
    lowered = text.lower()
    found: list[re.Match[str]] = []
    for answer in answers:
        value = str(answer).strip().lower()
        if value:
            found.extend(re.finditer(r"\b" + re.escape(value) + r"\b", lowered))
    return sorted(found, key=lambda match: (match.start(), match.end()))


def _highlight(text: str, answers: Sequence[str]) -> str:
    hits = _matches(text, answers)
    if not hits:
        return text
    pieces: list[str] = []
    cursor = 0
    for hit in hits:
        if hit.start() < cursor:
            continue
        pieces.extend((text[cursor:hit.start()], "[[", text[hit.start():hit.end()], "]]"))
        cursor = hit.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def build_context_variants(
    selected_passages: Sequence[str],
    retrieved_passages: Sequence[str],
    gold_answers: Sequence[str],
) -> ContextVariants:
    """Build deterministic variants using only already retrieved text."""
    selected = [str(value) for value in selected_passages]
    retrieved = [str(value) for value in retrieved_passages]
    selected_flags = [bool(_matches(value, gold_answers)) for value in selected]
    retrieved_flags = [bool(_matches(value, gold_answers)) for value in retrieved]
    answer_first = [value for value, flag in zip(selected, selected_flags) if flag]
    answer_first.extend(value for value, flag in zip(selected, selected_flags) if not flag)
    return ContextVariants(
        full_plain=tuple(value for value, flag in zip(selected, selected_flags) if flag),
        full_highlight=tuple(
            _highlight(value, gold_answers)
            for value, flag in zip(selected, selected_flags)
            if flag
        ),
        answer_first=tuple(answer_first),
        retrieved_answer_oracle=tuple(value for value, flag in zip(retrieved, retrieved_flags) if flag),
        selected_match_count=sum(selected_flags),
        oracle_match_count=sum(retrieved_flags),
        answer_first_changed=answer_first != selected,
    )


def _chat_or_plain_prompt(tokenizer: Any, *, system: str, user: str, fallback_prefix: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{fallback_prefix}\n\n{user}\nShort Answer:"


def build_gold_answer_copy_prompt(tokenizer: Any, question: str, gold_answer: str) -> str:
    system = "Copy the supplied candidate answer exactly. Return only the candidate answer, with no explanation or label."
    user = f"Question: {question}\nCandidate answer: {gold_answer}\n\nCopy the candidate answer:"
    return _chat_or_plain_prompt(tokenizer, system=system, user=user, fallback_prefix=system)


def generate_from_prompt(generator: Any, prompt: str) -> ProbeResult:
    import torch
    inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(generator.model.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        outputs = generator.model.generate(**inputs, max_new_tokens=generator.max_new_tokens, do_sample=False, pad_token_id=generator.tokenizer.eos_token_id)
    elapsed = (time.perf_counter() - started) * 1000.0
    input_length = inputs["input_ids"].shape[1]
    return ProbeResult(generator.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip(), elapsed)


def run_context_probe(generator: Any, question: str, passages: Sequence[str]) -> ProbeResult:
    started = time.perf_counter()
    forwarded = passages if isinstance(passages, list) else list(passages)
    return ProbeResult(generator.generate(question, forwarded), (time.perf_counter() - started) * 1000.0)


def run_gold_answer_copy_probe(generator: Any, question: str, gold_answer: str) -> ProbeResult:
    return generate_from_prompt(generator, build_gold_answer_copy_prompt(generator.tokenizer, question, gold_answer))


__all__ = ["CONTEXT_VARIANT_PROFILE", "GOLD_ANSWER_COPY_PROFILE", "ContextVariants", "ProbeResult", "build_context_variants", "build_gold_answer_copy_prompt", "generate_from_prompt", "run_context_probe", "run_gold_answer_copy_probe"]
