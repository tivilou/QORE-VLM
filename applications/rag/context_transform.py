"""Observation-only context transformations for the Phase 9A diagnostic.

The transformer runs after passage selection and immediately before generation.
It never changes passage identity or order, and it has no path back to the
selector.  The implementation deliberately accepts a tokenizer-like object so
the diagnostic can use the generator tokenizer in production and a tiny fake
tokenizer in deterministic tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Sequence


CONTEXT_TRANSFORMS = ("full", "uniform_head", "reader_window")


class ContextTransformError(ValueError):
    """Raised when a context transformation request is malformed."""


@dataclass(frozen=True)
class ContextTransformResult:
    """Transformed payload plus passage-free diagnostics."""

    texts: tuple[str, ...]
    transform: str
    budget_ratio: float
    full_token_count: int
    transformed_token_count: int
    budget_tokens: int
    span_candidates: int
    span_found: int
    span_truncated: int
    fallback_count: int

    @property
    def reduction_ratio(self) -> float:
        if self.full_token_count <= 0:
            return 0.0
        return 1.0 - (self.transformed_token_count / self.full_token_count)

    def compact(self) -> dict[str, Any]:
        """Return only scalar diagnostics suitable for an exchange artifact."""
        return {
            "context_transform": self.transform,
            "context_budget_ratio": self.budget_ratio,
            "context_full_token_count": self.full_token_count,
            "context_transformed_token_count": self.transformed_token_count,
            "context_budget_tokens": self.budget_tokens,
            "context_reduction_ratio": self.reduction_ratio,
            "context_span_candidates": self.span_candidates,
            "context_span_found": self.span_found,
            "context_span_truncated": self.span_truncated,
            "context_fallback_count": self.fallback_count,
        }


def _as_ids(encoded: Any) -> list[int]:
    """Extract a flat list of token IDs from common tokenizer return types."""
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", [])
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if encoded is None:
        return []
    return [int(value) for value in encoded]


def _encode(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None:
        return text.split()
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    return _as_ids(encoded)


def _decode(tokenizer: Any, ids: Sequence[int], fallback: str) -> str:
    if tokenizer is None:
        return " ".join(str(value) for value in ids)
    try:
        text = tokenizer.decode(
            list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
    except (AttributeError, TypeError, ValueError):
        return fallback
    return str(text).strip()


def _window_token_ids(ids: Sequence[int], start: int, length: int) -> tuple[int, ...]:
    return tuple(ids[start : start + max(0, length)])


def _allocate_uniform(lengths: Sequence[int], budget: int) -> list[int]:
    """Allocate a fixed budget with per-passage quotas differing by at most one."""
    quotas = [0] * len(lengths)
    remaining = max(0, min(int(budget), sum(max(0, int(v)) for v in lengths)))
    while remaining:
        progressed = False
        for index, length in enumerate(lengths):
            if quotas[index] >= max(0, int(length)):
                continue
            quotas[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    return quotas


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    width = len(needle)
    target = tuple(needle)
    for start in range(len(haystack) - width + 1):
        if tuple(haystack[start : start + width]) == target:
            return start, start + width
    return None


def _hypothesis_text(hypotheses: Any) -> str:
    if not isinstance(hypotheses, Sequence) or isinstance(hypotheses, (str, bytes)):
        return ""
    for item in hypotheses:
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if text:
                return text
    return ""


def _centered_window(
    passage_ids: Sequence[int],
    quota: int,
    hypothesis: str,
    tokenizer: Any,
) -> tuple[tuple[int, ...], bool, bool]:
    """Return a centered window, plus (found, truncated) span flags."""
    if quota <= 0 or not passage_ids:
        return (), False, False
    target_ids = _encode(tokenizer, hypothesis)
    span = _find_subsequence(passage_ids, target_ids)
    if span is None and hypothesis:
        # Reader and generator tokenizers can disagree on leading whitespace.
        # Trying a prefixed form recovers the common BPE boundary case.
        span = _find_subsequence(passage_ids, _encode(tokenizer, " " + hypothesis))
    if span is None:
        return _window_token_ids(passage_ids, 0, quota), False, False

    span_start, span_end = span
    span_length = span_end - span_start
    truncated = span_length > quota
    if truncated:
        start = min(span_start, max(0, len(passage_ids) - quota))
    else:
        left = max(0, (quota - span_length) // 2)
        start = min(max(0, span_start - left), max(0, len(passage_ids) - quota))
    return _window_token_ids(passage_ids, start, quota), True, truncated


def transform_context(
    passages: Sequence[str],
    *,
    tokenizer: Any,
    transform: str = "full",
    budget_ratio: float = 1.0,
    hypotheses: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> ContextTransformResult:
    """Apply a deterministic, observation-only context transform.

    ``hypotheses`` is expected to be aligned with ``passages`` and contains the
    DPR reader's top answer spans.  Missing or unlocatable spans fall back to a
    uniform head window and are counted; they never change passage order.
    """
    if transform not in CONTEXT_TRANSFORMS:
        raise ContextTransformError(
            f"unknown transform {transform!r}; choose one of {CONTEXT_TRANSFORMS}"
        )
    if not math.isfinite(float(budget_ratio)) or not 0.0 < float(budget_ratio) <= 1.0:
        raise ContextTransformError("budget_ratio must be in (0, 1]")

    original = tuple(str(value) for value in passages)
    tokenized = [_encode(tokenizer, text) for text in original]
    lengths = [len(ids) for ids in tokenized]
    full_count = sum(lengths)
    if transform == "full":
        return ContextTransformResult(
            texts=original,
            transform=transform,
            budget_ratio=float(budget_ratio),
            full_token_count=full_count,
            transformed_token_count=full_count,
            budget_tokens=full_count,
            span_candidates=0,
            span_found=0,
            span_truncated=0,
            fallback_count=0,
        )

    nonempty = sum(length > 0 for length in lengths)
    budget = min(full_count, max(nonempty, int(math.floor(full_count * budget_ratio))))
    quotas = _allocate_uniform(lengths, budget)
    output_ids: list[tuple[int, ...]] = []
    span_candidates = span_found = span_truncated = fallback_count = 0
    for index, ids in enumerate(tokenized):
        quota = quotas[index]
        if transform == "reader_window":
            hypothesis = _hypothesis_text(hypotheses[index] if hypotheses and index < len(hypotheses) else ())
            if hypothesis:
                span_candidates += 1
                window, found, truncated = _centered_window(ids, quota, hypothesis, tokenizer)
                span_found += int(found)
                span_truncated += int(truncated)
                if not found:
                    fallback_count += 1
            else:
                window, found, truncated = _window_token_ids(ids, 0, quota), False, False
                fallback_count += 1
        else:
            window = _window_token_ids(ids, 0, quota)
        output_ids.append(window)

    texts = tuple(_decode(tokenizer, ids, original[index]) for index, ids in enumerate(output_ids))
    transformed_count = sum(len(_encode(tokenizer, text)) for text in texts)
    return ContextTransformResult(
        texts=texts,
        transform=transform,
        budget_ratio=float(budget_ratio),
        full_token_count=full_count,
        transformed_token_count=transformed_count,
        budget_tokens=budget,
        span_candidates=span_candidates,
        span_found=span_found,
        span_truncated=span_truncated,
        fallback_count=fallback_count,
    )


__all__ = [
    "CONTEXT_TRANSFORMS",
    "ContextTransformError",
    "ContextTransformResult",
    "transform_context",
]
