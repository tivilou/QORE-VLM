"""Explicit boundary for the Generator-native evidence-anchor experiment.

This module is deliberately separate from the production Generator.  It owns
only prompt-span provenance, token geometry, a scoped residual hook, and the
disabled fallback policy.  Retrieval, selection, generation, evaluation, and
task labels stay outside this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


CANDIDATE_ID = "generator_native_evidence_anchor_transport"
PLUGIN_VERSION = "0.3.0"
INTERVENTION_LAYER = 30
ARMS = ("disabled", "reader", "control")
ACTIVE_ARMS = ("reader", "control")
PLUGIN_ORDER = (
    "prompt_span_provenance_adapter",
    "pre_final_anchor_capture_adapter",
    "disabled_anchor_transport",
    "reader_localized_anchor_transport",
    "geometry_matched_nonreader_transport",
)
ARM_PLUGIN_IDS = {
    "disabled": "disabled_anchor_transport",
    "reader": "reader_localized_anchor_transport",
    "control": "geometry_matched_nonreader_transport",
}
FALLBACK_POLICY_ID = "per_question_disabled_zero_anchor"
FORBIDDEN_KEYS = {
    "question",
    "questions",
    "passage",
    "passages",
    "answer",
    "answers",
    "prediction",
    "predictions",
    "gold",
    "gold_answer",
    "raw_prompt",
    "prompt_text",
    "evaluator_trace",
}


class BoundaryError(ValueError):
    """A deterministic contract failure with a compact, non-content code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(message or self.code)


CharSpan = tuple[int, int]
TokenRows = tuple[tuple[int, ...], ...]
OffsetRows = tuple[tuple[CharSpan, ...], ...]


@dataclass(frozen=True)
class ReaderSpanProvenance:
    """Gold-free reader spans in passage-local character coordinates."""

    passage_spans: tuple[tuple[CharSpan, ...], ...]
    hypothesis_count: int
    located_count: int
    rejected_count: int
    provider_status: str

    def __post_init__(self) -> None:
        if self.provider_status not in {"ok", "empty", "provider_error"}:
            raise BoundaryError("reader_provider_status_invalid")
        if min(self.hypothesis_count, self.located_count, self.rejected_count) < 0:
            raise BoundaryError("reader_provider_count_invalid")
        if self.located_count + self.rejected_count != self.hypothesis_count:
            raise BoundaryError("reader_provider_accounting_invalid")

    def compact(self) -> dict[str, Any]:
        return {
            "provider_id": "frozen_dpr_reader_hypotheses_v1",
            "provider_status": self.provider_status,
            "passage_count": len(self.passage_spans),
            "hypothesis_count": self.hypothesis_count,
            "located_count": self.located_count,
            "rejected_count": self.rejected_count,
            "span_count": sum(len(row) for row in self.passage_spans),
            "gold_used": False,
            "answer_labels_used": False,
            "evaluator_used": False,
            "generation_output_used": False,
        }


@dataclass(frozen=True)
class GenerationObservation:
    """One wrapped generation plus a content-free diagnostic projection."""

    text: str
    decision: ArmDecision
    provider: ReaderSpanProvenance
    mapping: PromptTokenSpanMap | None
    hook_call_count: int
    hook_prefill_seen: bool
    boundary_status: str

    def compact(self) -> dict[str, Any]:
        payload = {
            "candidate_id": CANDIDATE_ID,
            "plugin_version": PLUGIN_VERSION,
            "boundary_status": self.boundary_status,
            "decision": compact_decisions((self.decision,))[0],
            "provider": self.provider.compact(),
            "mapping": self.mapping.compact() if self.mapping is not None else None,
            "hook_call_count": int(self.hook_call_count),
            "hook_prefill_seen": bool(self.hook_prefill_seen),
            "raw_content_persisted": False,
        }
        if find_forbidden_fields(payload):
            raise BoundaryError("compact_privacy_violation")
        return payload


@dataclass(frozen=True)
class ArmDecision:
    """The arm used for one question, including an explicit fallback reason."""

    requested_arm: str
    effective_arm: str
    fallback: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.requested_arm not in ARMS or self.effective_arm not in ARMS:
            raise BoundaryError("unknown_arm")
        if self.fallback and self.effective_arm != "disabled":
            raise BoundaryError("fallback_must_disable")
        if self.fallback and not self.reason:
            raise BoundaryError("fallback_reason_missing")


@dataclass(frozen=True)
class PromptTokenSpanMap:
    """Batch token geometry; prompt text and token contents are never stored."""

    input_ids_hash: str
    attention_mask_hash: str
    offset_mapping_hash: str
    special_tokens_mask_hash: str
    prompt_hashes: tuple[str, ...]
    attention_mask: TokenRows
    effective_offsets: OffsetRows
    reader_tokens: TokenRows
    control_tokens: TokenRows
    lexical_tokens: TokenRows
    reader_lengths: tuple[tuple[int, ...], ...]
    control_lengths: tuple[tuple[int, ...], ...]
    fallback_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.prompt_hashes)
        fields = (
            self.attention_mask,
            self.effective_offsets,
            self.reader_tokens,
            self.control_tokens,
            self.lexical_tokens,
            self.reader_lengths,
            self.control_lengths,
            self.fallback_reasons,
        )
        if any(len(field) != batch_size for field in fields):
            raise BoundaryError("batch_field_length_mismatch")
        if not self.attention_mask:
            raise BoundaryError("empty_batch")
        sequence_length = len(self.attention_mask[0])
        if sequence_length < 1 or any(len(row) != sequence_length for row in self.attention_mask):
            raise BoundaryError("attention_mask_shape_invalid")
        if any(len(row) != sequence_length for row in self.effective_offsets):
            raise BoundaryError("offset_shape_invalid")

    @property
    def batch_size(self) -> int:
        return len(self.prompt_hashes)

    @property
    def sequence_length(self) -> int:
        return len(self.attention_mask[0])

    @property
    def geometry_match(self) -> bool:
        return all(reader == control for reader, control in zip(self.reader_lengths, self.control_lengths))

    @property
    def reader_control_disjoint(self) -> bool:
        return all(
            not set(reader).intersection(control)
            for reader, control in zip(self.reader_tokens, self.control_tokens)
        )

    def compact(self) -> dict[str, Any]:
        """Return a passage/question-free summary suitable for a manifest."""
        padding_sides = []
        for mask in self.attention_mask:
            active = [index for index, value in enumerate(mask) if value]
            first, last = active[0], active[-1]
            if first and last < len(mask) - 1:
                side = "both"
            elif first:
                side = "left"
            elif last < len(mask) - 1:
                side = "right"
            else:
                side = "none"
            padding_sides.append(side)
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "input_ids_hash": self.input_ids_hash,
            "attention_mask_hash": self.attention_mask_hash,
            "offset_mapping_hash": self.offset_mapping_hash,
            "special_tokens_mask_hash": self.special_tokens_mask_hash,
            "prompt_hashes": list(self.prompt_hashes),
            "reader_token_counts": [len(row) for row in self.reader_tokens],
            "control_token_counts": [len(row) for row in self.control_tokens],
            "reader_token_geometry": [list(row) for row in self.reader_lengths],
            "control_token_geometry": [list(row) for row in self.control_lengths],
            "lexical_token_counts": [len(row) for row in self.lexical_tokens],
            "reader_control_disjoint": self.reader_control_disjoint,
            "geometry_match": self.geometry_match,
            "padding_sides": padding_sides,
            "fallback_count": sum(reason is not None for reason in self.fallback_reasons),
            "fallback_reasons": [reason for reason in self.fallback_reasons if reason is not None],
        }


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_to_python(item) for item in value]
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(_to_python(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def _matrix(value: Any, name: str) -> list[list[Any]]:
    rows = _to_python(value)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
        raise BoundaryError(f"{name}_shape_invalid")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise BoundaryError(f"{name}_shape_invalid")
    return rows


def _offset_matrix(value: Any) -> list[list[list[int]]]:
    rows = _matrix(value, "offset_mapping")
    normalized: list[list[list[int]]] = []
    for row in rows:
        converted: list[list[int]] = []
        for pair in row:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise BoundaryError("offset_mapping_shape_invalid")
            try:
                converted.append([int(pair[0]), int(pair[1])])
            except (TypeError, ValueError) as exc:
                raise BoundaryError("offset_mapping_value_invalid") from exc
        normalized.append(converted)
    return normalized


def _binary_rows(value: Any, name: str, *, batch_size: int, width: int) -> TokenRows:
    rows = _matrix(value, name)
    if len(rows) != batch_size or any(len(row) != width for row in rows):
        raise BoundaryError(f"{name}_shape_invalid")
    normalized: list[tuple[int, ...]] = []
    for row in rows:
        converted = tuple(int(item) for item in row)
        if name in {"attention_mask", "special_tokens_mask"} and any(item not in (0, 1) for item in converted):
            raise BoundaryError(f"{name}_not_binary")
        normalized.append(converted)
    return tuple(normalized)


def _effective_offset_rows(
    offsets: list[list[list[int]]],
    special_tokens: TokenRows,
    attention_mask: TokenRows,
    prompts: Sequence[str],
) -> OffsetRows:
    rows: list[tuple[CharSpan, ...]] = []
    for row_index, row in enumerate(offsets):
        starts = [int(pair[0]) for pair in row]
        effective: list[CharSpan] = []
        for token_index, pair in enumerate(row):
            start, end = int(pair[0]), int(pair[1])
            if not attention_mask[row_index][token_index] or special_tokens[row_index][token_index]:
                effective.append((0, 0))
                continue
            if start < 0 or end < 0 or start > len(prompts[row_index]):
                raise BoundaryError("offset_out_of_prompt")
            if end <= start:
                next_starts = [
                    starts[next_index]
                    for next_index in range(token_index + 1, len(starts))
                    if attention_mask[row_index][next_index]
                    and not special_tokens[row_index][next_index]
                    and starts[next_index] > start
                ]
                end = next_starts[0] if next_starts else len(prompts[row_index])
            if not 0 <= start < end <= len(prompts[row_index]):
                raise BoundaryError("effective_offset_invalid")
            effective.append((start, end))
        rows.append(tuple(effective))
    return tuple(rows)


def _normalize_spans(spans: Sequence[Any], prompt_length: int) -> tuple[CharSpan, ...]:
    normalized: list[CharSpan] = []
    for span in spans:
        if isinstance(span, Mapping):
            start, end = span.get("start"), span.get("end")
        elif isinstance(span, (list, tuple)) and len(span) == 2:
            start, end = span
        else:
            raise BoundaryError("reader_span_shape_invalid")
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError) as exc:
            raise BoundaryError("reader_span_value_invalid") from exc
        if not 0 <= start < end <= prompt_length:
            raise BoundaryError("reader_span_out_of_prompt")
        normalized.append((start, end))
    if not normalized:
        raise BoundaryError("reader_span_empty")
    return tuple(normalized)


def _map_row_spans(offsets: Sequence[CharSpan], spans: Sequence[CharSpan]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reader: list[int] = []
    lengths: list[int] = []
    for start, end in spans:
        mapped = tuple(
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start and token_start < end and token_end > start
        )
        if not mapped:
            raise BoundaryError("reader_span_unmappable")
        if set(reader).intersection(mapped):
            raise BoundaryError("reader_span_overlap")
        reader.extend(mapped)
        lengths.append(len(mapped))
    if not reader:
        raise BoundaryError("reader_span_unmappable")
    return tuple(reader), tuple(lengths)


def _choose_control_tokens(
    lexical: Sequence[int],
    reader: Sequence[int],
    reader_lengths: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    total = sum(int(length) for length in reader_lengths)
    if total < 1 or total != len(reader):
        raise BoundaryError("reader_geometry_invalid")
    reader_set = set(reader)
    lexical = tuple(int(index) for index in lexical)
    for start in range(0, len(lexical) - total + 1):
        candidate = tuple(lexical[start : start + total])
        if not reader_set.intersection(candidate):
            return candidate, tuple(int(length) for length in reader_lengths)
    raise BoundaryError("control_geometry_unavailable")


def _prepare_token_batch(tokenized: Mapping[str, Any], prompts: Sequence[str]) -> tuple[list[list[int]], TokenRows, list[list[list[int]]], TokenRows, OffsetRows]:
    if not isinstance(tokenized, Mapping):
        raise BoundaryError("tokenized_input_not_mapping")
    for key in ("input_ids", "attention_mask", "offset_mapping", "special_tokens_mask"):
        if key not in tokenized:
            raise BoundaryError(f"missing_{key}")
    input_ids = _matrix(tokenized["input_ids"], "input_ids")
    batch_size, width = len(input_ids), len(input_ids[0])
    if len(prompts) != batch_size:
        raise BoundaryError("prompt_batch_size_mismatch")
    attention_mask = _binary_rows(tokenized["attention_mask"], "attention_mask", batch_size=batch_size, width=width)
    special_tokens = _binary_rows(tokenized["special_tokens_mask"], "special_tokens_mask", batch_size=batch_size, width=width)
    offsets = _offset_matrix(tokenized["offset_mapping"])
    if len(offsets) != batch_size or any(len(row) != width for row in offsets):
        raise BoundaryError("offset_mapping_shape_invalid")
    effective_offsets = _effective_offset_rows(offsets, special_tokens, attention_mask, prompts)
    for row in attention_mask:
        active = [index for index, value in enumerate(row) if value]
        if not active:
            raise BoundaryError("attention_mask_empty_row")
        first, last = active[0], active[-1]
        if any(value == 0 for value in row[first : last + 1]):
            raise BoundaryError("attention_mask_noncontiguous")
    return input_ids, attention_mask, offsets, special_tokens, effective_offsets


def _build_row_geometry(
    offsets: Sequence[CharSpan],
    special_tokens: Sequence[int],
    attention_mask: Sequence[int],
    prompt: str,
    reader_spans: Sequence[Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    lexical = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if attention_mask[index] and not special_tokens[index] and end > start
    )
    if not lexical:
        raise BoundaryError("lexical_tokens_empty")
    normalized_spans = _normalize_spans(reader_spans, len(prompt))
    reader, reader_lengths = _map_row_spans(offsets, normalized_spans)
    control, control_lengths = _choose_control_tokens(lexical, reader, reader_lengths)
    return reader, control, lexical, reader_lengths, control_lengths


def build_prompt_token_span_map(
    tokenized: Mapping[str, Any],
    prompts: Sequence[str],
    reader_spans: Sequence[Sequence[Any]],
) -> PromptTokenSpanMap:
    """Build a strict batch map; any invalid row raises ``BoundaryError``."""
    input_ids, attention_mask, raw_offsets, special_tokens, effective = _prepare_token_batch(tokenized, prompts)
    if len(reader_spans) != len(prompts):
        raise BoundaryError("reader_span_batch_size_mismatch")
    reader_rows: list[tuple[int, ...]] = []
    control_rows: list[tuple[int, ...]] = []
    lexical_rows: list[tuple[int, ...]] = []
    reader_lengths: list[tuple[int, ...]] = []
    control_lengths: list[tuple[int, ...]] = []
    for row_index, prompt in enumerate(prompts):
        reader, control, lexical, lengths, control_geometry = _build_row_geometry(
            effective[row_index],
            special_tokens[row_index],
            attention_mask[row_index],
            str(prompt),
            reader_spans[row_index],
        )
        reader_rows.append(reader)
        control_rows.append(control)
        lexical_rows.append(lexical)
        reader_lengths.append(lengths)
        control_lengths.append(control_geometry)
    return PromptTokenSpanMap(
        input_ids_hash=_stable_hash(input_ids),
        attention_mask_hash=_stable_hash(attention_mask),
        offset_mapping_hash=_stable_hash(raw_offsets),
        special_tokens_mask_hash=_stable_hash(special_tokens),
        prompt_hashes=tuple(_prompt_hash(prompt) for prompt in prompts),
        attention_mask=attention_mask,
        effective_offsets=effective,
        reader_tokens=tuple(reader_rows),
        control_tokens=tuple(control_rows),
        lexical_tokens=tuple(lexical_rows),
        reader_lengths=tuple(reader_lengths),
        control_lengths=tuple(control_lengths),
        fallback_reasons=tuple(None for _ in prompts),
    )


def build_prompt_token_span_batch_with_fallback(
    tokenized: Mapping[str, Any],
    prompts: Sequence[str],
    reader_spans: Sequence[Sequence[Any]],
    *,
    requested_arm: str = "reader",
) -> tuple[PromptTokenSpanMap, tuple[ArmDecision, ...]]:
    """Map every row and disable only rows that fail the reader contract."""
    if requested_arm not in ARMS:
        raise BoundaryError("unknown_arm")
    input_ids, attention_mask, raw_offsets, special_tokens, effective = _prepare_token_batch(tokenized, prompts)
    if len(reader_spans) != len(prompts):
        raise BoundaryError("reader_span_batch_size_mismatch")
    reader_rows: list[tuple[int, ...]] = []
    control_rows: list[tuple[int, ...]] = []
    lexical_rows: list[tuple[int, ...]] = []
    reader_lengths: list[tuple[int, ...]] = []
    control_lengths: list[tuple[int, ...]] = []
    reasons: list[str | None] = []
    decisions: list[ArmDecision] = []
    for row_index, prompt in enumerate(prompts):
        lexical = tuple(
            index
            for index, (start, end) in enumerate(effective[row_index])
            if attention_mask[row_index][index]
            and not special_tokens[row_index][index]
            and end > start
        )
        try:
            if requested_arm == "disabled":
                decisions.append(ArmDecision("disabled", "disabled", False, None))
                reasons.append(None)
                reader, control, lengths, control_geometry = (), (), (), ()
                reader_rows.append(())
                control_rows.append(())
                lexical_rows.append(tuple(lexical))
                reader_lengths.append(())
                control_lengths.append(())
                continue
            reader, control, lexical, lengths, control_geometry = _build_row_geometry(
                effective[row_index],
                special_tokens[row_index],
                attention_mask[row_index],
                str(prompt),
                reader_spans[row_index],
            )
            decisions.append(ArmDecision(requested_arm, requested_arm, False, None))
            reasons.append(None)
        except BoundaryError as exc:
            decisions.append(ArmDecision(requested_arm, "disabled", True, exc.code))
            reasons.append(exc.code)
            reader, control, lengths, control_geometry = (), (), (), ()
        reader_rows.append(tuple(reader))
        control_rows.append(tuple(control))
        lexical_rows.append(tuple(lexical))
        reader_lengths.append(tuple(lengths))
        control_lengths.append(tuple(control_geometry))
    mapping = PromptTokenSpanMap(
        input_ids_hash=_stable_hash(input_ids),
        attention_mask_hash=_stable_hash(attention_mask),
        offset_mapping_hash=_stable_hash(raw_offsets),
        special_tokens_mask_hash=_stable_hash(special_tokens),
        prompt_hashes=tuple(_prompt_hash(prompt) for prompt in prompts),
        attention_mask=attention_mask,
        effective_offsets=effective,
        reader_tokens=tuple(reader_rows),
        control_tokens=tuple(control_rows),
        lexical_tokens=tuple(lexical_rows),
        reader_lengths=tuple(reader_lengths),
        control_lengths=tuple(control_lengths),
        fallback_reasons=tuple(reasons),
    )
    return mapping, tuple(decisions)


def assert_prompt_token_identity(
    mapping: PromptTokenSpanMap,
    tokenized: Mapping[str, Any],
    prompts: Sequence[str],
) -> None:
    """Reject any serializer/tokenizer drift after the span map was built."""
    if len(prompts) != mapping.batch_size or tuple(_prompt_hash(prompt) for prompt in prompts) != mapping.prompt_hashes:
        raise BoundaryError("prompt_identity_changed")
    input_ids, attention_mask, raw_offsets, special_tokens, _ = _prepare_token_batch(tokenized, prompts)
    if _stable_hash(input_ids) != mapping.input_ids_hash:
        raise BoundaryError("input_ids_identity_changed")
    if _stable_hash(attention_mask) != mapping.attention_mask_hash:
        raise BoundaryError("attention_mask_identity_changed")
    if _stable_hash(raw_offsets) != mapping.offset_mapping_hash:
        raise BoundaryError("offset_mapping_identity_changed")
    if _stable_hash(special_tokens) != mapping.special_tokens_mask_hash:
        raise BoundaryError("special_tokens_identity_changed")


def _locate_unique_hypothesis(passage: str, hypothesis: str) -> CharSpan | None:
    """Locate one reader hypothesis without using answer labels or fuzzy scores."""
    candidate = str(hypothesis).strip()
    if not candidate:
        return None
    exact = [match.span() for match in re.finditer(re.escape(candidate), passage)]
    if len(exact) == 1:
        return int(exact[0][0]), int(exact[0][1])
    if exact:
        return None
    pieces = [re.escape(piece) for piece in candidate.split()]
    if not pieces:
        return None
    relaxed = [
        match.span()
        for match in re.finditer(r"\s+".join(pieces), passage, flags=re.IGNORECASE)
    ]
    if len(relaxed) != 1:
        return None
    return int(relaxed[0][0]), int(relaxed[0][1])


class FrozenDPRReaderSpanProvider:
    """Derive passage-local spans from the already-frozen DPR reader API.

    The returned reader scores are deliberately discarded.  Only pre-generation
    span hypotheses are consumed, and no label, gold answer, evaluator value,
    generated output, or selector feedback enters this provider.
    """

    TOP_M = 3
    MAX_ANSWER_TOKENS = 10

    def __init__(self, answer_scorer: Any) -> None:
        if not callable(getattr(answer_scorer, "score_passages_with_hypotheses", None)):
            raise BoundaryError("reader_provider_api_unavailable")
        self.answer_scorer = answer_scorer

    def provide(self, question: str, passages: Sequence[str]) -> ReaderSpanProvenance:
        passage_rows = tuple(str(passage) for passage in passages)
        if not passage_rows:
            return ReaderSpanProvenance((), 0, 0, 0, "empty")
        try:
            unused_scores, hypotheses = self.answer_scorer.score_passages_with_hypotheses(
                str(question),
                list(passage_rows),
                top_m=self.TOP_M,
                max_answer_tokens=self.MAX_ANSWER_TOKENS,
            )
            del unused_scores
        except Exception:
            return ReaderSpanProvenance(
                tuple(() for _ in passage_rows), 0, 0, 0, "provider_error"
            )
        if not isinstance(hypotheses, Sequence) or len(hypotheses) != len(passage_rows):
            return ReaderSpanProvenance(
                tuple(() for _ in passage_rows), 0, 0, 0, "provider_error"
            )
        located_rows: list[tuple[CharSpan, ...]] = []
        hypothesis_count = 0
        rejected_count = 0
        for passage, row in zip(passage_rows, hypotheses):
            if not isinstance(row, Sequence):
                return ReaderSpanProvenance(
                    tuple(() for _ in passage_rows), 0, 0, 0, "provider_error"
                )
            accepted: list[CharSpan] = []
            for item in row:
                hypothesis_count += 1
                if not isinstance(item, Mapping):
                    rejected_count += 1
                    continue
                span = _locate_unique_hypothesis(
                    passage, str(item.get("text") or item.get("normalized") or "")
                )
                if span is None or any(
                    span[0] < existing[1] and span[1] > existing[0]
                    for existing in accepted
                ):
                    rejected_count += 1
                    continue
                accepted.append(span)
            located_rows.append(tuple(sorted(accepted)))
        located_count = hypothesis_count - rejected_count
        status = "ok" if located_count else "empty"
        return ReaderSpanProvenance(
            tuple(located_rows),
            hypothesis_count,
            located_count,
            rejected_count,
            status,
        )


class GeneratorPromptSpanAdapter:
    """Map passage-local reader spans through the unchanged Generator prompt."""

    MAX_LENGTH = 4096

    @staticmethod
    def _prompt_spans(
        prompt: str,
        passages: Sequence[str],
        passage_spans: Sequence[Sequence[CharSpan]],
    ) -> tuple[CharSpan, ...]:
        if len(passages) != len(passage_spans):
            raise BoundaryError("reader_passage_span_count_mismatch")
        cursor = 0
        result: list[CharSpan] = []
        for index, (passage, spans) in enumerate(zip(passages, passage_spans), start=1):
            prefix = f"Passage {index}: "
            serialized = prefix + str(passage)
            start = prompt.find(serialized, cursor)
            if start < 0:
                raise BoundaryError("prompt_passage_serialization_mismatch")
            passage_start = start + len(prefix)
            passage_length = len(str(passage))
            for local_start, local_end in spans:
                if not 0 <= int(local_start) < int(local_end) <= passage_length:
                    raise BoundaryError("reader_span_out_of_passage")
                result.append((passage_start + int(local_start), passage_start + int(local_end)))
            cursor = start + len(serialized)
        if not result:
            raise BoundaryError("reader_span_empty")
        return tuple(result)

    @staticmethod
    def _tokenize(tokenizer: Any, prompt: str, *, geometry: bool) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": GeneratorPromptSpanAdapter.MAX_LENGTH,
        }
        if geometry:
            kwargs.update(return_offsets_mapping=True, return_special_tokens_mask=True)
        try:
            encoded = tokenizer(prompt, **kwargs)
        except Exception as exc:
            raise BoundaryError("prompt_tokenizer_geometry_unavailable") from exc
        if not isinstance(encoded, Mapping):
            raise BoundaryError("tokenized_input_not_mapping")
        return encoded

    def build(
        self,
        generator: Any,
        question: str,
        passages: Sequence[str],
        provenance: ReaderSpanProvenance,
        *,
        requested_arm: str,
    ) -> tuple[PromptTokenSpanMap, tuple[ArmDecision, ...]]:
        if requested_arm not in ARMS:
            raise BoundaryError("unknown_arm")
        passage_rows = tuple(str(passage) for passage in passages)
        first_prompt = generator._build_prompt(str(question), list(passage_rows))
        second_prompt = generator._build_prompt(str(question), list(passage_rows))
        if first_prompt != second_prompt:
            raise BoundaryError("prompt_serializer_nondeterministic")
        geometry = dict(self._tokenize(generator.tokenizer, first_prompt, geometry=True))
        production = self._tokenize(generator.tokenizer, second_prompt, geometry=False)
        for key in ("input_ids", "attention_mask"):
            if key not in production or key not in geometry:
                raise BoundaryError(f"missing_{key}")
            if _stable_hash(production[key]) != _stable_hash(geometry[key]):
                raise BoundaryError(f"{key}_geometry_tokenization_changed")
        if requested_arm == "disabled":
            prompt_spans: tuple[CharSpan, ...] = ()
        else:
            try:
                prompt_spans = self._prompt_spans(
                    first_prompt, passage_rows, provenance.passage_spans
                )
            except BoundaryError:
                prompt_spans = ()
        mapping, decisions = build_prompt_token_span_batch_with_fallback(
            geometry,
            (first_prompt,),
            (prompt_spans,),
            requested_arm=requested_arm,
        )
        assert_prompt_token_identity(mapping, geometry, (first_prompt,))
        return mapping, decisions


class _TokenizerIdentityProxy:
    def __init__(self, tokenizer: Any, mapping: PromptTokenSpanMap) -> None:
        self._tokenizer = tokenizer
        self._mapping = mapping
        self.call_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs != {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": GeneratorPromptSpanAdapter.MAX_LENGTH,
        }:
            raise BoundaryError("generator_tokenizer_arguments_changed")
        if _prompt_hash(prompt) != self._mapping.prompt_hashes[0]:
            raise BoundaryError("generator_prompt_identity_changed")
        encoded = self._tokenizer(prompt, **kwargs)
        if not isinstance(encoded, Mapping):
            raise BoundaryError("tokenized_input_not_mapping")
        if _stable_hash(encoded.get("input_ids")) != self._mapping.input_ids_hash:
            raise BoundaryError("generator_input_ids_identity_changed")
        if _stable_hash(encoded.get("attention_mask")) != self._mapping.attention_mask_hash:
            raise BoundaryError("generator_attention_mask_identity_changed")
        self.call_count += 1
        return encoded


class GeneratorCallIdentityGuard(AbstractContextManager["GeneratorCallIdentityGuard"]):
    """Observe the actual stable Generator call and restore it on every exit."""

    def __init__(self, generator: Any, mapping: PromptTokenSpanMap) -> None:
        if mapping.batch_size != 1:
            raise BoundaryError("generator_guard_requires_single_row")
        self.generator = generator
        self.mapping = mapping
        self.original_builder: Any | None = None
        self.original_tokenizer: Any | None = None
        self.proxy: _TokenizerIdentityProxy | None = None
        self.builder_call_count = 0

    def __enter__(self) -> "GeneratorCallIdentityGuard":
        self.original_builder = self.generator._build_prompt
        self.original_tokenizer = self.generator.tokenizer
        self.proxy = _TokenizerIdentityProxy(self.original_tokenizer, self.mapping)

        def guarded_builder(question: str, passages: Sequence[str]) -> str:
            prompt = self.original_builder(question, list(passages))
            if _prompt_hash(prompt) != self.mapping.prompt_hashes[0]:
                raise BoundaryError("generator_prompt_identity_changed")
            self.builder_call_count += 1
            return prompt

        self.generator.tokenizer = self.proxy
        self.generator._build_prompt = guarded_builder
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.original_builder is not None:
            self.generator._build_prompt = self.original_builder
        if self.original_tokenizer is not None:
            self.generator.tokenizer = self.original_tokenizer
        if exc_type is None and (
            self.builder_call_count != 1
            or self.proxy is None
            or self.proxy.call_count != 1
        ):
            raise BoundaryError("generator_call_identity_not_observed")


class GeneratorNativeEvidenceAnchorObserver:
    """Call the stable Generator under one scoped, report-only transport arm."""

    def __init__(
        self,
        generator: Any,
        reader_provider: FrozenDPRReaderSpanProvider,
        *,
        adapter: GeneratorPromptSpanAdapter | None = None,
        registry: "ResidualPluginRegistry | None" = None,
    ) -> None:
        self.generator = generator
        self.reader_provider = reader_provider
        self.adapter = adapter or GeneratorPromptSpanAdapter()
        self.registry = registry or ResidualPluginRegistry()

    @staticmethod
    def _empty_provenance(passage_count: int, status: str) -> ReaderSpanProvenance:
        return ReaderSpanProvenance(
            tuple(() for _ in range(passage_count)), 0, 0, 0, status
        )

    def generate(
        self,
        question: str,
        passages: Sequence[str],
        *,
        requested_arm: str,
    ) -> GenerationObservation:
        if requested_arm not in ARMS:
            raise BoundaryError("unknown_arm")
        passage_rows = tuple(str(passage) for passage in passages)
        provenance = (
            self._empty_provenance(len(passage_rows), "empty")
            if requested_arm == "disabled"
            else self.reader_provider.provide(str(question), passage_rows)
        )
        try:
            mapping, decisions = self.adapter.build(
                self.generator,
                str(question),
                passage_rows,
                provenance,
                requested_arm=requested_arm,
            )
            decision = decisions[0]
            hook = self.registry.create_hook(
                self.generator.model,
                mapping,
                requested_arm=requested_arm,
                decisions=decisions,
            )
        except BoundaryError as exc:
            text = self.generator.generate(str(question), list(passage_rows))
            return GenerationObservation(
                text=str(text),
                decision=ArmDecision(requested_arm, "disabled", True, exc.code),
                provider=provenance,
                mapping=None,
                hook_call_count=0,
                hook_prefill_seen=False,
                boundary_status="unwrapped_disabled_fallback",
            )
        with hook, GeneratorCallIdentityGuard(self.generator, mapping):
            text = self.generator.generate(str(question), list(passage_rows))
        return GenerationObservation(
            text=str(text),
            decision=decision,
            provider=provenance,
            mapping=mapping,
            hook_call_count=hook.call_count,
            hook_prefill_seen=hook.prefill_seen,
            boundary_status="wrapped",
        )


def validate_plugin_allowlist(allowlist: Sequence[str], composition_order: Sequence[str] | None = None) -> tuple[str, ...]:
    """Validate the formal registry; filesystem discovery is never used."""
    resolved = tuple(str(item) for item in allowlist)
    if resolved != PLUGIN_ORDER:
        raise BoundaryError("plugin_allowlist_not_frozen")
    if composition_order is not None and tuple(str(item) for item in composition_order) != resolved:
        raise BoundaryError("plugin_composition_order_mismatch")
    return resolved


class ResidualPluginRegistry:
    """Explicit registry for the three mutually exclusive transport arms."""

    def __init__(self, allowlist: Sequence[str] = PLUGIN_ORDER) -> None:
        self.allowlist = validate_plugin_allowlist(allowlist)

    def arm_plugin_id(self, arm: str) -> str:
        if arm not in ARMS:
            raise BoundaryError("unknown_arm")
        return ARM_PLUGIN_IDS[arm]

    def create_hook(
        self,
        model: Any,
        mapping: PromptTokenSpanMap,
        *,
        requested_arm: str,
        decisions: Sequence[ArmDecision] | None = None,
        alpha: float | None = None,
    ) -> "ResidualAnchorHook":
        if requested_arm not in ARMS:
            raise BoundaryError("unknown_arm")
        if decisions is None:
            arms = tuple(requested_arm for _ in range(mapping.batch_size))
        else:
            if len(decisions) != mapping.batch_size:
                raise BoundaryError("decision_batch_size_mismatch")
            arms = tuple(decision.effective_arm for decision in decisions)
            if any(decision.requested_arm != requested_arm for decision in decisions):
                raise BoundaryError("decision_requested_arm_mismatch")
        active = {arm for arm in arms if arm != "disabled"}
        if len(active) > 1 or (active and requested_arm not in active):
            raise BoundaryError("mutually_exclusive_arms_required")
        return ResidualAnchorHook(model, mapping, arms=arms, alpha=alpha)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - unsupported host
        raise BoundaryError("torch_unavailable") from exc
    return torch


def _find_layers(model: Any, layer_index: int = INTERVENTION_LAYER) -> Sequence[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None or len(layers) <= layer_index:
        raise BoundaryError("intervention_layer_unavailable")
    return layers


def _extract_hidden(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any, str]:
    if args:
        return args[0], "args"
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"], "kwargs"
    raise BoundaryError("hidden_states_missing")


def _replace_hidden(args: tuple[Any, ...], kwargs: Mapping[str, Any], hidden: Any, location: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if location == "args":
        return (hidden, *args[1:]), dict(kwargs)
    updated = dict(kwargs)
    updated["hidden_states"] = hidden
    return args, updated


def _rms_normalize(vector: Any, epsilon: float = 1.0e-6) -> Any:
    torch = _import_torch()
    rms = torch.sqrt(torch.mean(vector.float() * vector.float()) + epsilon)
    return vector / rms.to(dtype=vector.dtype)


class ResidualAnchorHook(AbstractContextManager["ResidualAnchorHook"]):
    """Batch-safe, context-scoped pre-hook with per-question disabled rows."""

    def __init__(
        self,
        model: Any,
        mapping: PromptTokenSpanMap,
        *,
        arms: Sequence[str],
        alpha: float | None = None,
        layer_index: int = INTERVENTION_LAYER,
    ) -> None:
        if len(arms) != mapping.batch_size:
            raise BoundaryError("hook_arm_batch_size_mismatch")
        if any(arm not in ARMS for arm in arms):
            raise BoundaryError("unknown_arm")
        active = {arm for arm in arms if arm != "disabled"}
        if len(active) > 1:
            raise BoundaryError("mutually_exclusive_arms_required")
        for row_index, arm in enumerate(arms):
            selected = mapping.reader_tokens[row_index] if arm == "reader" else mapping.control_tokens[row_index]
            if arm != "disabled" and (not selected or not mapping.lexical_tokens[row_index]):
                raise BoundaryError("active_arm_missing_geometry")
        self.model = model
        self.mapping = mapping
        self.arms = tuple(arms)
        self.alpha = alpha
        self.layer_index = int(layer_index)
        self.handle: Any | None = None
        self.prefill_seen = False
        self.call_count = 0
        self.anchor: Any | None = None
        self.shared_mean: Any | None = None
        self.raw_anchor: Any | None = None
        self.last_prefill_shape: tuple[int, ...] | None = None
        self.last_decode_shape: tuple[int, ...] | None = None

    def __enter__(self) -> "ResidualAnchorHook":
        layers = _find_layers(self.model, self.layer_index)
        expected_alpha = 1.0 / math.sqrt(len(layers))
        if self.alpha is None:
            self.alpha = expected_alpha
        elif not math.isfinite(float(self.alpha)) or not math.isclose(float(self.alpha), expected_alpha, rel_tol=0.0, abs_tol=1.0e-12):
            raise BoundaryError("alpha_rule_violation")
        try:
            self.handle = layers[self.layer_index].register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
        except TypeError as exc:
            raise BoundaryError("kwargs_aware_hook_unavailable") from exc
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _forward_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        torch = _import_torch()
        hidden, location = _extract_hidden(args, kwargs)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise BoundaryError("hidden_states_shape_invalid")
        if hidden.shape[0] != self.mapping.batch_size:
            raise BoundaryError("hook_batch_size_changed")
        self.call_count += 1
        active_rows = [index for index, arm in enumerate(self.arms) if arm != "disabled"]
        if not active_rows:
            return None
        if not self.prefill_seen:
            if hidden.shape[1] != self.mapping.sequence_length:
                raise BoundaryError("prefill_sequence_shape_changed")
            anchors: list[Any] = []
            shared_means: list[Any] = []
            raw_anchors: list[Any] = []
            for row_index in range(self.mapping.batch_size):
                if row_index not in active_rows:
                    zero = hidden.new_zeros((hidden.shape[-1],))
                    anchors.append(zero)
                    shared_means.append(zero)
                    raw_anchors.append(zero)
                    continue
                selected_indices = (
                    self.mapping.reader_tokens[row_index]
                    if self.arms[row_index] == "reader"
                    else self.mapping.control_tokens[row_index]
                )
                lexical_indices = self.mapping.lexical_tokens[row_index]
                if not selected_indices or not lexical_indices:
                    raise BoundaryError("active_arm_missing_geometry")
                selected = hidden[row_index, list(selected_indices), :].detach()
                lexical = hidden[row_index, list(lexical_indices), :].detach()
                shared_mean = lexical.mean(dim=0)
                raw_anchor = selected.mean(dim=0) - shared_mean
                shared_means.append(shared_mean)
                raw_anchors.append(raw_anchor)
                anchors.append(_rms_normalize(raw_anchor))
            self.anchor = torch.stack(anchors, dim=0)
            self.shared_mean = torch.stack(shared_means, dim=0)
            self.raw_anchor = torch.stack(raw_anchors, dim=0)
            self.prefill_seen = True
            self.last_prefill_shape = tuple(int(value) for value in hidden.shape)
        elif hidden.shape[1] == 1:
            self.last_decode_shape = tuple(int(value) for value in hidden.shape)

        if self.anchor is None:
            raise BoundaryError("anchor_not_captured")
        modified = hidden.clone()
        delta = (float(self.alpha) * self.anchor).to(device=hidden.device, dtype=hidden.dtype)
        if hidden.shape[1] == 1:
            positions = [0] * self.mapping.batch_size
        elif not self.prefill_seen or hidden.shape[1] == self.mapping.sequence_length:
            positions = []
            for mask in self.mapping.attention_mask:
                active = [index for index, value in enumerate(mask) if value]
                positions.append(active[-1])
        else:
            positions = [hidden.shape[1] - 1] * self.mapping.batch_size
        for row_index in active_rows:
            modified[row_index, positions[row_index], :] = modified[row_index, positions[row_index], :] + delta[row_index]
        return _replace_hidden(args, kwargs, modified, location)


def compact_decisions(decisions: Sequence[ArmDecision]) -> list[dict[str, Any]]:
    return [
        {
            "requested_arm": decision.requested_arm,
            "effective_arm": decision.effective_arm,
            "fallback": decision.fallback,
            "reason": decision.reason,
        }
        for decision in decisions
    ]


def find_forbidden_fields(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(find_forbidden_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_fields(child, f"{path}[{index}]"))
    return found


def build_compact_boundary_manifest(
    mapping: PromptTokenSpanMap,
    decisions: Sequence[ArmDecision],
    *,
    config_sha256: str = "unavailable",
    code_revision: str = "unavailable",
) -> dict[str, Any]:
    """Create a compact smoke manifest with explicit all-question accounting."""
    if len(decisions) != mapping.batch_size:
        raise BoundaryError("decision_batch_size_mismatch")
    manifest: dict[str, Any] = {
        "schema_version": "rag-selector.generator-native-boundary.v1",
        "candidate_id": CANDIDATE_ID,
        "plugin_version": PLUGIN_VERSION,
        "registry": {
            "mode": "explicit_allowlist",
            "allowlist": list(PLUGIN_ORDER),
            "active_transport_arms": sorted({decision.effective_arm for decision in decisions}),
        },
        "fallback_policy": {
            "id": FALLBACK_POLICY_ID,
            "invalid_row_action": "disabled_zero_anchor",
            "all_questions_accounted": True,
            "drop_invalid_rows": False,
            "retry_invalid_rows": False,
            "feedback_to_selector": False,
        },
        "mapping": mapping.compact(),
        "decisions": compact_decisions(decisions),
        "accounting": {
            "input_questions": mapping.batch_size,
            "emitted_questions": len(decisions),
            "fallback_questions": sum(decision.fallback for decision in decisions),
            "unresolved_questions": 0,
        },
        "reproducibility": {
            "config_sha256": config_sha256,
            "code_revision": code_revision,
            "dataset_accessed": False,
            "retrieval_called": False,
            "selector_called": False,
            "evaluator_called": False,
        },
    }
    forbidden = find_forbidden_fields(manifest)
    if forbidden:
        raise BoundaryError("compact_privacy_violation")
    return manifest


__all__ = [
    "ACTIVE_ARMS",
    "ARMS",
    "ARM_PLUGIN_IDS",
    "ArmDecision",
    "BoundaryError",
    "CANDIDATE_ID",
    "FALLBACK_POLICY_ID",
    "PLUGIN_ORDER",
    "PLUGIN_VERSION",
    "PromptTokenSpanMap",
    "ReaderSpanProvenance",
    "FrozenDPRReaderSpanProvider",
    "GeneratorCallIdentityGuard",
    "GeneratorPromptSpanAdapter",
    "GenerationObservation",
    "GeneratorNativeEvidenceAnchorObserver",
    "ResidualAnchorHook",
    "ResidualPluginRegistry",
    "assert_prompt_token_identity",
    "build_compact_boundary_manifest",
    "build_prompt_token_span_batch_with_fallback",
    "build_prompt_token_span_map",
    "compact_decisions",
    "find_forbidden_fields",
    "validate_plugin_allowlist",
]
