"""Reader-span energy decoding for observation-only RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping, Sequence


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(str(text), add_special_tokens=False)
    values = encoded.get("input_ids", []) if isinstance(encoded, Mapping) else []
    if values and isinstance(values[0], list):
        values = values[0]
    return tuple(int(value) for value in values)


def _longest_prefix_suffix(prefix: Sequence[int], history: Sequence[int]) -> int:
    limit = min(len(prefix), len(history))
    for width in range(limit, 0, -1):
        if tuple(history[-width:]) == tuple(prefix[:width]):
            return width
    return 0


@dataclass(frozen=True)
class SpanLattice:
    """A weighted finite-state span prior over generated token prefixes."""

    spans: tuple[tuple[tuple[int, ...], float], ...]

    def __post_init__(self) -> None:
        if not self.spans:
            return
        total = 0.0
        seen: set[tuple[int, ...]] = set()
        for tokens, weight in self.spans:
            if not tokens or tokens in seen or not math.isfinite(float(weight)) or float(weight) <= 0.0:
                raise ValueError("span lattice requires unique non-empty positive spans")
            seen.add(tokens)
            total += float(weight)
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("span lattice weights must sum to one")

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def token_count(self) -> int:
        return sum(len(tokens) for tokens, _ in self.spans)

    def next_token_energy(self, generated_ids: Sequence[int]) -> dict[int, float]:
        """Return non-negative log energies for valid span continuations.

        A completed or unmatched history may begin a new span. ``log1p`` keeps
        the fixed reader contribution positive while bounded by the normalized
        lattice mass; no fitted temperature or confidence threshold is used.
        """

        mass: dict[int, float] = {}
        for tokens, weight in self.spans:
            matched = _longest_prefix_suffix(tokens, generated_ids)
            if matched == len(tokens):
                matched = 0
            token = int(tokens[matched])
            mass[token] = mass.get(token, 0.0) + float(weight)
        return {token: float(math.log1p(value)) for token, value in mass.items() if value > 0.0}


@dataclass(frozen=True)
class DecodeResult:
    text: str
    generated_token_count: int
    energy_active_steps: int
    used_energy_decoder: bool


def lattice_from_reader_hypotheses(
    hypotheses_by_passage: Sequence[Sequence[Mapping[str, Any]]],
    passage_scores: Sequence[float],
    tokenizer: Any,
    *,
    max_spans: int,
) -> SpanLattice:
    """Build a deterministic normalized lattice from frozen reader spans."""

    if max_spans < 1:
        raise ValueError("max_spans must be positive")
    if len(hypotheses_by_passage) != len(passage_scores):
        raise ValueError("reader hypotheses and passage scores must align")
    merged: dict[tuple[int, ...], float] = {}
    for hypotheses, passage_score in zip(hypotheses_by_passage, passage_scores):
        score = max(0.0, float(passage_score))
        if not math.isfinite(score):
            continue
        for item in hypotheses:
            probability = float(item.get("probability", 0.0))
            if not math.isfinite(probability) or probability <= 0.0:
                continue
            tokens = _token_ids(tokenizer, str(item.get("text") or item.get("normalized") or ""))
            if not tokens:
                continue
            merged[tokens] = merged.get(tokens, 0.0) + probability * score
    ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[:max_spans]
    total = sum(weight for _, weight in ranked)
    if total <= 0.0:
        return SpanLattice(())
    return SpanLattice(tuple((tokens, float(weight / total)) for tokens, weight in ranked))


def matched_control_lattice(
    tokenizer: Any,
    passages: Sequence[str],
    reader_lattice: SpanLattice,
    *,
    seed_key: str,
) -> SpanLattice:
    """Create a same-shape, score-independent span control from selected text."""

    if not reader_lattice.spans:
        return SpanLattice(())
    source: list[int] = []
    for passage in passages:
        source.extend(_token_ids(tokenizer, str(passage)))
    if not source:
        return SpanLattice(())
    control: list[tuple[tuple[int, ...], float]] = []
    used: set[tuple[int, ...]] = set()
    for index, (reader_tokens, weight) in enumerate(reader_lattice.spans):
        width = min(len(reader_tokens), len(source))
        choices = max(1, len(source) - width + 1)
        digest = hashlib.sha256(f"{seed_key}:{index}:{width}".encode("utf-8")).digest()
        start = int.from_bytes(digest[:8], "big") % choices
        candidate = tuple(source[start : start + width])
        if candidate in used:
            for offset in range(1, choices):
                replacement = tuple(source[(start + offset) % choices : (start + offset) % choices + width])
                if len(replacement) == width and replacement not in used:
                    candidate = replacement
                    break
        if not candidate or candidate in used:
            raise ValueError("selected context cannot provide a matched unique control span")
        used.add(candidate)
        control.append((candidate, weight))
    if len(control) != len(reader_lattice.spans):
        raise ValueError("matched control must preserve reader lattice cardinality")
    total = sum(weight for _, weight in control)
    return SpanLattice(tuple((tokens, float(weight / total)) for tokens, weight in control))


def decode_with_span_energy(
    generator: Any,
    question: str,
    passages: Sequence[str],
    lattice: SpanLattice,
) -> DecodeResult:
    """Generate once with a token-level lattice energy, or delegate exactly.

    The empty-lattice branch calls the stable Generator directly. This is the
    required disabled-path parity behavior and does not invoke custom decoding.
    """

    if not lattice.spans:
        text = generator.generate(question, list(passages))
        return DecodeResult(text=text, generated_token_count=0, energy_active_steps=0, used_energy_decoder=False)

    import torch

    prompt = generator._build_prompt(question, list(passages))
    inputs = generator.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(generator.model.device) for key, value in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    past_key_values = None
    current_ids = input_ids
    energy_active_steps = 0
    eos_token_id = getattr(generator.tokenizer, "eos_token_id", None)

    with torch.no_grad():
        for _ in range(int(generator.max_new_tokens)):
            outputs = generator.model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :].float()
            energy = lattice.next_token_energy(generated)
            if energy:
                energy_active_steps += 1
                for token_id, value in energy.items():
                    if 0 <= int(token_id) < logits.shape[-1]:
                        logits[0, int(token_id)] += float(value)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            past_key_values = outputs.past_key_values
            current_ids = next_token
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)),
                dim=1,
            )
    text = generator.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return DecodeResult(text=text, generated_token_count=len(generated), energy_active_steps=energy_active_steps, used_energy_decoder=True)


__all__ = [
    "DecodeResult",
    "SpanLattice",
    "decode_with_span_energy",
    "lattice_from_reader_hypotheses",
    "matched_control_lattice",
]
