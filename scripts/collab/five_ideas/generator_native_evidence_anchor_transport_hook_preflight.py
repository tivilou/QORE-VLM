#!/usr/bin/env python3
"""No-task-data actual-model preflight for the Generator-native anchor hook.

The fixture loads the already cached model, builds one synthetic prompt, and
calls only the existing Generator model/tokenizer objects. It does not access
Wiki-DPR, task data, retrieval, selection, labels, answers, or evaluation.
The hook is installed on a single Llama decoder layer through a context
manager and is removed on every exit path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
INTERVENTION_LAYER = 30
SURROGATE_THRESHOLD = 0.90
PARITY_TOLERANCE = 1.0e-6
RMS_EPSILON = 1.0e-6
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


class HookPreflightError(RuntimeError):
    """Raised when an actual-model interface contract cannot be established."""


@dataclass(frozen=True)
class SpanMapping:
    reader_tokens: tuple[int, ...]
    control_tokens: tuple[int, ...]
    lexical_tokens: tuple[int, ...]
    reader_lengths: tuple[int, ...]
    control_lengths: tuple[int, ...]


def _import_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on unsupported hosts
        raise HookPreflightError(f"torch/transformers unavailable: {exc}") from exc
    return torch, AutoTokenizer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_tensor(tensor: Any) -> str:
    contiguous = tensor.detach().to("cpu").contiguous()
    return _sha256_bytes(contiguous.numpy().tobytes())


def _resolve_model_path(override: str | None = None) -> Path:
    """Resolve an existing local snapshot without downloading anything."""
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    for env_name in ("MODEL_NAME_OR_PATH", "GENERATOR_MODEL_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    home = Path.home()
    cache_roots = [
        Path(os.environ[name]).expanduser()
        for name in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
        if os.environ.get(name)
    ]
    cache_roots.extend(
        [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers",
        ]
    )
    for root in cache_roots:
        if root.name == "hub":
            model_root = root / "models--NousResearch--Meta-Llama-3-8B-Instruct"
        else:
            model_root = root / "models--NousResearch--Meta-Llama-3-8B-Instruct"
        candidates.append(model_root / "snapshots" / MODEL_REVISION)
        snapshot_root = model_root / "snapshots"
        if snapshot_root.is_dir():
            candidates.extend(sorted(snapshot_root.iterdir(), reverse=True))

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "config.json").is_file() and (candidate / "tokenizer_config.json").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates[:8])
    raise HookPreflightError(f"cached model snapshot not found; searched {searched}")


def _project_root() -> Path:
    configured = os.environ.get("QORE_PROJECT_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "applications").is_dir() and (candidate / "configs").is_dir():
            return candidate
    script_path = Path(__file__).resolve()
    for candidate in (script_path.parent, *script_path.parents):
        if (candidate / "applications").is_dir() and (candidate / "configs").is_dir():
            return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "applications").is_dir() and (cwd / "configs").is_dir():
        return cwd
    raise HookPreflightError("cannot locate project root")


def _git_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _find_layers(model: Any) -> Sequence[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise HookPreflightError("model.model.layers is unavailable")
    if len(layers) <= INTERVENTION_LAYER:
        raise HookPreflightError(f"model has {len(layers)} layers; layer 30 is unavailable")
    return layers


def _model_device(model: Any) -> Any:
    torch, _ = _import_runtime()
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(model: Any) -> Any:
    torch, _ = _import_runtime()
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.dtype
    configured = getattr(getattr(model, "config", None), "torch_dtype", None)
    if isinstance(configured, torch.dtype):
        return configured
    return torch.float16


def _module_device(module: Any, fallback: Any) -> Any:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return fallback


def _build_synthetic_prompt() -> tuple[str, list[tuple[int, int]]]:
    """Return synthetic text and one reader interval; no task item is used."""
    prompt = (
        "Answer using the synthetic context below.\n\n"
        "Context:\n"
        "Passage 1: alpha beta gamma delta epsilon.\n\n"
        "Passage 2: neutral zero one two three four.\n\n"
        "Question: Which synthetic marker is present?\n"
        "Short Answer:"
    )
    reader_text = "alpha beta gamma"
    start = prompt.index(reader_text)
    return prompt, [(start, start + len(reader_text))]


def _offset_rows(encoded: Mapping[str, Any], text_length: int) -> list[tuple[int, int]]:
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise HookPreflightError("fast tokenizer offset_mapping is required")
    first = offsets[0].tolist() if hasattr(offsets[0], "tolist") else offsets[0]
    special = encoded.get("special_tokens_mask")
    if special is None:
        special_first = [0] * len(first)
    else:
        special_first = special[0].tolist() if hasattr(special[0], "tolist") else special[0]
    starts = [int(row[0]) for row in first]
    rows: list[tuple[int, int]] = []
    for index, row in enumerate(first):
        start, end = int(row[0]), int(row[1])
        if int(special_first[index]):
            rows.append((0, 0))
            continue
        # Llama's fast tokenizer reports the start of whitespace-prefixed BPE
        # pieces but often leaves the end equal to the start. Recover that end
        # from the next non-special token start; this is deterministic and does
        # not infer any task or answer content.
        if end <= start:
            next_starts = [candidate for candidate in starts[index + 1 :] if candidate > start]
            end = next_starts[0] if next_starts else text_length
        if not 0 <= start < end <= text_length:
            raise HookPreflightError("tokenizer returned an invalid effective offset")
        rows.append((start, end))
    if not rows:
        raise HookPreflightError("tokenizer returned no offsets")
    return rows


def _map_char_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> tuple[int, ...]:
    if not 0 <= start < end:
        raise HookPreflightError("invalid synthetic reader interval")
    mapped = tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    )
    if not mapped:
        raise HookPreflightError("reader interval did not map to lexical tokens")
    return mapped


def _choose_control_tokens(
    offsets: Sequence[tuple[int, int]],
    reader_tokens: Sequence[int],
) -> tuple[int, ...]:
    lexical = [index for index, (start, end) in enumerate(offsets) if end > start]
    reader_set = set(reader_tokens)
    span_length = len(reader_tokens)
    for start in range(0, len(lexical) - span_length + 1):
        window = tuple(lexical[start : start + span_length])
        if not reader_set.intersection(window):
            return window
    raise HookPreflightError("no disjoint geometry-matched control window exists")


def _build_mapping(tokenizer: Any) -> tuple[Any, SpanMapping, dict[str, Any]]:
    torch, _ = _import_runtime()
    prompt, reader_intervals = _build_synthetic_prompt()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
    )
    offsets = _offset_rows(encoded, len(prompt))
    reader_tokens = tuple(
        token
        for start, end in reader_intervals
        for token in _map_char_span(offsets, start, end)
    )
    control_tokens = _choose_control_tokens(offsets, reader_tokens)
    lexical_tokens = tuple(index for index, (start, end) in enumerate(offsets) if end > start)
    mapping = SpanMapping(
        reader_tokens=reader_tokens,
        control_tokens=control_tokens,
        lexical_tokens=lexical_tokens,
        reader_lengths=(len(reader_tokens),),
        control_lengths=(len(control_tokens),),
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    compact = {
        "input_ids_hash": _hash_tensor(input_ids),
        "attention_mask_hash": _hash_tensor(attention_mask),
        "prompt_token_count": int(input_ids.shape[1]),
        "reader_token_geometry": list(mapping.reader_lengths),
        "control_token_geometry": list(mapping.control_lengths),
        "reader_token_count": len(mapping.reader_tokens),
        "control_token_count": len(mapping.control_tokens),
        "lexical_token_count": len(mapping.lexical_tokens),
        "reader_control_disjoint": not set(mapping.reader_tokens).intersection(mapping.control_tokens),
        "prompt_source": "synthetic_fixture_only",
    }
    return encoded, mapping, compact


def _rms_normalize(vector: Any, epsilon: float = RMS_EPSILON) -> Any:
    torch, _ = _import_runtime()
    rms = torch.sqrt(torch.mean(vector.float() * vector.float()) + epsilon)
    return vector / rms.to(dtype=vector.dtype)


def _extract_hidden(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[Any, str]:
    if args:
        return args[0], "args"
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"], "kwargs"
    raise HookPreflightError("decoder hook received no hidden_states")


def _replace_hidden(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    hidden: Any,
    location: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if location == "args":
        return (hidden, *args[1:]), dict(kwargs)
    updated = dict(kwargs)
    updated["hidden_states"] = hidden
    return args, updated


class ResidualAnchorHook:
    """Context-scoped pre-hook for one mutually exclusive fixture arm."""

    def __init__(
        self,
        model: Any,
        token_indices: Sequence[int],
        lexical_indices: Sequence[int],
        arm: str,
        alpha: float,
    ) -> None:
        if arm not in {"disabled", "reader", "control"}:
            raise HookPreflightError(f"unknown hook arm: {arm}")
        self.model = model
        self.token_indices = tuple(int(index) for index in token_indices)
        self.lexical_indices = tuple(int(index) for index in lexical_indices)
        self.arm = arm
        self.alpha = float(alpha)
        self.handle: Any | None = None
        self.prefill_seen = False
        self.call_count = 0
        self.anchor: Any | None = None
        self.shared_mean: Any | None = None
        self.selected_mean: Any | None = None
        self.raw_anchor: Any | None = None
        self.last_prefill_shape: tuple[int, ...] | None = None
        self.last_decode_shape: tuple[int, ...] | None = None

    def __enter__(self) -> "ResidualAnchorHook":
        layers = _find_layers(self.model)
        layer = layers[INTERVENTION_LAYER]
        try:
            self.handle = layer.register_forward_pre_hook(self._forward_pre_hook, with_kwargs=True)
        except TypeError as exc:
            raise HookPreflightError("kwargs-aware register_forward_pre_hook is unavailable") from exc
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _forward_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        torch, _ = _import_runtime()
        hidden, location = _extract_hidden(args, kwargs)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise HookPreflightError("layer-30 hidden_states must be [batch, sequence, hidden]")
        if hidden.shape[0] != 1:
            raise HookPreflightError("fixture requires batch size one")
        self.call_count += 1
        if not self.prefill_seen:
            if hidden.shape[1] <= max((*self.token_indices, *self.lexical_indices), default=-1):
                raise HookPreflightError("first layer-30 call is not a full prompt prefill")
            self.prefill_seen = True
            self.last_prefill_shape = tuple(int(value) for value in hidden.shape)
            lexical = hidden[:, self.lexical_indices, :].detach()
            selected = hidden[:, self.token_indices, :].detach()
            self.shared_mean = lexical.mean(dim=(0, 1))
            self.selected_mean = selected.mean(dim=(0, 1))
            self.raw_anchor = self.selected_mean - self.shared_mean
            self.anchor = _rms_normalize(self.raw_anchor)
        elif hidden.shape[1] == 1:
            self.last_decode_shape = tuple(int(value) for value in hidden.shape)

        if self.arm == "disabled":
            return None
        if self.anchor is None:
            raise HookPreflightError("anchor was not captured before transport")
        modified = hidden.clone()
        delta = (self.alpha * self.anchor).to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        modified[:, -1:, :] = modified[:, -1:, :] + delta
        return _replace_hidden(args, kwargs, modified, location)


def _max_abs(left: Any, right: Any) -> float:
    torch, _ = _import_runtime()
    return float(torch.max(torch.abs(left.detach().float() - right.detach().float())).item())


def _r2(actual: Any, predicted: Any) -> float:
    torch, _ = _import_runtime()
    actual = actual.detach().float().reshape(-1)
    predicted = predicted.detach().float().reshape(-1)
    centered = actual - actual.mean()
    total = torch.sum(centered * centered)
    if float(total.item()) <= 1.0e-12:
        return 1.0
    residual = torch.sum((actual - predicted) * (actual - predicted))
    return float((1.0 - residual / total).item())


def _fit_rescaled_surrogate(actual: Any, base: Any) -> tuple[Any, float, float]:
    torch, _ = _import_runtime()
    actual = actual.detach().float().reshape(-1)
    base = base.detach().float().reshape(-1)
    denominator = torch.sum(base * base)
    scale = torch.sum(actual * base) / denominator if float(denominator.item()) > 1.0e-12 else torch.tensor(0.0)
    predicted = scale * base
    return predicted, float(scale.item()), _r2(actual, predicted)


def _model_forward(model: Any, encoded: Mapping[str, Any], hook: ResidualAnchorHook | None = None) -> Any:
    torch, _ = _import_runtime()
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    with torch.no_grad():
        if hook is None:
            return model(**inputs, use_cache=False, return_dict=True)
        return model(**inputs, use_cache=False, return_dict=True)


def _greedy_generate(model: Any, tokenizer: Any, encoded: Mapping[str, Any], max_new_tokens: int = 2) -> Any:
    torch, _ = _import_runtime()
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    with torch.no_grad():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )


def _run_arm(model: Any, tokenizer: Any, encoded: Mapping[str, Any], mapping: SpanMapping, arm: str, alpha: float) -> dict[str, Any]:
    layer = _find_layers(model)[INTERVENTION_LAYER]
    before_hooks = len(getattr(layer, "_forward_pre_hooks", {}))
    token_indices = mapping.reader_tokens if arm == "reader" else mapping.control_tokens
    with ResidualAnchorHook(model, token_indices, mapping.lexical_tokens, arm, alpha) as hook:
        forward = _model_forward(model, encoded, hook)
        generated = _greedy_generate(model, tokenizer, encoded)
        anchor_hash = _hash_tensor(hook.anchor) if hook.anchor is not None else None
        shared_mean_tensor = hook.shared_mean.detach() if hook.shared_mean is not None else None
        raw_anchor_tensor = hook.raw_anchor.detach() if hook.raw_anchor is not None else None
        anchor_tensor = hook.anchor.detach() if hook.anchor is not None else None
        prefill_seen = hook.prefill_seen
        call_count = hook.call_count
        prefill_shape = hook.last_prefill_shape
        decode_shape = hook.last_decode_shape
    after_hooks = len(getattr(layer, "_forward_pre_hooks", {}))
    scores = getattr(generated, "scores", ())
    first_score_hash = _hash_tensor(scores[0]) if scores else None
    sequence = generated.sequences.detach().to("cpu")
    return {
        "arm": arm,
        "logits": forward.logits.detach(),
        "last_hidden_state": getattr(forward, "last_hidden_state", None),
        "sequence_hash": _hash_tensor(sequence),
        "sequence_length": int(sequence.shape[1]),
        "generated_token_count": max(0, int(sequence.shape[1] - encoded["input_ids"].shape[1])),
        "first_score_hash": first_score_hash,
        "anchor_hash": anchor_hash,
        "anchor_tensor": anchor_tensor,
        "shared_mean_tensor": shared_mean_tensor,
        "raw_anchor_tensor": raw_anchor_tensor,
        "hook_prefill_seen": prefill_seen,
        "hook_call_count": call_count,
        "prefill_shape": list(prefill_shape) if prefill_shape else None,
        "decode_shape": list(decode_shape) if decode_shape else None,
        "hooks_before": before_hooks,
        "hooks_after": after_hooks,
        "generated": generated,
    }


def _compact_arm(arm_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arm_result.items()
        if key
        not in {
            "logits",
            "last_hidden_state",
            "generated",
            "anchor_tensor",
            "shared_mean_tensor",
            "raw_anchor_tensor",
        }
    }


def _forbidden(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def run_preflight(model_path: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    torch, AutoTokenizer = _import_runtime()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    resolved = _resolve_model_path(model_path)
    config_path = resolved / "config.json"
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise HookPreflightError("a fast tokenizer is required for exact offset mapping")

    from applications.rag.generation import Generator

    generator = Generator(
        str(resolved),
        device="auto",
        torch_dtype=torch.float16,
        max_new_tokens=2,
        use_chat_template=False,
    )
    model = generator.model
    layers = _find_layers(model)
    encoded, mapping, mapping_report = _build_mapping(tokenizer)
    layer = layers[INTERVENTION_LAYER]
    initial_hooks = len(getattr(layer, "_forward_pre_hooks", {}))
    num_layers = int(len(layers))
    alpha = 1.0 / math.sqrt(num_layers)

    baseline = _run_arm(model, tokenizer, encoded, mapping, "disabled", alpha)
    reader = _run_arm(model, tokenizer, encoded, mapping, "reader", alpha)
    control = _run_arm(model, tokenizer, encoded, mapping, "control", alpha)

    disabled_parity = _max_abs(baseline["logits"], _model_forward(model, encoded).logits)
    reader_delta = reader["logits"][:, -1, :] - baseline["logits"][:, -1, :]
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or reader["anchor_tensor"] is None:
        raise HookPreflightError("model lm_head or reader anchor is unavailable")
    with torch.no_grad():
        reader_anchor_vector = reader["anchor_tensor"].detach()
        model_device = _model_device(model)
        head_device = _module_device(lm_head, model_device)
        terminal_base = lm_head(reader_anchor_vector.to(device=head_device, dtype=_model_dtype(model)))
    _, surrogate_scale, surrogate_r2 = _fit_rescaled_surrogate(reader_delta, terminal_base)
    geometry_match = mapping.reader_lengths == mapping.control_lengths
    shared_mean_delta = _max_abs(reader["shared_mean_tensor"], control["shared_mean_tensor"])
    control_centered_anchor_l2 = float(torch.linalg.vector_norm(control["raw_anchor_tensor"].detach().float()).item())
    cleanup_ok = initial_hooks == len(getattr(layer, "_forward_pre_hooks", {})) == baseline["hooks_after"] == reader["hooks_after"] == control["hooks_after"]
    checks = {
        "layer_reachability": {
            "status": "pass" if num_layers > INTERVENTION_LAYER and num_layers - INTERVENTION_LAYER >= 2 else "fail",
            "layer_index": INTERVENTION_LAYER,
            "total_layers": num_layers,
            "downstream_blocks": num_layers - INTERVENTION_LAYER,
        },
        "span_mapping": {
            "status": "pass" if mapping_report["reader_control_disjoint"] and geometry_match else "fail",
            **mapping_report,
        },
        "disabled_path_parity": {
            "status": "pass" if disabled_parity <= PARITY_TOLERANCE else "fail",
            "max_abs_logit_delta": disabled_parity,
            "tolerance": PARITY_TOLERANCE,
        },
        "hook_lifecycle_cleanup": {
            "status": "pass" if cleanup_ok else "fail",
            "initial_hooks": initial_hooks,
            "final_hooks": len(getattr(layer, "_forward_pre_hooks", {})),
        },
        "geometry_matched_null": {
            "status": "pass" if geometry_match and mapping_report["reader_control_disjoint"] and shared_mean_delta <= PARITY_TOLERANCE else "fail",
            "geometry_match": geometry_match,
            "shared_mean_ablated": True,
            "shared_mean_delta": shared_mean_delta,
            "control_centered_anchor_l2": control_centered_anchor_l2,
            "reader_anchor_hash": reader["anchor_hash"],
            "control_anchor_hash": control["anchor_hash"],
        },
        "non_logit_equivalence": {
            "status": "pass" if surrogate_r2 < SURROGATE_THRESHOLD else "fail",
            "surrogate_r2": surrogate_r2,
            "threshold": SURROGATE_THRESHOLD,
            "fitted_rescale": surrogate_scale,
        },
    }
    failures = [name for name, result in checks.items() if result["status"] != "pass"]
    report: dict[str, Any] = {
        "schema_version": "rag-selector.generator-native-hook-preflight.v1",
        "candidate_id": "generator_native_evidence_anchor_transport",
        "fixture": "actual_model_no_task_data",
        "scope": {
            "task_data_accessed": False,
            "wiki_dpr_accessed": False,
            "retrieval_called": False,
            "selector_called": False,
            "evaluator_called": False,
            "model_loaded": True,
            "gpu_used": bool(torch.cuda.is_available()),
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_config_sha256": _sha256_file(config_path),
            "resolved_snapshot_path_sha256": _sha256_bytes(str(resolved).encode("utf-8")),
            "num_layers": num_layers,
            "hidden_size": int(getattr(model.config, "hidden_size", -1)),
            "torch_dtype": str(next(parameter for parameter in model.parameters() if parameter.device.type != "meta").dtype),
        },
        "interface": {
            "hook": "register_forward_pre_hook(with_kwargs=True)",
            "insertion_layer": INTERVENTION_LAYER,
            "downstream_blocks": num_layers - INTERVENTION_LAYER,
            "alpha_rule": "1/sqrt(num_hidden_layers)",
            "prompt_token_hash": mapping_report["input_ids_hash"],
            "attention_mask_hash": mapping_report["attention_mask_hash"],
        },
        "checks": checks,
        "arms": {
            "disabled": _compact_arm(baseline),
            "reader": _compact_arm(reader),
            "control": _compact_arm(control),
        },
        "status": "pass" if not failures else "kill",
        "failed_checks": failures,
        "claim_ceiling": "diagnostic",
        "next_action": "synthetic_hook_contract_passed" if not failures else "close_candidate",
        "timing": {"wall_seconds": round(time.monotonic() - started, 3)},
    }
    privacy_errors = _forbidden(report)
    if privacy_errors:
        report["status"] = "kill"
        report["failed_checks"] = sorted(set(report["failed_checks"] + ["compact_privacy"]))
        report["privacy_errors"] = privacy_errors
    report["provenance"] = {
        "project_git_revision": _git_revision(root),
        "model_revision": MODEL_REVISION,
        "data_policy": "synthetic fixture only; no task data or raw prompt persisted",
    }
    report["sha256"] = _sha256_bytes(json.dumps(report, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None, help="Existing local model snapshot; no download is attempted.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = run_preflight(args.model_path)
    except HookPreflightError as exc:
        report = {
            "schema_version": "rag-selector.generator-native-hook-preflight.v1",
            "candidate_id": "generator_native_evidence_anchor_transport",
            "fixture": "actual_model_no_task_data",
            "status": "kill",
            "failed_checks": ["fixture_execution"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_ceiling": "diagnostic",
            "scope": {"task_data_accessed": False, "wiki_dpr_accessed": False},
        }
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
