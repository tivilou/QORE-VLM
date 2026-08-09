"""Validation and normalization for configuration-driven development gates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # Legacy analyzer mode can run without YAML installed.
    yaml = None


DEFAULT_CONFIGS = (
    {"name": "qore_dpr", "kind": "qore", "args": {"method": "qore", "gamma": 0.5}},
    {
        "name": "qore_as_control",
        "kind": "qore",
        "args": {"method": "qore", "gamma": 0.5, "delta": 0.0, "use_answer_scorer": True},
    },
    {
        "name": "qore_as_idea6",
        "kind": "qore",
        "args": {
            "method": "qore",
            "gamma": 0.5,
            "delta": 0.1,
            "complementarity_method": "dpr",
            "use_answer_scorer": True,
        },
    },
    {"name": "topk_as", "kind": "baseline", "args": {"method": "topk", "use_answer_scorer": True}},
    {
        "name": "mmr_as",
        "kind": "baseline",
        "args": {"method": "mmr", "lambda_mmr": 0.7, "use_answer_scorer": True},
    },
)
DEFAULT_PAIRS = (
    ("qore_dpr", "qore_as_control"),
    ("qore_as_control", "qore_as_idea6"),
    ("qore_as_idea6", "topk_as"),
    ("qore_as_idea6", "mmr_as"),
)
DEFAULT_SHARED_ARGS = {
    "dataset": "nq_open",
    "split": "validation",
    "max_samples": 200,
    "corpus_mode": "wiki_dpr",
    "K": 5,
    "lam": 2.0,
    "seed": 42,
    "dump_passages": True,
}
DEFAULT_CACHE_STATS = {
    "available": False,
    "hits": None,
    "misses": None,
    "reason": "Answer Scorer cache is not implemented in Phase 1",
}


class ManifestError(ValueError):
    """Raised when a development gate manifest is malformed or ambiguous."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a mapping")
    return dict(value)


def _args(value: Any, name: str) -> dict[str, Any]:
    result = _mapping(value, name)
    forbidden = sorted(set(result) & {"output_dir", "output_file"})
    if forbidden:
        raise ManifestError(f"{name} cannot override runner-owned keys: {', '.join(forbidden)}")
    for key, item in result.items():
        if not isinstance(key, str) or not key or not isinstance(item, (str, int, float, bool)):
            raise ManifestError(f"{name} contains an unsupported argument value for {key!r}")
    return result


def _normalize(raw: Mapping[str, Any], source: Path | None) -> dict[str, Any]:
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ManifestError(f"unsupported schema_version: {schema_version!r}")
    gate = _mapping(raw.get("gate"), "gate")
    name = gate.get("name")
    if not isinstance(name, str) or not name:
        raise ManifestError("gate.name must be a non-empty string")
    shared = _args(gate.get("shared_args", {}), "gate.shared_args")
    if not shared:
        raise ManifestError("gate.shared_args must not be empty")
    specs = gate.get("configurations")
    if not isinstance(specs, list) or not specs:
        raise ManifestError("gate.configurations must be a non-empty list")
    configurations: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, value in enumerate(specs):
        spec = _mapping(value, f"gate.configurations[{index}]")
        config_name = spec.get("name")
        if not isinstance(config_name, str) or not config_name:
            raise ManifestError(f"gate.configurations[{index}].name must be non-empty")
        if config_name in names:
            raise ManifestError(f"duplicate configuration name: {config_name}")
        kind = spec.get("kind", "baseline")
        if kind not in {"qore", "baseline"}:
            raise ManifestError(f"{config_name}: kind must be qore or baseline")
        names.add(config_name)
        configurations.append({
            "name": config_name,
            "kind": kind,
            "args": _args(spec.get("args", {}), f"{config_name}.args"),
            "description": str(spec.get("description", "")),
        })

    pairs_raw = gate.get("paired_comparisons", list(DEFAULT_PAIRS))
    if not isinstance(pairs_raw, list):
        raise ManifestError("gate.paired_comparisons must be a list")
    pairs: list[tuple[str, str]] = []
    for index, value in enumerate(pairs_raw):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ManifestError(f"gate.paired_comparisons[{index}] must contain [left, right]")
        left, right = value
        if left not in names or right not in names or left == right:
            raise ManifestError(f"invalid paired comparison: {value!r}")
        pairs.append((str(left), str(right)))

    cache_stats = dict(DEFAULT_CACHE_STATS)
    cache_stats.update(_mapping(raw.get("cache_stats", {}), "cache_stats"))
    if not isinstance(cache_stats.get("available"), bool):
        raise ManifestError("cache_stats.available must be boolean")
    if not cache_stats["available"] and not isinstance(cache_stats.get("reason"), str):
        raise ManifestError("cache_stats.reason is required when cache is unavailable")

    return {
        "schema_version": 1,
        "source": str(source) if source else None,
        "gate": {
            "name": name,
            "description": str(gate.get("description", "")),
            "shared_args": shared,
            "configurations": configurations,
            "paired_comparisons": [list(pair) for pair in pairs],
        },
        "cache_stats": cache_stats,
    }


def default_manifest() -> dict[str, Any]:
    return _normalize(
        {
            "schema_version": 1,
            "gate": {
                "name": "five_idea_development_gate",
                "description": "Matched 200-question development gate for the frozen five configurations.",
                "shared_args": DEFAULT_SHARED_ARGS,
                "configurations": list(DEFAULT_CONFIGS),
                "paired_comparisons": [list(pair) for pair in DEFAULT_PAIRS],
            },
            "cache_stats": DEFAULT_CACHE_STATS,
        },
        None,
    )


def load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_manifest()
    if yaml is None:
        raise ManifestError("PyYAML is required to load a gate manifest")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _normalize(_mapping(raw, "manifest"), path.resolve())


def configuration_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {spec["name"]: spec for spec in manifest["gate"]["configurations"]}
