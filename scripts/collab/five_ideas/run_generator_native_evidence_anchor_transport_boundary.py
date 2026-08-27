#!/usr/bin/env python3
"""Run the no-task-data batch/padding/fallback boundary smoke fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - unsupported host
    raise SystemExit(f"ERROR: PyYAML unavailable: {exc}") from exc


SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents):
    if (candidate / "applications").is_dir() and (candidate / "configs").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        PROJECT_ROOT = candidate
        break
else:  # pragma: no cover - malformed checkout
    raise SystemExit("ERROR: cannot locate project root")

from applications.rag.generator_native_evidence_anchor_transport import (  # noqa: E402
    BoundaryError,
    PLUGIN_ORDER,
    PLUGIN_VERSION,
    assert_prompt_token_identity,
    build_compact_boundary_manifest,
    build_prompt_token_span_batch_with_fallback,
    find_forbidden_fields,
    validate_plugin_allowlist,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/generator_native_evidence_anchor_transport_boundary.yaml"
DEFAULT_PLAN = PROJECT_ROOT / "configs/experiments/generator_native_evidence_anchor_transport_boundary_plan.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _load_contract(config_path: Path, plan_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise BoundaryError("contract_read_failed") from exc
    phase = config.get("phase")
    plugin = config.get("plugin")
    mapping = config.get("mapping")
    reader_span_provider = config.get("reader_span_provider")
    generator_adapter = config.get("generator_adapter")
    fallback = config.get("fallback")
    if not isinstance(phase, dict) or phase.get("name") != "generator_native_evidence_anchor_transport_boundary":
        raise BoundaryError("config_phase_mismatch")
    if phase.get("authorization") != "implemented_observation_only" or phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise BoundaryError("config_authorization_mismatch")
    if any(phase.get(key) is not False for key in ("task_data_accessed", "wiki_dpr_accessed", "retrieval_called", "selector_called", "evaluator_called")):
        raise BoundaryError("config_data_boundary_mismatch")
    if not isinstance(plugin, dict) or plugin.get("discovery") != "explicit_allowlist":
        raise BoundaryError("config_registry_mismatch")
    validate_plugin_allowlist(plugin.get("allowlist", ()), plugin.get("composition_order", ()))
    if tuple(plugin.get("transport_arms", ())) != ("disabled", "reader", "control"):
        raise BoundaryError("config_arm_order_mismatch")
    if not isinstance(mapping, dict) or any(mapping.get(key) is not True for key in ("require_input_ids", "require_attention_mask", "require_offset_mapping", "require_special_tokens_mask", "allow_batch", "allow_left_padding", "allow_right_padding", "reject_noncontiguous_attention", "exact_prompt_token_identity")):
        raise BoundaryError("config_mapping_contract_mismatch")
    expected_provider = {
        "provider_id": "frozen_dpr_reader_hypotheses_v1",
        "backend": "dpr",
        "source": "selected_passages_before_generation_only",
        "top_m": 3,
        "max_answer_tokens": 10,
        "unique_text_location_required": True,
        "overlapping_spans_rejected": True,
        "scores_discarded": True,
        "gold_used": False,
        "answer_labels_used": False,
        "evaluator_used": False,
        "generation_output_used": False,
    }
    if reader_span_provider != expected_provider:
        raise BoundaryError("config_reader_span_provider_mismatch")
    expected_adapter = {
        "serializer": "frozen_generator_build_prompt",
        "tokenizer_arguments": "frozen_generator_generate",
        "second_serialization_identity_check": True,
        "production_generator_source_modified": False,
        "generation_retry_on_error": False,
        "hook_scope": "one_generate_call",
    }
    if generator_adapter != expected_adapter:
        raise BoundaryError("config_generator_adapter_mismatch")
    if not isinstance(fallback, dict) or fallback.get("policy_id") != "per_question_disabled_zero_anchor" or fallback.get("invalid_row_action") != "disabled_zero_anchor" or any(fallback.get(key) is not False for key in ("drop_invalid_rows", "retry_invalid_rows", "feedback_to_selector")) or fallback.get("unresolved_rows_allowed") is not False:
        raise BoundaryError("config_fallback_contract_mismatch")
    if not isinstance(plan, dict) or plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v2" or plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise BoundaryError("plan_contract_mismatch")
    discovery = plan.get("discovery", {})
    composition = plan.get("composition", {})
    validate_plugin_allowlist(discovery.get("allowlist", ()), composition.get("order", ()))
    plugins = plan.get("plugins")
    if not isinstance(plugins, list) or len(plugins) != len(PLUGIN_ORDER):
        raise BoundaryError("plan_plugin_count_mismatch")
    if tuple(plugin.get("id") for plugin in plugins) != PLUGIN_ORDER:
        raise BoundaryError("plan_plugin_order_mismatch")
    if any(plugin.get("version") != PLUGIN_VERSION for plugin in plugins):
        raise BoundaryError("plan_plugin_version_mismatch")
    if plan.get("reproducibility", {}).get("config_path") != "configs/experiments/generator_native_evidence_anchor_transport_boundary.yaml":
        raise BoundaryError("plan_config_path_mismatch")
    return config, plan


def _synthetic_fixture() -> tuple[tuple[str, ...], dict[str, list[list[int]]], tuple[tuple[tuple[int, int], ...], ...]]:
    prompts = ("alpha beta gamma delta", "one two three four five", "bad input")
    tokenized = {
        "input_ids": [
            [101, 11, 12, 13, 14, 102, 0, 0],
            [0, 0, 201, 21, 22, 23, 24, 25],
            [301, 31, 32, 302, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        "special_tokens_mask": [
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
        ],
        "offset_mapping": [
            [[0, 0], [0, 5], [6, 10], [11, 16], [17, 22], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 3], [4, 7], [8, 13], [14, 18], [19, 23]],
            [[0, 0], [0, 3], [4, 9], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ],
    }
    spans = (((0, 10),), ((0, 3),), ((99, 100),))
    return prompts, tokenized, spans


def run(config_path: Path = DEFAULT_CONFIG, plan_path: Path = DEFAULT_PLAN, output_path: Path | None = None) -> dict[str, Any]:
    config, _ = _load_contract(config_path, plan_path)
    prompts, tokenized, spans = _synthetic_fixture()
    mapping, decisions = build_prompt_token_span_batch_with_fallback(tokenized, prompts, spans, requested_arm="reader")
    assert_prompt_token_identity(mapping, tokenized, prompts)
    manifest = build_compact_boundary_manifest(
        mapping,
        decisions,
        config_sha256=_sha256(config_path),
        code_revision=_git_revision(),
    )
    manifest["checks"] = {
        "explicit_allowlist": "pass" if tuple(manifest["registry"]["allowlist"]) == PLUGIN_ORDER else "fail",
        "batch_mapping": "pass" if mapping.batch_size == 3 and mapping.sequence_length == 8 else "fail",
        "left_right_padding": "pass" if mapping.compact()["padding_sides"] == ["right", "left", "right"] else "fail",
        "geometry_matched_control": "pass" if mapping.geometry_match and mapping.reader_control_disjoint else "fail",
        "prompt_token_identity": "pass",
        "all_question_fallback": "pass" if manifest["accounting"]["unresolved_questions"] == 0 and manifest["accounting"]["emitted_questions"] == manifest["accounting"]["input_questions"] else "fail",
        "compact_privacy": "pass" if not find_forbidden_fields(manifest) else "fail",
    }
    manifest["status"] = "pass" if all(value == "pass" for value in manifest["checks"].values()) else "kill"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
    try:
        if args.validate_only:
            _load_contract(config_path.resolve(), plan_path.resolve())
            print(json.dumps({"status": "valid", "candidate_id": "generator_native_evidence_anchor_transport", "allowlist": list(PLUGIN_ORDER)}, sort_keys=True))
            return 0
        output = args.output
        if output is not None and not output.is_absolute():
            output = PROJECT_ROOT / output
        manifest = run(config_path.resolve(), plan_path.resolve(), output)
        if output is None:
            print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True))
        else:
            print(f"Wrote boundary smoke report: {output}")
        return 0 if manifest["status"] == "pass" else 1
    except (BoundaryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
