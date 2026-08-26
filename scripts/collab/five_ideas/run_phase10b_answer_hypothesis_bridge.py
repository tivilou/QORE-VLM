#!/usr/bin/env python3
"""Validate the planned Phase 10B answer-hypothesis bridge contract.

The runtime is intentionally gated while the screen remains unauthorized. Use
``--validate-only`` to check configuration and plugin composition without
loading Wiki-DPR, a generator, or a GPU model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_ORDER = [
    "frozen_initial_retrieval_observer",
    "consensus_hypothesis_builder",
    "auxiliary_bridge_retrieval",
    "frozen_union_qore_selector",
    "bridge_order_and_privacy_audit",
]
EXPECTED_SLICES = {
    "screen": (2700, 50, "nq_open validation[2700:2750]"),
    "formal": (2750, 200, "nq_open validation[2750:2950]"),
    "replication": (2950, 200, "nq_open validation[2950:3150]"),
}
MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"


class Phase10BConfigError(RuntimeError):
    pass


def project_root() -> Path:
    script = Path(__file__).resolve()
    for candidate in (script.parent, *script.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise Phase10BConfigError("cannot locate project root")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise Phase10BConfigError(f"cannot read config: {exc}") from exc
    phase = document.get("phase")
    if not isinstance(phase, dict):
        raise Phase10BConfigError("config.phase must be a mapping")
    return phase


def load_plan(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase10BConfigError(f"cannot read plan: {exc}") from exc
    if not isinstance(document, dict):
        raise Phase10BConfigError("plan must be a mapping")
    return document


def validate_contract(config_path: Path, plan_path: Path) -> dict[str, Any]:
    phase = load_yaml(config_path)
    plan = load_plan(plan_path)
    if phase.get("name") != "phase10b_answer_hypothesis_bridge":
        raise Phase10BConfigError("unexpected phase name")
    if phase.get("schema_version") != 1:
        raise Phase10BConfigError("schema_version must be 1")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise Phase10BConfigError("Phase 10B must be diagnostic-only and mutation-free")
    if phase.get("authorization") != "planned_observation_only":
        raise Phase10BConfigError("runtime must remain planned_observation_only")

    dataset = phase.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") != "nq_open" or dataset.get("split") != "validation":
        raise Phase10BConfigError("dataset identity is not frozen")
    for stage, (offset, count, label) in EXPECTED_SLICES.items():
        spec = dataset.get(stage)
        if not isinstance(spec, dict):
            raise Phase10BConfigError(f"dataset.{stage} is missing")
        if (int(spec.get("sample_offset", -1)), int(spec.get("max_samples", -1)), spec.get("slice"), spec.get("fresh_slice")) != (offset, count, label, True):
            raise Phase10BConfigError(f"dataset.{stage} slice is not frozen")

    retrieval = phase.get("retrieval")
    if retrieval != {
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "initial_top_k": 50,
        "bridge_top_k": 50,
        "union_max_candidates": 100,
    }:
        raise Phase10BConfigError("retrieval contract is not frozen")

    selection = phase.get("selection")
    expected_selection = {
        "method": "qore", "K": 5, "num_reads": 100, "lam": 2.0,
        "seed": 42, "gamma": 1.0, "delta": 0.0,
        "complementarity_method": None, "qore_prefilter_size": None,
        "direct_solve_max_n": 20, "use_answer_scorer": True,
        "answer_scorer_backend": "dpr",
    }
    if selection != expected_selection:
        raise Phase10BConfigError("selection contract is not frozen")

    generator = phase.get("generator")
    if not isinstance(generator, dict) or generator.get("model_id") != MODEL_ID or generator.get("revision") != MODEL_REVISION or int(generator.get("max_new_tokens", -1)) != 32 or generator.get("decoding") != "greedy":
        raise Phase10BConfigError("generator contract is not frozen")

    bridge = phase.get("bridge")
    expected_bridge = {
        "plugin_id": "answer_hypothesis_evidence_bridge",
        "min_support": 2,
        "min_probability": 0.2,
        "query_template": "question + answer_hypothesis",
        "decision_order": "pre_generation_only",
        "gold_or_evaluator_available": False,
        "generated_answer_available": False,
        "qore_feedback": False,
        "final_context_K": 5,
    }
    if bridge != expected_bridge:
        raise Phase10BConfigError("bridge contract is not frozen")

    if phase.get("arms") != ["baseline_frozen_query", "always_bridge", "consensus_gated_bridge"]:
        raise Phase10BConfigError("arm order is not frozen")
    outputs = phase.get("outputs") or {}
    forbidden = outputs.get("forbidden_fields")
    expected_forbidden = ["question", "passages", "gold_answers", "prediction", "candidate_text", "bridge_query", "raw_prompt", "evaluator_trace"]
    if outputs.get("compact_only") is not True or outputs.get("bridge_query_persisted") is not False or outputs.get("candidate_text_persisted") is not False or forbidden != expected_forbidden:
        raise Phase10BConfigError("output privacy contract is not frozen")

    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1":
        raise Phase10BConfigError("unexpected plan schema")
    if plan.get("authorization") != "planned_observation_only":
        raise Phase10BConfigError("plan authorization is not planned_observation_only")
    plugins = plan.get("plugins")
    if not isinstance(plugins, list) or [item.get("id") for item in plugins] != EXPECTED_ORDER:
        raise Phase10BConfigError("plugin order is not frozen")
    discovery = plan.get("discovery") or {}
    composition = plan.get("composition") or {}
    if discovery.get("allowlist") != EXPECTED_ORDER or composition.get("order") != EXPECTED_ORDER:
        raise Phase10BConfigError("allowlist/composition mismatch")
    if composition.get("mode") != "sequential":
        raise Phase10BConfigError("composition must be sequential")
    return {
        "status": "valid",
        "phase": phase["name"],
        "authorization": phase["authorization"],
        "dataset": {stage: spec["slice"] for stage, spec in ((name, dataset[name]) for name in EXPECTED_SLICES)},
        "plugins": EXPECTED_ORDER,
        "selection_mutation": False,
        "report_only": True,
        "wiki_dpr_started": False,
        "model_loaded": False,
    }


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/phase10b_answer_hypothesis_bridge.yaml")
    parser.add_argument("--plan", type=Path, default=root / "configs/experiments/phase10b_answer_hypothesis_bridge_plan.json")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else root / args.config
    plan = args.plan if args.plan.is_absolute() else root / args.plan
    try:
        result = validate_contract(config.resolve(), plan.resolve())
        if not args.validate_only:
            raise Phase10BConfigError("Phase 10B screen is not authorized; use --validate-only")
    except (Phase10BConfigError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
