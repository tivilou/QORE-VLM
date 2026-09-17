#!/usr/bin/env python3
"""Run the fixed 100-case Top-50 -> Top-5 TREG/ECAD replay.

The runner reuses the two registered 50-question case studies and their
rank-aligned Silver vote bundle.  It never reruns retrieval and it never sends
Silver/gold fields to a selector.  Missing signal artifacts are built locally
with the fixed DPR Reader and NLI model, then the two new selectors are
compared with the registered QORE and Top-k memberships.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


_SCRIPT_PATH = Path(__file__).resolve()


def _root() -> Path:
    for candidate in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
        if (candidate / "configs").is_dir() and (candidate / "applications").is_dir():
            return candidate
    raise RuntimeError("cannot locate project root")


ROOT = _root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_COMPACT_BYTES = 1_048_576
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
K = 5
EXPECTED_SELECTORS = ("qore_as", "topk_as", "treg_reader_gain", "ecad_conflict_aware")
FORBIDDEN_COMPACT_FIELDS = frozenset(
    {
        "question",
        "passages",
        "gold_answers",
        "prediction",
        "raw_prompt",
        "prompt",
        "text",
        "token_ids",
        "selected_ids",
        "retrieved_ids",
        "evidence",
    }
)


class ReplayError(RuntimeError):
    """Raised when the replay contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False, timeout=3
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_branch() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as exc:
        raise ReplayError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"YAML root must be a mapping: {path}")
    return value


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"ensure_ascii": False}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2
    path.write_text(json.dumps(value, **kwargs) + "\n", encoding="utf-8")


def _validate_plan(plan_path: Path) -> None:
    plan = _load_json(plan_path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v1":
        raise ReplayError("unexpected TREG/ECAD plugin plan schema")
    if plan.get("project") != "Q-DUET-VLM" or plan.get("authorization") != "implemented":
        raise ReplayError("TREG/ECAD plugin plan is not implemented")
    expected = [
        "registered_100_case_loader",
        "treg_fixed_reader_gain",
        "ecad_pairwise_nli",
        "selector_only_comparator",
        "compact_replay_renderer",
    ]
    if plan.get("discovery", {}).get("mode") != "explicit_allowlist":
        raise ReplayError("plugin discovery must be explicit")
    if plan.get("discovery", {}).get("allowlist") != expected:
        raise ReplayError("plugin allowlist changed")
    if plan.get("composition", {}).get("mode") != "sequential" or plan.get("composition", {}).get("order") != expected:
        raise ReplayError("plugin composition order changed")


def validate_contract(config_path: Path, plan_path: Path) -> dict[str, Any]:
    """Validate the local replay contract without loading models or data."""

    document = _load_yaml(config_path)
    phase = document.get("phase")
    if not isinstance(phase, Mapping):
        raise ReplayError("phase section is missing")
    if phase.get("name") != "selector_replay_100_treg_ecad" or int(phase.get("schema_version", -1)) != 1:
        raise ReplayError("unexpected replay identity")
    if phase.get("evidence_tier") != "L0_diagnostic" or phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise ReplayError("replay must remain diagnostic and selection-only")
    base_config = Path(str(phase.get("base_case_config", "")))
    base_plan = Path(str(phase.get("base_case_plan", "")))
    base_config = base_config if base_config.is_absolute() else ROOT / base_config
    base_plan = base_plan if base_plan.is_absolute() else ROOT / base_plan
    try:
        from scripts.collab.five_ideas.run_silver_oracle_top5_100 import validate_contract as validate_base

        base_contract = validate_base(base_config.resolve(), base_plan.resolve())
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise ReplayError(f"registered 100-case base contract failed: {exc}") from exc
    if base_contract.get("case_count") != EXPECTED_CASES:
        raise ReplayError("base contract does not expose 100 cases")
    selectors = phase.get("selectors")
    if not isinstance(selectors, list) or tuple(item.get("id") for item in selectors if isinstance(item, Mapping)) != EXPECTED_SELECTORS:
        raise ReplayError("selector allowlist/order changed")
    treg = phase.get("treg")
    if not isinstance(treg, Mapping) or treg.get("reader_model_id") != "facebook/dpr-reader-single-nq-base" or treg.get("reader_revision") != "38f47a4986084c53447ba92ab0a83076b58d86a8":
        raise ReplayError("TREG Reader identity changed")
    if int(treg.get("k", -1)) != K or int(treg.get("b4_size", -1)) != 4 or treg.get("gain_definition") != "Reader(q,B4+p)-Reader(q,B4)":
        raise ReplayError("TREG protocol changed")
    ecad = phase.get("ecad")
    if not isinstance(ecad, Mapping) or ecad.get("nli_model_id") != "cross-encoder/nli-MiniLM2-L6-H768" or ecad.get("nli_revision") != "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d":
        raise ReplayError("ECAD NLI identity changed")
    if int(ecad.get("k", -1)) != K or int(ecad.get("pair_count_per_case", -1)) != 1225 or ecad.get("pair_order") != "top50_position_i_lt_j":
        raise ReplayError("ECAD protocol changed")
    outputs = phase.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("root") != "exchange/five_ideas/selector_replay_100_treg_ecad":
        raise ReplayError("output root changed")
    if int(outputs.get("github_max_bytes", -1)) != MAX_COMPACT_BYTES:
        raise ReplayError("GitHub compact threshold changed")
    if set(outputs.get("forbidden_compact_fields", ())) != FORBIDDEN_COMPACT_FIELDS:
        raise ReplayError("compact privacy field set changed")
    _validate_plan(plan_path)
    return {
        "status": "valid",
        "case_count": EXPECTED_CASES,
        "top50_count": EXPECTED_TOP50,
        "selectors": list(EXPECTED_SELECTORS),
        "base_config": str(base_config.resolve()),
        "base_plan": str(base_plan.resolve()),
        "model_loaded": False,
        "wiki_dpr_started": False,
    }


def _direct_cases(path: Path) -> list[dict[str, Any]]:
    """Load a complete 100-case detail file when the collaborator has one."""

    root = _load_json(path)
    cases = root.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ReplayError("--input must contain exactly 100 cases")
    normalized: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise ReplayError(f"input case {number} is invalid")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise ReplayError(f"input case {number} must contain Top-50")
        selectors = case.get("selectors")
        by_selector = {
            str(item.get("selector_id")): item
            for item in selectors or []
            if isinstance(item, Mapping)
        }
        memberships: dict[str, list[str]] = {}
        for selector_id in ("qore_as", "topk_as"):
            selected = by_selector.get(selector_id, {}).get("selected_top_5")
            if not isinstance(selected, list) or len(selected) != K:
                raise ReplayError(f"input case {number} lacks {selector_id}")
            memberships[selector_id] = [str(item.get("id")) for item in selected if isinstance(item, Mapping)]
        if any(not isinstance(item, Mapping) or not isinstance(item.get("evidence"), Mapping) for item in top50):
            raise ReplayError("--input must carry the registered Silver evidence fields")
        normalized.append(
            {
                "source_id": str(case.get("source_id", "direct_input")),
                "source_case_number": int(case.get("case_number", number)),
                "global_index": case.get("dataset_position", number - 1),
                "question_id": str(case.get("question_id", f"case_{number}")),
                "question": str(case.get("question", "")),
                "gold_answers": [str(value) for value in case.get("gold_answers", [])],
                "top_50": [dict(item) for item in top50],
                "baseline_memberships": memberships,
            }
        )
    return normalized


def _registered_cases(base_config: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.collab.five_ideas.run_silver_oracle_top5_100 import load_registered_cases

    base_phase = _load_yaml(base_config)["phase"]
    cases, provenance, _ = load_registered_cases(ROOT, base_phase, args)
    if len(cases) != EXPECTED_CASES:
        raise ReplayError("registered loader did not return 100 cases")
    return cases, provenance


def _online(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for item in case["top_50"]:
        if not isinstance(item, Mapping):
            raise ReplayError("Top-50 candidate is not an object")
        output.append(
            {
                "id": str(item.get("id", "")),
                "title": str(item.get("title") or ""),
                "text": str(item.get("text") or item.get("passage") or ""),
                "retrieved_rank": int(item.get("retrieved_rank", -1)),
                "retrieval_score": float(item.get("retrieval_score", 0.0)),
                "answer_scorer_score": float(item.get("answer_scorer_score")),
            }
        )
    if len(output) != EXPECTED_TOP50 or len({row["id"] for row in output}) != EXPECTED_TOP50:
        raise ReplayError("Top-50 candidate identity is invalid")
    return output


def _ids_to_indices(online: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> list[int]:
    positions = {str(row["id"]): index for index, row in enumerate(online)}
    if len(ids) != K or len(set(ids)) != K or any(identifier not in positions for identifier in ids):
        raise ReplayError("stored baseline membership is invalid")
    selected = [positions[str(identifier)] for identifier in ids]
    return sorted(selected, key=lambda index: (int(online[index]["retrieved_rank"]), str(online[index]["id"])))


def _token_set(value: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _redundancy(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    texts = [f"{row.get('title', '')}. {row.get('text', '')}" for row in selected]
    sets = [_token_set(text) for text in texts]
    pairs = [_jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    titles = [str(row.get("title") or "") for row in selected]
    title_pairs = [titles[i] == titles[j] and bool(titles[i]) for i in range(len(titles)) for j in range(i + 1, len(titles))]
    return {
        "token_jaccard_mean": sum(pairs) / len(pairs) if pairs else 0.0,
        "same_title_pair_fraction": sum(title_pairs) / len(title_pairs) if title_pairs else 0.0,
        "unique_title_count": len(set(titles)),
    }


def _evidence_flags(item: Mapping[str, Any]) -> tuple[bool, bool]:
    evidence = item.get("evidence")
    if not isinstance(evidence, Mapping):
        return False, False
    broad = evidence.get("positive_consensus")
    direct = evidence.get("direct_consensus")
    if not isinstance(broad, bool):
        broad = evidence.get("consensus_label") in {"direct", "partial"}
    if not isinstance(direct, bool):
        direct = evidence.get("consensus_label") == "direct"
    return bool(broad), bool(direct)


def _set_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in ids)).encode("utf-8")).hexdigest()


def _case_row(
    case_number: int,
    case: Mapping[str, Any],
    online: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    oracle_ids: set[str],
    *,
    selection_time_ms: float,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    selected = [online[index] for index in selected_indices]
    selected_ids = [str(item["id"]) for item in selected]
    broad = {str(item["id"]): _evidence_flags(item)[0] for item in case["top_50"]}
    direct = {str(item["id"]): _evidence_flags(item)[1] for item in case["top_50"]}
    overlap = len(set(selected_ids) & oracle_ids)
    precision = overlap / K
    recall = overlap / len(oracle_ids) if oracle_ids else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0 if oracle_ids else None
    return {
        "case_number": case_number,
        "selected_set_sha256": _set_hash(selected_ids),
        "selected_ranks": [int(item["retrieved_rank"]) for item in selected],
        "selected_answer_scorer_scores": [round(float(item["answer_scorer_score"]), 8) for item in selected],
        "top50_broad_positive_count": sum(broad.values()),
        "top50_direct_positive_count": sum(direct.values()),
        "selected_broad_positive_count": sum(broad.get(identifier, False) for identifier in selected_ids),
        "selected_direct_positive_count": sum(direct.get(identifier, False) for identifier in selected_ids),
        "retrieval_miss": sum(broad.values()) == 0,
        "selector_miss": sum(broad.get(identifier, False) for identifier in selected_ids) == 0 and sum(broad.values()) > 0,
        "silver_oracle_overlap_count": overlap,
        "silver_oracle_precision": precision,
        "silver_oracle_recall": recall,
        "silver_oracle_set_f1": f1,
        "redundancy": _redundancy(selected),
        "selection_time_ms": round(float(selection_time_ms), 6),
        "selector_diagnostics": dict(diagnostics),
    }


def _nested(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _summary(method: str, rows: Sequence[Mapping[str, Any]], status: str) -> dict[str, Any]:
    def mean(path: str) -> float | None:
        values = [float(value) for row in rows if (value := _nested(row, path)) is not None]
        return sum(values) / len(values) if values else None

    def median(path: str) -> float | None:
        values = [float(value) for row in rows if (value := _nested(row, path)) is not None]
        return statistics.median(values) if values else None

    overlaps = [int(row["silver_oracle_overlap_count"]) for row in rows]
    broad = [int(row["selected_broad_positive_count"]) for row in rows]
    direct = [int(row["selected_direct_positive_count"]) for row in rows]
    top_broad = sum(int(row["top50_broad_positive_count"]) for row in rows)
    top_direct = sum(int(row["top50_direct_positive_count"]) for row in rows)
    return {
        "method": method,
        "status": status,
        "case_count": len(rows),
        "mean_silver_oracle_overlap": mean("silver_oracle_overlap_count"),
        "median_silver_oracle_overlap": median("silver_oracle_overlap_count"),
        "mean_silver_oracle_precision": mean("silver_oracle_precision"),
        "mean_silver_oracle_recall": mean("silver_oracle_recall"),
        "mean_silver_oracle_set_f1": mean("silver_oracle_set_f1"),
        "mean_selected_broad_positive_count": mean("selected_broad_positive_count"),
        "mean_selected_direct_positive_count": mean("selected_direct_positive_count"),
        "broad_passage_retention": sum(broad) / top_broad if top_broad else None,
        "direct_passage_retention": sum(direct) / top_direct if top_direct else None,
        "retrieval_miss_cases": sum(bool(row["retrieval_miss"]) for row in rows),
        "selector_miss_cases": sum(bool(row["selector_miss"]) for row in rows),
        "mean_token_jaccard_redundancy": mean("redundancy.token_jaccard_mean"),
        "mean_same_title_pair_fraction": mean("redundancy.same_title_pair_fraction"),
        "mean_unique_title_count": mean("redundancy.unique_title_count"),
        "mean_selection_time_ms": mean("selection_time_ms"),
        "median_selection_time_ms": median("selection_time_ms"),
        "max_selection_time_ms": max((float(row["selection_time_ms"]) for row in rows), default=0.0),
        "overlap_distribution": dict(sorted(Counter(overlaps).items())),
    }


def _paired(rows: Mapping[str, Sequence[Mapping[str, Any]]], left: str, right: str) -> dict[str, int]:
    left_map = {int(row["case_number"]): row for row in rows[left]}
    right_map = {int(row["case_number"]): row for row in rows[right]}
    values = [(left_map[number]["silver_oracle_overlap_count"], right_map[number]["silver_oracle_overlap_count"]) for number in sorted(set(left_map) & set(right_map))]
    return {
        "left_wins": sum(left_value > right_value for left_value, right_value in values),
        "ties": sum(left_value == right_value for left_value, right_value in values),
        "right_wins": sum(left_value < right_value for left_value, right_value in values),
    }


def _forbidden(value: Any, path: str = "$root") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_COMPACT_FIELDS:
                found.append(f"{path}.{key}")
            found.extend(_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden(child, f"{path}[{index}]"))
    return found


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TREG and ECAD: 100-case Top-50 to Top-5 replay",
        "",
        "This is an L0 selector-alignment replay on a fixed Top-50. Silver oracle labels are read only after selection and cannot establish Generator utility or L1/L2 evidence.",
        "",
        f"- Cases: {report['protocol']['case_count']} x Top-{report['protocol']['top50_count']}; K={report['protocol']['k']}",
        f"- Registered input vote bundle: `{report['input']['vote_bundle_sha256']}`",
        f"- Code revision: `{report['provenance'].get('git_revision') or 'unavailable'}`",
        "",
        "## Selector comparison",
        "",
        "| Method | Silver overlap | Silver set-F1 | Broad retained | Direct retained | Selector miss | Token redundancy | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, summary in report["summaries"].items():
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {method} | {fmt(summary['mean_silver_oracle_overlap'])} | {fmt(summary['mean_silver_oracle_set_f1'])} | "
            f"{fmt(summary['broad_passage_retention'])} | {fmt(summary['direct_passage_retention'])} | "
            f"{summary['selector_miss_cases']} | {fmt(summary['mean_token_jaccard_redundancy'])} | {fmt(summary['median_selection_time_ms'])} |"
        )
    lines.extend(["", "## Paired Silver-overlap comparison", "", "| Method | vs QORE win/tie/loss | vs Top-k win/tie/loss |", "|---|---:|---:|"])
    for method, comparison in report["paired_comparisons"].items():
        lines.append(
            f"| {method} | {comparison['vs_qore']['left_wins']}/{comparison['vs_qore']['ties']}/{comparison['vs_qore']['right_wins']} | "
            f"{comparison['vs_topk']['left_wins']}/{comparison['vs_topk']['ties']}/{comparison['vs_topk']['right_wins']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Retrieval miss cases: {report['diagnostics']['retrieval_miss_cases']}/{report['protocol']['case_count']}; those cases are not selector failures.",
            "- TREG uses a fixed DPR Reader and B4 conditional gain. ECAD uses complete provenance-locked pairwise NLI scores.",
            "- A positive Silver-alignment result is only a pre-generation signal for a later frozen-Generator screen; it is not a deployment or paper claim.",
            "",
            report["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def _write_manifest(
    output_dir: Path,
    report: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> None:
    exchange_target = f"five_ideas/selector_replay_100_treg_ecad/{output_dir.name}"
    exchange_files = ["treg_gain_artifact.json", "ecad_pairwise_nli.json", "report.md"]
    compact_files = ["summary.json", "run_metadata.json", "upload_manifest.json"]
    files = []
    # A manifest cannot include its own digest without a circular value.  It
    # remains listed in ``compact_files`` as the manifest itself; the file
    # inventory covers every other exchange/compact artifact.
    for name in exchange_files + [value for value in compact_files if value != "upload_manifest.json"]:
        path = output_dir / name
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path), "exchange_path": f"{exchange_target}/{name}"})
    manifest = {
        "schema_version": 1,
        "artifact_type": "selector_replay_100_treg_ecad_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "diagnostic_only": True,
        "target_directory": exchange_target,
        "github_allowed_if_unchanged": compact_files,
        "exchange_required_files": exchange_files,
        "files": files,
        "publication_rule": "Compact GitHub files must be <=1 MiB and contain no raw question, passage, answer, evidence, prediction, or selected/retrieved ids.",
        "provenance": report["provenance"],
    }
    if _forbidden(manifest):
        raise ReplayError("upload manifest contains forbidden compact fields")
    _write_json(output_dir / "upload_manifest.json", manifest)


def run(args: argparse.Namespace) -> Path | None:
    started = time.perf_counter()
    config_path = (args.config if args.config.is_absolute() else ROOT / args.config).resolve()
    plan_path = (args.plan if args.plan.is_absolute() else ROOT / args.plan).resolve()
    contract = validate_contract(config_path, plan_path)
    if args.validate_only:
        print(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        return None
    phase = _load_yaml(config_path)["phase"]
    if args.input:
        input_path = args.input.resolve()
        cases = _direct_cases(input_path)
        source_provenance = [{"mode": "direct_input", "path_name": input_path.name, "sha256": _sha256(input_path)}]
        vote_bundle_sha256 = "direct_input"
    else:
        base_config = ROOT / str(phase["base_case_config"])
        cases, source_provenance = _registered_cases(base_config, args)
        vote_bundle_sha256 = "unknown"
        for item in source_provenance:
            bundle = item.get("compact_vote_bundle")
            if isinstance(bundle, Mapping) and bundle.get("sha256"):
                vote_bundle_sha256 = str(bundle["sha256"])
    if len(cases) != EXPECTED_CASES:
        raise ReplayError("exactly 100 cases are required")

    # Keep post-hoc evidence labels in the diagnostic view only.  Signal
    # builders and validators receive the same minimal online contract as the
    # selectors, so evidence cannot leak into TREG or ECAD inputs.
    online_cases = [
        {
            "question": str(case.get("question", "")),
            "question_id": str(case.get("question_id", "")),
            "top_50": _online(case),
        }
        for case in cases
    ]

    treg_artifact = _load_json(args.treg_artifact.resolve()) if args.treg_artifact else None
    ecad_artifact = _load_json(args.ecad_artifact.resolve()) if args.ecad_artifact else None
    from applications.rag.ecad_selector import (
        FixedNLIModel,
        NLI_MODEL_ID,
        NLI_REVISION,
        build_pairwise_artifact,
        select as ecad_select,
        validate_pairwise_artifact,
    )
    from applications.rag.treg_selector import (
        FixedDPRReader,
        READER_MODEL_ID,
        READER_REVISION,
        build_gain_artifact,
        select as treg_select,
        validate_gain_artifact,
    )
    treg_cfg = phase["treg"]
    ecad_cfg = phase["ecad"]
    if treg_artifact is None:
        reader = FixedDPRReader(
            model_id=str(treg_cfg["reader_model_id"]),
            revision=str(treg_cfg["reader_revision"]),
            device=args.device,
            batch_size=int(treg_cfg["batch_size"]),
            passage_token_budget=int(treg_cfg["passage_token_budget"]),
            max_length=int(treg_cfg["reader_max_length"]),
        )
        treg_artifact = build_gain_artifact(online_cases, reader, k=K, b4_size=4)
        del reader
    treg_maps = validate_gain_artifact(treg_artifact, online_cases, expected_reader_model_id=READER_MODEL_ID, expected_reader_revision=READER_REVISION, k=K, b4_size=4)
    if ecad_artifact is None:
        nli = FixedNLIModel(
            model_id=str(ecad_cfg["nli_model_id"]),
            revision=str(ecad_cfg["nli_revision"]),
            device=args.device,
            batch_size=int(ecad_cfg["batch_size"]),
            max_length=int(ecad_cfg["max_length"]),
        )
        ecad_artifact = build_pairwise_artifact(online_cases, nli)
        del nli
    ecad_maps = validate_pairwise_artifact(ecad_artifact, online_cases, expected_model_id=NLI_MODEL_ID, expected_revision=NLI_REVISION)

    from applications.rag.silver_oracle_selector import select as oracle_select

    method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in EXPECTED_SELECTORS}
    output_cases: list[dict[str, Any]] = []
    retrieval_miss_cases = 0
    for case_number, case in enumerate(cases, start=1):
        online = _online(case)
        top50 = case["top_50"]
        oracle_indices = oracle_select(top50, k=K).selected_indices
        oracle_ids = {str(top50[index]["id"]) for index in oracle_indices}
        retrieval_miss_cases += int(not any(_evidence_flags(item)[0] for item in top50))
        selections: list[tuple[str, list[int], dict[str, Any]]] = []
        for method, membership_key in (("qore_as", "qore_as"), ("topk_as", "topk_as")):
            indices = _ids_to_indices(online, case["baseline_memberships"][membership_key])
            selections.append((method, indices, {"source": "registered_membership"}))
        begin = time.perf_counter()
        treg_result = treg_select(online, treg_maps[case_number], k=K, b4_size=4)
        selections.append(("treg_reader_gain", list(treg_result.selected_indices), {"base_size": len(treg_result.base_indices), "residual_count": len(treg_result.residual_indices), "selected_gain_mean": sum(treg_result.gains) / len(treg_result.gains) if treg_result.gains else 0.0, "trace_length": len(treg_result.trace)}))
        treg_ms = (time.perf_counter() - begin) * 1000.0
        begin = time.perf_counter()
        ecad_result = ecad_select(online, ecad_maps[case_number], k=K, conflict_penalty=float(ecad_cfg["conflict_penalty"]))
        selections.append(("ecad_conflict_aware", list(ecad_result.selected_indices), {"conflict_penalty": float(ecad_cfg["conflict_penalty"]), "trace_length": len(ecad_result.trace)}))
        ecad_ms = (time.perf_counter() - begin) * 1000.0
        case_methods: dict[str, dict[str, Any]] = {}
        for method, indices, diagnostics in selections:
            elapsed = 0.0 if method in {"qore_as", "topk_as"} else treg_ms if method == "treg_reader_gain" else ecad_ms
            row = _case_row(case_number, case, online, indices, oracle_ids, selection_time_ms=elapsed, diagnostics=diagnostics)
            method_rows[method].append(row)
            case_methods[method] = row
        case_methods_compact = {method: dict(row) for method, row in case_methods.items()}
        output_cases.append({"case_number": case_number, "source_id": str(case.get("source_id", "")), "question_id_sha256": hashlib.sha256(str(case.get("question_id", "")).encode("utf-8")).hexdigest(), "methods": case_methods_compact})
        if args.progress and (case_number == EXPECTED_CASES or case_number % 10 == 0):
            print(f"  selector replay: {case_number}/{EXPECTED_CASES}", flush=True)

    summaries = {method: _summary(method, rows, "registered_baseline" if method in {"qore_as", "topk_as"} else "replayed") for method, rows in method_rows.items()}
    paired = {method: {"vs_qore": _paired(method_rows, method, "qore_as"), "vs_topk": _paired(method_rows, method, "topk_as")} for method in EXPECTED_SELECTORS if method not in {"qore_as", "topk_as"}}
    provenance = {"git_revision": _git_revision(), "script": str(_SCRIPT_PATH.relative_to(ROOT)), "script_sha256": _sha256(_SCRIPT_PATH), "source_provenance": source_provenance}
    report: dict[str, Any] = {
        "schema_version": "rag.selector_replay_100_treg_ecad.v1",
        "artifact_type": "selector_only_replay_benchmark",
        "diagnostic_only": True,
        "input": {"case_count": EXPECTED_CASES, "top50_count": EXPECTED_TOP50, "vote_bundle_sha256": vote_bundle_sha256},
        "protocol": {"case_count": EXPECTED_CASES, "top50_count": EXPECTED_TOP50, "k": K, "silver_oracle_used_for_selection": False, "treg_gain_definition": "Reader(q,B4+p)-Reader(q,B4)", "ecad_pair_count_per_case": 1225, "generator_rerun": False},
        "provenance": provenance,
        "treg": {"model_id": READER_MODEL_ID, "revision": READER_REVISION, "artifact_schema": treg_artifact["schema_version"]},
        "ecad": {"model_id": NLI_MODEL_ID, "revision": NLI_REVISION, "artifact_schema": ecad_artifact["schema_version"]},
        "summaries": summaries,
        "paired_comparisons": paired,
        "diagnostics": {"retrieval_miss_cases": retrieval_miss_cases, "case_count_validated": len(output_cases), "treg_gain_case_count": len(treg_maps), "ecad_pair_count_total": sum(len(value) for value in ecad_maps.values())},
        "cases": output_cases,
        "next_step": "若 TREG 或 ECAD 在相同 100 题上同时改善 Silver 对齐、证据保留且不过度增加冗余，才授权下一步冻结 Generator 的 paired screen；否则按逐题失败模式关闭该方向。",
    }
    if _forbidden(report):
        raise ReplayError("compact summary contains forbidden fields")
    output_root = args.output_root if args.output_root is not None else Path(str(phase["outputs"]["root"]))
    output_root = output_root if output_root.is_absolute() else ROOT / output_root
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{timestamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "treg_gain_artifact.json", treg_artifact, compact=True)
    _write_json(output_dir / "ecad_pairwise_nli.json", ecad_artifact, compact=True)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_json(output_dir / "summary.json", report)
    if (output_dir / "summary.json").stat().st_size > MAX_COMPACT_BYTES:
        raise ReplayError("summary.json exceeds the 1 MiB GitHub threshold")
    metadata = {
        "schema_version": 1,
        "artifact_type": "selector_replay_100_treg_ecad_run_metadata",
        "status": "completed",
        "diagnostic_only": True,
        "git": {"commit": provenance["git_revision"], "branch": _git_branch()},
        "config": {"path_name": config_path.name, "sha256": _sha256(config_path)},
        "plugin_plan": {"path_name": plan_path.name, "sha256": _sha256(plan_path)},
        "reader": {"model_id": READER_MODEL_ID, "revision": READER_REVISION, "config_sha256": treg_artifact["provenance"].get("config_sha256")},
        "nli": {"model_id": NLI_MODEL_ID, "revision": NLI_REVISION, "config_sha256": ecad_artifact["provenance"].get("config_sha256")},
        "source_provenance": source_provenance,
        "outputs": {"root": f"five_ideas/selector_replay_100_treg_ecad/{output_dir.name}", "compact_files": ["summary.json", "run_metadata.json", "upload_manifest.json"], "exchange_files": ["treg_gain_artifact.json", "ecad_pairwise_nli.json", "report.md"]},
        "timing_ms": {"total": (time.perf_counter() - started) * 1000.0},
    }
    if _forbidden(metadata):
        raise ReplayError("run metadata contains forbidden fields")
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_manifest(output_dir, report, phase)
    print(f"Completed: {output_dir}")
    print(f"Exchange target: five_ideas/selector_replay_100_treg_ecad/{output_dir.name}")
    print("Compact files: summary.json run_metadata.json upload_manifest.json")
    print("Exchange files: treg_gain_artifact.json ecad_pairwise_nli.json report.md")
    return output_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/selector_replay_100_treg_ecad.yaml")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/experiments/selector_replay_100_treg_ecad_plan.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--input", type=Path, help="optional complete 100-case detail JSON; default uses registered 50+50 sources")
    parser.add_argument("--treg-artifact", type=Path)
    parser.add_argument("--ecad-artifact", type=Path)
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--detail-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--detail-source-b", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-source-b", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--panel-source-a", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--panel-source-b", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (ReplayError, OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
