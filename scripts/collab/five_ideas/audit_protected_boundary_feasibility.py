#!/usr/bin/env python3
"""Audit whether the protected-boundary screen could reach its training targets.

This is a deterministic, post-hoc audit of an already validated screen.  It
does not load a model, use labels beyond the frozen training targets, or tune a
selector.  The output intentionally contains question hashes and candidate
indices only; passage text, questions, answers, and annotations are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_INPUT = Path(".cache/protected_boundary_result/20261012T151854474980Z/case_study.json")
DEFAULT_OUTPUT = Path(".cache/protected_boundary_feasibility_audit/20261013T000000Z")
METHODS = ("frozen_reader_topk", "quantum_semantic", "classical_semantic")
LEARNED_METHODS = ("quantum_semantic", "classical_semantic")
EPS = 1e-9
NEAR_TIE_STD_FRACTION = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("case study must contain an object-valued cases list")
    return value


def finite_vector(record: Mapping[str, Any], method: str) -> np.ndarray:
    values = np.asarray(record["methods"][method]["score_vector"], dtype=np.float64)
    if values.shape != (50,) or not np.isfinite(values).all():
        raise ValueError(f"{method}: expected 50 finite scores")
    return values


def stable_selected(record: Mapping[str, Any], method: str) -> list[int]:
    selected = record["methods"][method]["selected_indices"]
    if not isinstance(selected, list) or len(selected) != 5:
        raise ValueError(f"{method}: expected five selected indices")
    return [int(index) for index in selected]


def _case_std(base: np.ndarray) -> float:
    return max(float(np.std(base)), 1e-6)


def _required_positive_uplift(base: np.ndarray, positive: int, selected: Sequence[int]) -> float:
    """Continuous uplift needed for a missed item to reach the baseline fifth.

    This is a lower bound: it changes only the positive score and leaves the
    other 49 scores fixed.  A tiny epsilon makes the quantity a crossing
    requirement rather than a tie requirement.  The exact stable tie order is
    reported separately by the screen and is not used to tune this audit.
    """
    if positive in selected:
        return 0.0
    threshold = float(np.min(base[list(selected)]))
    return max(0.0, threshold - float(base[positive]) + EPS)


def _pair_required_gap(base: np.ndarray, positive: int, negative: int) -> float:
    return max(0.0, float(base[negative]) - float(base[positive]) + EPS)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(array.mean()),
            "median": float(np.median(array)), "min": float(array.min()),
            "max": float(array.max())}


def _crosses_pair(scores: np.ndarray, positive: int, negative: int) -> bool:
    # A score tie is conservatively not called a rescue because stable order is
    # dependent on the candidate metadata, which is deliberately omitted here.
    return bool(scores[positive] > scores[negative] + EPS)


def _audit_training(rows: Sequence[Mapping[str, Any]], residual_bound: float) -> dict[str, Any]:
    pair_count = 0
    method_results: dict[str, Any] = {}
    for method in LEARNED_METHODS:
        pair_records: list[dict[str, Any]] = []
        counts = {"structurally_unreachable": 0, "reachable_but_not_learned": 0,
                  "observed_pair_rescue": 0, "harmful": 0, "near_tie": 0}
        required_pair: list[float] = []
        required_boundary: list[float] = []
        observed_pair: list[float] = []
        available_relative: list[float] = []
        observed_boundary: list[float] = []
        for row in rows:
            target = row["training_target"]
            base = finite_vector(row, "frozen_reader_topk")
            scores = finite_vector(row, method)
            selected = stable_selected(row, "frozen_reader_topk")
            spread = _case_std(base)
            theoretical_relative_capacity = 2.0 * float(residual_bound) * spread
            delta = scores - base
            case_hash = str(target.get("question_sha256", ""))
            for positive, negative in target["draft_pairs"]:
                positive, negative = int(positive), int(negative)
                req_pair = _pair_required_gap(base, positive, negative)
                req_boundary = _required_positive_uplift(base, positive, selected)
                obs_pair = float(delta[positive] - delta[negative])
                pair_crossed = _crosses_pair(scores, positive, negative)
                structurally_unreachable = req_pair > theoretical_relative_capacity + EPS
                near_tie = req_pair <= NEAR_TIE_STD_FRACTION * spread
                harmful = obs_pair < -EPS
                if structurally_unreachable:
                    counts["structurally_unreachable"] += 1
                elif not pair_crossed:
                    counts["reachable_but_not_learned"] += 1
                if pair_crossed:
                    counts["observed_pair_rescue"] += 1
                if harmful:
                    counts["harmful"] += 1
                if near_tie:
                    counts["near_tie"] += 1
                pair_records.append({
                    "question_sha256": case_hash,
                    "positive_index": positive,
                    "negative_index": negative,
                    "required_pair_relative_uplift": req_pair,
                    "required_boundary_uplift": req_boundary,
                    "observed_pair_relative_uplift": obs_pair,
                    "theoretical_pair_capacity": theoretical_relative_capacity,
                    "observed_positive_correction": float(delta[positive]),
                    "observed_negative_correction": float(delta[negative]),
                    "observed_pair_rescue": pair_crossed,
                    "structurally_unreachable": structurally_unreachable,
                    "near_tie": near_tie,
                    "harmful": harmful,
                })
                required_pair.append(req_pair)
                required_boundary.append(req_boundary)
                observed_pair.append(obs_pair)
                available_relative.append(theoretical_relative_capacity)
                observed_boundary.append(float(delta[positive]))
        pair_count = len(pair_records)
        method_results[method] = {
            "pair_count": pair_count,
            "counts": counts,
            "required_pair_relative_uplift": _summary(required_pair),
            "required_boundary_uplift": _summary(required_boundary),
            "observed_pair_relative_uplift": _summary(observed_pair),
            "theoretical_pair_capacity": _summary(available_relative),
            "observed_positive_correction": _summary(observed_boundary),
            "pair_records": pair_records,
        }
    return {"cases": len(rows), "pairs": pair_count, "residual_bound": residual_bound,
            "near_tie_rule": f"required pair gap <= {NEAR_TIE_STD_FRACTION:.2f} * per-case base-score std",
            "methods": method_results}


def _audit_evaluation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        base = finite_vector(row, "frozen_reader_topk")
        baseline_selected = stable_selected(row, "frozen_reader_topk")
        threshold = float(np.min(base[baseline_selected]))
        margins = {"fifth_minus_sixth": float(np.sort(base)[-5] - np.sort(base)[-6])}
        method_rows: dict[str, Any] = {}
        for method in LEARNED_METHODS:
            scores = finite_vector(row, method)
            selected = stable_selected(row, method)
            # Selection is a set-valued Top-5 outcome.  Ordering changes among
            # the five selected items (often a stable tie-break) are not a
            # retrieval change and must not inflate the screen-change count.
            changed = set(selected) != set(baseline_selected)
            delta = scores - base
            method_rows[method] = {
                "changed_selection": changed,
                "selected_indices": selected,
                "selected_overlap_with_reader": len(set(selected) & set(baseline_selected)),
                "max_abs_correction": float(np.max(np.abs(delta))),
                "mean_abs_correction": float(np.mean(np.abs(delta))),
                "selected_boundary_margin_after": float(np.min(scores[selected]) - np.max(np.delete(scores, selected))),
                "baseline_fifth_score": threshold,
            }
        records.append({
            "case_number": index,
            "question_sha256": str(row.get("question_sha256", row.get("training_target", {}).get("question_sha256", ""))),
            "baseline_selected_indices": baseline_selected,
            "baseline_boundary_margin": margins,
            "methods": method_rows,
        })
    return {
        "cases": len(rows),
        "methods": {
            method: {
                "selection_changes": int(sum(r["methods"][method]["changed_selection"] for r in records)),
                "selection_unchanged": int(sum(not r["methods"][method]["changed_selection"] for r in records)),
                "max_abs_correction": _summary([r["methods"][method]["max_abs_correction"] for r in records]),
                "mean_abs_correction": _summary([r["methods"][method]["mean_abs_correction"] for r in records]),
            } for method in LEARNED_METHODS
        },
        "case_records": records,
    }


def audit(document: Mapping[str, Any], *, residual_bound: float) -> dict[str, Any]:
    rows = document["cases"]
    training = [row for row in rows if row.get("lane") == "training" and row.get("training_target")]
    evaluation = [row for row in rows if row.get("lane") == "evaluation"]
    if len(training) != 59 or len(evaluation) != 100:
        raise ValueError(f"expected 59 training and 100 evaluation rows, got {len(training)} and {len(evaluation)}")
    return {
        "schema_version": "rag.protected_boundary_feasibility_audit.v1",
        "input_schema_version": document.get("schema_version"),
        "claim_ceiling": "L0_diagnostic",
        "method": "frozen_reader_topk baseline; no model fitting, no label/tuning, no text export",
        "training": _audit_training(training, residual_bound),
        "evaluation": _audit_evaluation(evaluation),
    }


def render_report(result: Mapping[str, Any], input_path: Path) -> str:
    lines = ["# Protected-boundary feasibility audit", "", f"Input: `{input_path.as_posix()}`", "",
             "This report is deterministic and post-hoc. It contains no question, passage, answer, or annotation text.", ""]
    train = result["training"]
    lines += ["## Training protected pairs", "", "| Method | Pairs | Structurally unreachable | Reachable but not learned | Pair rescue | Harmful | Near-tie |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in LEARNED_METHODS:
        counts = train["methods"][method]["counts"]
        lines.append(f"| {method} | {train['methods'][method]['pair_count']} | {counts['structurally_unreachable']} | {counts['reachable_but_not_learned']} | {counts['observed_pair_rescue']} | {counts['harmful']} | {counts['near_tie']} |")
    lines += ["", "`Structurally unreachable` means the required positive-vs-negative score gap exceeds the configured two-sided residual bound for that question. `Reachable but not learned` means it is inside that bound but the final trained scores still do not invert the protected pair. These are feasibility diagnostics, not gold labels.", "", "## Evaluation boundary", ""]
    ev = result["evaluation"]
    for method in LEARNED_METHODS:
        stats = ev["methods"][method]
        lines.append(f"- **{method}**: selection changed in {stats['selection_changes']}/100 cases; unchanged in {stats['selection_unchanged']}/100; max absolute correction median {stats['max_abs_correction']['median']:.6g}.")
    lines += ["", "## Decision use", "", "Use this audit before changing residual bounds, epochs, or supervision. A large structurally-unreachable fraction calls for a representational/score-scale redesign; a large reachable-but-not-learned fraction calls for diagnosing features, objective, or optimization. This artifact does not establish quantum attribution or L1/L2 utility.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--residual-bound", type=float, default=0.25)
    args = parser.parse_args()
    if args.residual_bound <= 0:
        raise SystemExit("--residual-bound must be positive")
    document = load(args.input)
    result = audit(document, residual_bound=args.residual_bound)
    result["input"] = {"path": args.input.as_posix(), "sha256": sha256(args.input)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(result, args.input), encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "training_pairs": result["training"]["pairs"], "evaluation_cases": result["evaluation"]["cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
