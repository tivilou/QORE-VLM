#!/usr/bin/env python3
"""Extract compact, auditable diagnostics from a five-configuration RAG run.

This script reads the large evaluator JSON files locally and writes only scalar
metrics, ranks, derived QUBO statistics, and a passage-free QUBO payload. It
never copies question text, answers, predictions, passage text, or embeddings.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

from gate_manifest import ManifestError, configuration_map, load_manifest


CONFIGS = ("qore_dpr", "qore_as_control", "qore_as_idea6", "topk_as", "mmr_as")
PAIRED = (
    ("qore_dpr", "qore_as_control"),
    ("qore_as_control", "qore_as_idea6"),
    ("qore_as_idea6", "topk_as"),
    ("qore_as_idea6", "mmr_as"),
)
SCALAR_METRICS = (
    "recall", "precision", "redundancy", "diversity", "em", "f1",
    "selection_time_ms", "generation_time_ms",
)
SHARED_CONFIG_KEYS = ("dataset", "split", "corpus_mode", "max_samples", "K", "seed")


class AnalysisError(RuntimeError):
    """Raised when the input run is not safe to analyze as a matched matrix."""


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def number(value: Any, name: str) -> float:
    if not finite(value):
        raise AnalysisError(f"{name} is not a finite number: {value!r}")
    return float(value)


def fmt_list(values: Iterable[Any]) -> str:
    return ";".join(str(int(v)) for v in values)


def mean_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return statistics.fmean(vals) if vals else None


def std_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if finite(v)]
    return statistics.pstdev(vals) if vals else None


def upper_values(matrix: list[list[float]], selected: set[int] | None = None) -> list[float]:
    values: list[float] = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if selected is None or (i in selected and j in selected):
                values.append(float(matrix[i][j]))
    return values


def stats(values: Iterable[float], prefix: str) -> dict[str, float | None]:
    vals = sorted(float(v) for v in values if finite(v))
    if not vals:
        return {f"{prefix}_{suffix}": None for suffix in ("mean", "std", "min", "q10", "q50", "q90", "max")}

    def q(p: float) -> float:
        pos = p * (len(vals) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    return {
        f"{prefix}_mean": statistics.fmean(vals),
        f"{prefix}_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        f"{prefix}_min": vals[0],
        f"{prefix}_q10": q(0.10),
        f"{prefix}_q50": q(0.50),
        f"{prefix}_q90": q(0.90),
        f"{prefix}_max": vals[-1],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(path: Path, display_path: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": display_path,
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def selected_ranks(sample: dict[str, Any]) -> list[int]:
    selected = sample.get("selected_passages") or []
    ranks = [row.get("retrieved_rank") for row in selected if isinstance(row, dict)]
    ranks = [int(v) for v in ranks if isinstance(v, (int, float))]
    if ranks:
        return sorted(set(ranks))
    candidates = sample.get("all_candidates") or []
    return sorted(int(row["retrieved_rank"]) for row in candidates if row.get("selected"))


def gold_ranks(sample: dict[str, Any]) -> list[int]:
    candidates = sample.get("all_candidates") or []
    return sorted(int(row["retrieved_rank"]) for row in candidates if row.get("is_gold"))


def validate_matrix(matrix: Any, size: int, name: str) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != size:
        raise AnalysisError(f"{name} must have {size} rows")
    result: list[list[float]] = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != size:
            raise AnalysisError(f"{name}[{i}] must have {size} columns")
        result.append([number(value, f"{name}[{i}][]") for value in row])
    return result


def qore_features(sample: dict[str, Any], config: dict[str, Any], max_enumerations: int) -> tuple[dict[str, Any], dict[str, Any]]:
    diag = sample.get("qubo")
    if not isinstance(diag, dict):
        raise AnalysisError(f"{sample.get('question_id')}: missing QUBO diagnostics")
    for key in ("a", "b", "w", "x", "pool_ranks"):
        if key not in diag:
            raise AnalysisError(f"{sample.get('question_id')}: QUBO diagnostics missing {key}")

    a = [number(v, "a[]") for v in diag["a"]]
    n = len(a)
    b = validate_matrix(diag["b"], n, "b")
    w = validate_matrix(diag["w"], n, "w")
    x = [int(v) for v in diag["x"]]
    pool_ranks = [int(v) for v in diag["pool_ranks"]]
    if len(x) != n or len(pool_ranks) != n:
        raise AnalysisError(f"{sample.get('question_id')}: QUBO vector/rank lengths do not match a")
    if any(v not in (0, 1) for v in x):
        raise AnalysisError(f"{sample.get('question_id')}: x is not binary")
    if len(set(pool_ranks)) != len(pool_ranks) or any(v < 0 for v in pool_ranks):
        raise AnalysisError(f"{sample.get('question_id')}: invalid pool_ranks")

    k = int(diag.get("K", config.get("K", 0)))
    selected = {i for i, value in enumerate(x) if value}
    if len(selected) != k:
        raise AnalysisError(f"{sample.get('question_id')}: sum(x)={len(selected)} but K={k}")
    expected_selected_ranks = sorted(pool_ranks[i] for i in selected)
    observed_selected_ranks = selected_ranks(sample)
    if observed_selected_ranks and observed_selected_ranks != expected_selected_ranks:
        raise AnalysisError(f"{sample.get('question_id')}: selected ranks disagree with QUBO x/pool_ranks")

    all_candidates = sample.get("all_candidates") or []
    candidate_gold = {int(row["retrieved_rank"]) for row in all_candidates if row.get("is_gold")}
    pool_rank_set = set(pool_ranks)
    selected_rank_set = set(expected_selected_ranks)
    gamma = number(diag.get("gamma_effective", config.get("gamma", 1.0)), "gamma")
    lam = number(diag.get("lam", config.get("lam", 0.0)), "lam")
    b_pool = upper_values(b)
    w_pool = upper_values(w)
    b_selected = upper_values(b, selected)
    w_selected = upper_values(w, selected)
    residual_pool = [gamma * b[i][j] - w[i][j] for i, j in itertools.combinations(range(n), 2)]
    residual_selected = [gamma * b[i][j] - w[i][j] for i, j in itertools.combinations(sorted(selected), 2)]
    quality_sum = sum(a[i] for i in selected)
    pair_sum = sum(w_selected)
    penalty = lam * (len(selected) - k) ** 2
    objective = -quality_sum + pair_sum + penalty
    qubo_energy = objective - lam * (k ** 2)

    best, best_energy, combination_count = None, None, None
    if math.comb(n, k) <= max_enumerations:
        combination_count = math.comb(n, k)
        best_energy = math.inf
        for combo in itertools.combinations(range(n), k):
            value = -sum(a[i] for i in combo) + sum(w[i][j] for i, j in itertools.combinations(combo, 2))
            best_energy = min(best_energy, value)
        best = abs(objective - best_energy) <= 1e-8

    row: dict[str, Any] = {
        "configuration": config["name"],
        "question_id": sample["question_id"],
        "n_candidates": diag.get("n_candidates"),
        "prefiltered": bool(diag.get("prefiltered", False)),
        "pool_size": n,
        "K": k,
        "lam": lam,
        "gamma_effective": gamma,
        "selected_count": len(selected),
        "selected_ranks": fmt_list(expected_selected_ranks),
        "gold_ranks": fmt_list(sorted(candidate_gold)),
        "gold_in_pool": int(bool(candidate_gold & pool_rank_set)),
        "gold_selected": int(bool(candidate_gold & selected_rank_set)),
        "gold_pool_count": len(candidate_gold & pool_rank_set),
        "gold_selected_count": len(candidate_gold & selected_rank_set),
        "quality_sum_selected": quality_sum,
        "quality_mean_selected": mean_or_none(a[i] for i in selected),
        "pair_count_selected": len(b_selected),
        "b_pair_sum_selected": sum(b_selected),
        "w_pair_sum_selected": pair_sum,
        "residual_pair_sum_selected": sum(residual_selected),
        "objective_without_constant": objective,
        "qubo_energy_recomputed": qubo_energy,
        "recorded_energy": diag.get("energy"),
        "recorded_terms_total": (diag.get("terms") or {}).get("total"),
        "exact_optimal_under_w": best,
        "best_objective_under_w": best_energy,
        "objective_regret": None if best_energy is None else objective - best_energy,
        "combination_count": combination_count,
    }
    row.update(stats(a, "a_pool"))
    row.update(stats([a[i] for i in selected], "a_selected"))
    row.update(stats(b_pool, "b_pool"))
    row.update(stats(b_selected, "b_selected"))
    row.update(stats(w_pool, "w_pool"))
    row.update(stats(w_selected, "w_selected"))
    row.update(stats(residual_pool, "residual_pool"))
    row.update(stats(residual_selected, "residual_selected"))
    payload = {
        "configuration": config["name"],
        "question_id": sample["question_id"],
        "K": k,
        "lam": lam,
        "gamma_effective": gamma,
        "n_candidates": diag.get("n_candidates"),
        "prefiltered": bool(diag.get("prefiltered", False)),
        "a": a,
        "b": b,
        "w": w,
        "x": x,
        "pool_ranks": pool_ranks,
        "candidate_flags": [
            {
                "retrieved_rank": int(row["retrieved_rank"]),
                "score": row.get("score"),
                "is_gold": bool(row.get("is_gold", False)),
                "selected": bool(row.get("selected", False)),
            }
            for row in all_candidates
        ],
    }
    return row, payload


def scalar_row(sample: dict[str, Any], config_name: str) -> dict[str, Any]:
    row = {"configuration": config_name, "question_id": sample.get("question_id")}
    for key in SCALAR_METRICS:
        row[key] = sample.get(key)
    row["retrieval_hit"] = sample.get("answer_hit_at_retrieved")
    row["selected_ranks"] = fmt_list(selected_ranks(sample))
    row["gold_ranks"] = fmt_list(gold_ranks(sample))
    trace = (sample.get("qubo") or {}).get("enhancer_trace")
    if isinstance(trace, list):
        compact_trace = []
        for item in trace:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and finite(item.get("elapsed_ms")):
                compact_trace.append(f"{item['name']}:{float(item['elapsed_ms']):.6f}")
        row["plugin_timing_available"] = bool(compact_trace)
        row["plugin_timing_ms"] = ";".join(compact_trace)
    else:
        row["plugin_timing_available"] = False
        row["plugin_timing_ms"] = ""
    return row


def bootstrap_delta(diffs: list[float], reps: int, seed: int) -> tuple[float, float]:
    if not diffs:
        return math.nan, math.nan
    rng = random.Random(seed)
    n = len(diffs)
    samples = []
    for _ in range(reps):
        samples.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    lo = samples[max(0, int(0.025 * (len(samples) - 1)))]
    hi = samples[min(len(samples) - 1, int(0.975 * (len(samples) - 1)))]
    return lo, hi


def paired_effects(
    rows: dict[str, dict[str, dict[str, Any]]],
    pairs: list[tuple[str, str]],
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for pair_index, (left, right) in enumerate(pairs):
        common = sorted(set(rows[left]) & set(rows[right]), key=str)
        for metric in ("recall", "f1", "em", "redundancy", "selection_time_ms"):
            diffs = []
            for qid in common:
                lv, rv = rows[left][qid].get(metric), rows[right][qid].get(metric)
                if finite(lv) and finite(rv):
                    diffs.append(float(rv) - float(lv))
            if not diffs:
                output.append({"left": left, "right": right, "metric": metric, "n": 0})
                continue
            lo, hi = bootstrap_delta(diffs, reps, seed + pair_index * 100 + len(metric))
            output.append({
                "left": left, "right": right, "metric": metric, "n": len(diffs),
                "mean_delta": statistics.fmean(diffs),
                "wins": sum(v > 1e-12 for v in diffs),
                "losses": sum(v < -1e-12 for v in diffs),
                "ties": sum(abs(v) <= 1e-12 for v in diffs),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: dict[str, dict[str, dict[str, Any]]], config_names: list[str]) -> list[dict[str, Any]]:
    output = []
    for config in config_names:
        values = rows[config]
        item: dict[str, Any] = {"configuration": config, "samples": len(values)}
        hits = [v.get("retrieval_hit") for v in values.values() if isinstance(v.get("retrieval_hit"), bool)]
        item["retrieval_hits"] = sum(hits) if hits else None
        for metric in ("recall", "f1", "em", "redundancy", "selection_time_ms", "generation_time_ms"):
            vals = [v.get(metric) for v in values.values() if finite(v.get(metric))]
            item[f"mean_{metric}"] = statistics.fmean(vals) if vals else None
            item[f"std_{metric}"] = statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        output.append(item)
    return output


def validate_provided_aggregates(
    aggregates: list[dict[str, Any]], configs: dict[str, dict[str, Any]], tolerance: float = 1e-9
) -> None:
    for aggregate in aggregates:
        config_name = aggregate["configuration"]
        provided = configs[config_name]["provided_metrics"]
        if provided.get("n_samples") != aggregate["samples"]:
            raise AnalysisError(
                f"{config_name}: provided n_samples={provided.get('n_samples')!r} "
                f"does not match extracted samples={aggregate['samples']}"
            )
        for metric in ("recall", "f1", "em", "redundancy", "selection_time_ms", "generation_time_ms"):
            expected = aggregate.get(f"mean_{metric}")
            observed = provided.get(f"mean_{metric}")
            if expected is None and observed is None:
                continue
            if not finite(expected) or not finite(observed) or abs(float(expected) - float(observed)) > tolerance:
                raise AnalysisError(
                    f"{config_name}: provided mean_{metric}={observed!r} "
                    f"does not match extracted value={expected!r}"
                )


def make_readme(run_dir: Path, output_dir: Path, summary: dict[str, Any]) -> str:
    try:
        publish_path = output_dir.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        publish_path = output_dir.as_posix()
    lines = [
        "# Diagnostics Analysis",
        "",
        f"- Source run: `{run_dir.name}`",
        f"- Configurations: {', '.join(summary['gate']['configurations'])}",
        f"- Samples per configuration: {summary['validation']['samples_per_config']}",
        f"- Bootstrap: {summary['options']['bootstrap_reps']} repetitions, seed {summary['options']['bootstrap_seed']}",
        "",
        "The analyzer ran locally on the original result.json files. This directory contains no question text, answers, predictions, passage text, or embeddings.",
        "",
        "## Files",
        "",
        "- `aggregate.csv`: recomputed scalar metrics.",
        "- `paired_effects.csv`: question-level paired deltas and deterministic bootstrap 95% intervals.",
        "- `per_question.csv`: scalar metrics, retrieval/gold flags, and selected ranks for every configuration.",
        "- `qubo_diagnostics.csv`: QUBO-derived quality, redundancy, interaction, residual-complementarity, objective, and optimality statistics.",
        "- `plugin_timing.csv`: passage-free per-question enhancer timing when emitted by the selector.",
        "- `qubo_payload.jsonl.gz`: passage-free `a/b/w/x/pool_ranks` payload for exact offline re-scoring of the QUBO pool.",
        "- `summary.json`: machine-readable validation, provenance, aggregate, and paired results.",
        "",
        "## Interpretation guard",
        "",
        "The current pilot fixes K=5. It reports the ingredients for a cohesion scale-law analysis, but cannot establish a delta*(K-1) law without a K sweep or held-out evaluation.",
        "The recorded diagnostics energy/terms are retained for comparison; `qubo_diagnostics.csv` also recomputes the actual interaction objective from w, which is the solver matrix for enhancer runs.",
        "",
        "## Publish",
        "",
        f"`git add {publish_path}/ && git commit -m 'results: add compact diagnostics analysis' && git push`",
    ]
    return "\n".join(lines) + "\n"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise AnalysisError(f"Run directory does not exist: {run_dir}")
    output_dir = (args.output_dir or (run_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest = load_manifest(args.gate_config.resolve() if args.gate_config else None)
    except ManifestError as exc:
        raise AnalysisError(str(exc)) from exc
    config_specs = configuration_map(manifest)
    config_names = [spec["name"] for spec in manifest["gate"]["configurations"]]
    pairs = [tuple(pair) for pair in manifest["gate"]["paired_comparisons"]]
    configs: dict[str, dict[str, Any]] = {}
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    qore_rows: list[dict[str, Any]] = []
    plugin_timing_rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    source_info: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None
    reference_qids: list[Any] | None = None

    for config_name in config_names:
        path = run_dir / config_name / "result.json"
        if not path.exists():
            raise AnalysisError(f"Missing result file: {path}")
        source_info.append(source_manifest(path, f"{config_name}/result.json"))
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config = dict(data.get("config") or {})
        config["name"] = config_name
        if reference_config is None:
            reference_config = config
        else:
            shared_keys = tuple(manifest["gate"]["shared_args"])
            for key in shared_keys:
                if key in config and key in reference_config and config.get(key) != reference_config.get(key):
                    raise AnalysisError(f"Config mismatch for {key}: {config_name}={config.get(key)!r}, reference={reference_config.get(key)!r}")
        samples = data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise AnalysisError(f"{config_name}: samples is missing or empty")
        config_rows: dict[str, dict[str, Any]] = {}
        config_qubo = config_specs[config_name]["kind"] == "qore"
        for sample in samples:
            qid = sample.get("question_id")
            if qid in config_rows:
                raise AnalysisError(f"{config_name}: duplicate question_id {qid!r}")
            compact = scalar_row(sample, config_name)
            config_rows[qid] = compact
            trace = (sample.get("qubo") or {}).get("enhancer_trace")
            if isinstance(trace, list):
                for item in trace:
                    if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                        raise AnalysisError(f"{config_name}/{qid}: malformed enhancer trace")
                    if not finite(item.get("elapsed_ms")):
                        raise AnalysisError(f"{config_name}/{qid}: enhancer elapsed_ms is not finite")
                    plugin_timing_rows.append({
                        "configuration": config_name,
                        "question_id": qid,
                        "plugin": item["name"],
                        "mode": item.get("mode"),
                        "elapsed_ms": float(item["elapsed_ms"]),
                        "input_norm": item.get("input_norm"),
                        "output_norm": item.get("output_norm"),
                        "delta_norm": item.get("delta_norm"),
                    })
            if config_qubo:
                qrow, payload = qore_features(sample, config, args.max_enumerations)
                qore_rows.append(qrow)
                payloads.append(payload)
            elif sample.get("qubo") is not None:
                raise AnalysisError(f"{config_name}: unexpected QUBO diagnostics in baseline sample {qid!r}")
        qids = list(config_rows)
        if reference_qids is None:
            reference_qids = qids
        elif set(qids) != set(reference_qids):
            raise AnalysisError(f"{config_name}: question IDs do not match qore_dpr")
        rows[config_name] = config_rows
        configs[config_name] = {"config": config, "provided_metrics": data.get("metrics") or {}, "samples": len(samples)}
        del data, samples

    assert reference_config is not None and reference_qids is not None
    for qid in reference_qids:
        hits = [rows[name][qid].get("retrieval_hit") for name in config_names]
        if not all(isinstance(v, bool) for v in hits):
            raise AnalysisError(f"retrieval hit is missing for question {qid!r}")
        if len(set(hits)) != 1:
            raise AnalysisError(f"retrieval hit mismatch for question {qid!r}")
    if any(len(rows[name]) != len(reference_qids) for name in config_names):
        raise AnalysisError("sample counts are not matched")

    aggregates = aggregate_rows(rows, config_names)
    validate_provided_aggregates(aggregates, configs)
    paired = paired_effects(rows, pairs, args.bootstrap_reps, args.bootstrap_seed)
    plugin_summary: list[dict[str, Any]] = []
    for key in sorted({(row["configuration"], row["plugin"]) for row in plugin_timing_rows}):
        config_name, plugin = key
        values = [row["elapsed_ms"] for row in plugin_timing_rows if (row["configuration"], row["plugin"]) == key]
        plugin_summary.append({
            "configuration": config_name,
            "plugin": plugin,
            "samples": len(values),
            "mean_elapsed_ms": statistics.fmean(values),
            "std_elapsed_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        })
    validation = {
        "samples_per_config": len(reference_qids),
        "question_ids_matched": True,
        "retrieval_hit_matched": True,
        "qore_diagnostics": len(qore_rows),
        "expected_qore_diagnostics": len(reference_qids) * sum(spec["kind"] == "qore" for spec in manifest["gate"]["configurations"]),
        "qore_diagnostics_complete": len(qore_rows) == len(reference_qids) * sum(spec["kind"] == "qore" for spec in manifest["gate"]["configurations"]),
    }
    if not validation["qore_diagnostics_complete"]:
        raise AnalysisError("QORE diagnostics are not complete")

    per_question = [rows[name][qid] for name in config_names for qid in reference_qids]
    summary = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "gate": {
            "name": manifest["gate"]["name"],
            "configurations": config_names,
            "paired_comparisons": [list(pair) for pair in pairs],
            "manifest": manifest.get("source"),
        },
        "sources": source_info,
        "configs": configs,
        "validation": validation,
        "options": {
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
            "max_enumerations": args.max_enumerations,
        },
        "aggregates": aggregates,
        "paired_effects": paired,
        "plugin_timing": plugin_summary,
        "cache_stats": manifest["cache_stats"],
        "reproducibility": {},
        "privacy": {"passage_text": False, "question_text": False, "answers": False, "predictions": False, "embeddings": False},
    }
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid run metadata: {metadata_path}") from exc
        summary["reproducibility"] = {
            "git": metadata.get("git", {}),
            "python": metadata.get("python", {}),
            "gate_config": metadata.get("gate_config", {}),
        }

    write_csv(output_dir / "aggregate.csv", aggregates)
    write_csv(output_dir / "paired_effects.csv", paired)
    write_csv(output_dir / "per_question.csv", per_question)
    write_csv(output_dir / "qubo_diagnostics.csv", qore_rows)
    write_csv(output_dir / "plugin_timing.csv", plugin_timing_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload_path = output_dir / "qubo_payload.jsonl.gz"
    with payload_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0) as gzip_handle:
            for payload in payloads:
                line = json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
                gzip_handle.write(line.encode("utf-8"))
    artifact_bytes = payload_path.stat().st_size
    if artifact_bytes > args.max_artifact_mb * 1024 * 1024:
        raise AnalysisError(f"Compressed QUBO payload is {artifact_bytes} bytes; exceeds --max-artifact-mb={args.max_artifact_mb}")
    summary["artifacts"] = {"qubo_payload_bytes": artifact_bytes}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(make_readme(run_dir, output_dir, summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Five-configuration diagnostics run directory")
    parser.add_argument("--gate-config", type=Path, help="YAML gate manifest; defaults to legacy five configurations")
    parser.add_argument("--output-dir", type=Path, help="Output directory; default RUN_DIR/analysis")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--max-enumerations", type=int, default=100000)
    parser.add_argument("--max-artifact-mb", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.bootstrap_reps < 100 or args.max_enumerations < 1 or args.max_artifact_mb < 1:
            raise AnalysisError("bootstrap/artifact limits are invalid")
        summary = analyze(args)
        print(f"Wrote diagnostics analysis to {Path(args.output_dir or args.run_dir / 'analysis').resolve()}")
        print(f"Validated {summary['validation']['samples_per_config']} matched questions and {summary['validation']['qore_diagnostics']} QORE diagnostics")
        return 0
    except (AnalysisError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
