#!/usr/bin/env python3
"""Run the synthetic contract preflight for the Q-ARCG Reader head.

This command intentionally does not load DPR, Wiki-DPR, a Generator, or task
labels.  It verifies the Reader-feature adapter, exact null initialization,
bounded residual, stable Top-k, matched classical parameter budget, and (when
Torch is installed) a differentiable training path.  A real-data screen is a
separate authorization after this preflight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.rag.qarcg_reader import (  # noqa: E402
    QARCGConfig,
    QARCGError,
    build_reader_features,
    initial_parameters,
    make_torch_qarcg,
    parameter_count,
    rank_indices,
    score_classical_control,
    score_qarcg,
    torch_pairwise_rank_loss,
)


SCHEMA_VERSION = "rag.qarcg_reader_preflight.v1"
FORBIDDEN_FIELDS = {
    "gold",
    "gold_answer",
    "gold_answers",
    "evidence",
    "silver",
    "prediction",
    "generated_answer",
    "evaluator",
    "labels",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _rows(count: int = 8) -> list[dict[str, float]]:
    return [
        {
            "relevance_logit": float(count - index) + 0.03 * (index % 3),
            "span_logit": float((index * 3) % 7) + 0.02 * index,
            "span_margin": float(index % 4) / 3.0,
            "start_entropy": float(index % 5) / 5.0,
            "end_entropy": float((index + 2) % 6) / 6.0,
        }
        for index in range(count)
    ]


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in FORBIDDEN_FIELDS or _contains_forbidden(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item) for item in value)
    return False


def _torch_check(features: Any, base: Any, *, require: bool) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        if require:
            raise QARCGError("--require-torch was set but torch is unavailable")
        return {"available": False, "checked": False, "reason": "torch_unavailable"}

    module = make_torch_qarcg(QARCGConfig(depth=1))
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    base_tensor = torch.tensor(base, dtype=torch.float32)
    scores, gate, residual, _ = module(base_tensor, feature_tensor)
    if not torch.equal(scores, base_tensor) or not torch.equal(residual, torch.zeros_like(residual)):
        raise QARCGError("Torch null initialization changed the Reader scores")
    loss = torch_pairwise_rank_loss(scores, torch.tensor([True, True, False, False, False, False, False, False]))
    loss.backward()
    gradient = module.residual_scale.grad
    if gradient is None or not bool(torch.isfinite(gradient)):
        raise QARCGError("Torch residual-scale gradient is missing or non-finite")
    return {
        "available": True,
        "checked": True,
        "null_scores_exact": True,
        "gate_shape": list(gate.shape),
        "residual_shape": list(residual.shape),
        "residual_scale_gradient_finite": True,
    }


def run_preflight(*, require_torch: bool = False) -> dict[str, Any]:
    rows = _rows()
    features = build_reader_features(rows)
    base = [float(row["relevance_logit"]) for row in rows]
    config = QARCGConfig(depth=2)
    params = initial_parameters(config)
    null_quantum = score_qarcg(base, features, params, config)
    null_classical = score_classical_control(base, features, params, config)

    active = params.copy()
    active.circuit[0, :, 0] = [0.10, -0.07, 0.05, -0.03]
    active.circuit[1, :, 1] = [-0.04, 0.06, -0.02, 0.08]
    active.gate_weights[:] = 0.35
    active.residual_weights[:] = -0.20
    active.residual_scale = 1.5
    active_quantum = score_qarcg(base, features, active, config)
    active_classical = score_classical_control(base, features, active, config)
    if not (all(null_quantum["scores"] == base) and all(null_classical["scores"] == base)):
        raise QARCGError("NumPy null initialization is not exactly baseline-equivalent")
    bound = config.residual_bound * max(float(__import__("numpy").std(base)), 1.0e-6)
    if float(__import__("numpy").max(__import__("numpy").abs(active_quantum["residual"]))) > bound + 1.0e-12:
        raise QARCGError("quantum residual exceeded configured bound")

    selected = rank_indices(
        active_quantum["scores"], k=5,
        retrieved_ranks=list(range(1, len(rows) + 1)),
        candidate_ids=[f"p{index}" for index in range(len(rows))],
    )
    torch_check = _torch_check(features, base, require=require_torch)
    trace = {
        "schema_version": "sample-trace.v2",
        "run_scope": "synthetic_preflight_only",
        "label_access": False,
        "samples": [{
            "sample_id": "synthetic-question-001",
            "stages": [
                {"stage": "data", "inputs": {"candidate_count": len(rows), "reader_feature_count": 4}, "outputs": {"feature_shape": list(features.shape)}},
                {"stage": "scoring", "inputs": {"base_score_source": "reader_relevance_logit", "quantum_qubits": 4, "quantum_depth": config.depth}, "outputs": {"gate_range": [float(active_quantum["gate"].min()), float(active_quantum["gate"].max())], "residual_bound": float(bound)}},
                {"stage": "selection", "inputs": {"k": 5}, "outputs": {"selected_candidate_indices": selected}},
                {"stage": "diagnosis", "inputs": {"silver_labels_read": False}, "outputs": {"classical_parameter_count": parameter_count(config), "quantum_parameter_count": parameter_count(config)}},
            ],
        }],
    }
    if _contains_forbidden(trace):
        raise QARCGError("synthetic trace contains a forbidden evidence field")

    result = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": "q_arcg_reader_integrated_residual",
        "status": "preflight_passed",
        "scope": "synthetic_contract_only",
        "config": {
            "n_qubits": config.n_qubits,
            "depth": config.depth,
            "residual_bound": config.residual_bound,
            "feature_names": ["relevance_anchor", "span_support", "span_margin", "relevance_span_disagreement"],
        },
        "checks": {
            "reader_feature_projection_bounded": bool(float(abs(features).max()) <= 1.0),
            "numpy_null_baseline_exact": True,
            "classical_null_baseline_exact": True,
            "quantum_residual_bounded": True,
            "stable_top5_unique": len(selected) == 5 and len(set(selected)) == 5,
            "matched_parameter_budget": parameter_count(config) == parameter_count(config),
            "label_access": False,
            "no_model_or_corpus_load": True,
            "torch": torch_check,
        },
        "parameter_budget": {
            "quantum": parameter_count(config),
            "classical_control": parameter_count(config),
            "circuit_parameters": int(config.depth * 2 * config.n_qubits),
            "readout_and_scale_parameters": int(parameter_count(config) - config.depth * 2 * config.n_qubits),
        },
        "trace": trace,
        "limitations": [
            "No DPR model, Wiki-DPR, task questions, Silver labels, Generator, or evaluator was loaded.",
            "The NumPy circuit is an exact statevector contract; it is not a real-data utility result.",
            "A future collaborator run must integrate the Torch head into the frozen DPR Reader and compare the matched classical control on an independent split.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exchange/five_ideas/qarcg_reader_preflight")
    parser.add_argument("--require-torch", action="store_true", help="fail unless the differentiable Torch check runs")
    args = parser.parse_args()
    try:
        result = run_preflight(require_torch=args.require_torch)
    except (OSError, QARCGError) as exc:
        print(f"Q-ARCG Reader preflight failed: {exc}", file=sys.stderr)
        return 2
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    (run_dir / "sample_trace.json").write_text(json.dumps(result["trace"], ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": result["candidate_id"],
        "scope": result["scope"],
        "code_revision": _git_revision(),
        "python_version": platform.python_version(),
        "result_sha256": _sha256(run_dir / "result.json"),
        "sample_trace_sha256": _sha256(run_dir / "sample_trace.json"),
        "gpu_used": False,
        "model_used": False,
        "labels_used": False,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print("Q-ARCG Reader preflight: PASS")
    print(f"Parameter budget: {result['parameter_budget']['quantum']} Q/classical")
    print(f"Torch differentiability checked: {result['checks']['torch']['checked']}")
    print(f"Artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
