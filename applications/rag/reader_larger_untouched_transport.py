"""Label-blind transport and replay for the frozen 256-question Reader study.

The module accepts only the immutable cohort, Reader representations and the
already-fitted sequential heads.  Answers, support labels, Silver judgments
and generator outputs are deliberately outside every public interface.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head
from applications.rag.reader_larger_untouched import validate_cohort
from applications.rag.reader_larger_untouched_manifest import (
    file_sha256,
    validate_scoring_manifest,
    verify_cohort,
    verify_reader_snapshot,
    verify_sequential_checkpoint,
)
from applications.rag.reader_unseen_transport import stable_top5, transport_metrics


ARTIFACT_SCHEMA = "rag.reader_larger_untouched_transport.v1"
TRACE_SCHEMA = "sample-trace.v2"
FEATURE_KEYS = ("identity_scores", "cls", "pooled")
SCORE_KEYS = ("identity_scores", "quantum_scores", "classical_scores")
VALUE_KEYS = ("identity_scores", "cls", "pooled", "quantum_scores", "classical_scores")


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def array_spec(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "sha256": array_sha256(array),
    }


def candidate_text(candidate: Mapping[str, Any]) -> str:
    body = candidate.get("text")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("candidate text must be non-empty")
    title = str(candidate.get("title") or "").strip()
    return f"{title}. {body.strip()}" if title else body.strip()


def case_record(case: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    question = case.get("question")
    candidates = case.get("candidates")
    if not isinstance(question, str) or not isinstance(candidates, list) or len(candidates) != 50:
        raise ValueError("invalid frozen case")
    online_texts = [candidate_text(candidate) for candidate in candidates]
    input_payload = question + "\n" + "\n".join(online_texts)
    return {
        "prefix": prefix,
        "question_sha256": case["question_sha256"],
        "candidate_id_sha256": [candidate["candidate_id_sha256"] for candidate in candidates],
        "candidate_retrieved_ranks": [int(candidate["retrieved_rank"]) for candidate in candidates],
        "input_sha256": hashlib.sha256(input_payload.encode("utf-8")).hexdigest(),
    }


def verify_manifest_sources(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    root = Path(root).resolve()
    expected = manifest.get("provenance", {}).get("method_source_hashes")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("manifest has no frozen method sources")
    actual: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        path = (root / relative).resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"invalid frozen method source: {relative}")
        observed = file_sha256(path)
        if observed != digest:
            raise ValueError(f"frozen method source drift: {relative}")
        actual[str(relative)] = observed
    return actual


def verify_bound_inputs(
    *,
    root: Path,
    manifest_path: Path,
    cohort_dir: Path,
    reader_snapshot: Path,
    sequential_values: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay all immutable bindings before loading the Reader or scoring."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_check = validate_scoring_manifest(manifest)
    if file_sha256(Path(manifest_path)) != "c46eab8726c5003dc56d7f8b3494888cf222ed63be4e3202e4e49e6d3df8267b":
        raise ValueError("immutable scoring-manifest file drift")
    cohort_check = verify_cohort(Path(cohort_dir), manifest)
    reader_check = verify_reader_snapshot(Path(reader_snapshot), manifest)
    checkpoint_check = verify_sequential_checkpoint(Path(sequential_values), manifest)
    source_check = verify_manifest_sources(Path(root), manifest)
    cohort = json.loads((Path(cohort_dir) / "cohort.json").read_text(encoding="utf-8"))
    validate_cohort(cohort, expected_cases=256)
    return manifest, cohort, {
        "manifest": manifest_check,
        "cohort": cohort_check,
        "reader": reader_check,
        "checkpoint": checkpoint_check,
        "method_source_hashes": source_check,
    }


def restore_rotation_head(
    values: np.lib.npyio.NpzFile,
    family: str,
    manifest: Mapping[str, Any],
    expected_weight: np.ndarray,
    expected_bias: np.ndarray,
    *,
    device: Any,
) -> Any:
    import torch

    if family not in ("quantum", "classical"):
        raise ValueError("unknown head family")
    weight = values[f"{family}__head__weight"].copy()
    bias = values[f"{family}__head__bias"].copy()
    np.testing.assert_array_equal(weight.reshape(-1), np.asarray(expected_weight).reshape(-1))
    np.testing.assert_array_equal(bias.reshape(()), np.asarray(expected_bias).reshape(()))
    config = RotationConfig(
        layers=int(manifest["head"]["layers"]),
        conditioning=int(manifest["head"]["conditioning"]),
        seed=int(manifest["head"]["seed"]),
    )
    head = make_rotation_head(torch.from_numpy(weight), torch.from_numpy(bias), family=family, config=config)
    state = {
        name: torch.from_numpy(values[f"{family}__head__{name}"].copy())
        for name in head.state_dict()
    }
    head.load_state_dict(state, strict=True)
    with torch.no_grad():
        final_theta = torch.from_numpy(values[f"{family}__final_theta"].copy())
        head.theta.copy_(final_theta.reshape_as(head.theta).to(head.theta))
    return head.to(device).eval()


def score_heads(
    cls: np.ndarray,
    pooled: np.ndarray,
    quantum: Any,
    classical: Any,
    *,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    cls_value = np.asarray(cls, dtype=np.float32)
    pooled_value = np.asarray(pooled, dtype=np.float32)
    if cls_value.ndim != 3 or pooled_value.shape != (*cls_value.shape[:2], 5 * cls_value.shape[2]):
        raise ValueError("Reader feature shape mismatch")
    q_rows: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    with torch.no_grad():
        for row in range(cls_value.shape[0]):
            cls_tensor = torch.from_numpy(cls_value[row]).to(device)
            pooled_tensor = torch.from_numpy(pooled_value[row]).to(device)
            q_rows.append(quantum(cls_tensor, pooled_tensor)["scores"].detach().cpu().numpy().astype(np.float32))
            c_rows.append(classical(cls_tensor, pooled_tensor)["scores"].detach().cpu().numpy().astype(np.float32))
    return np.stack(q_rows), np.stack(c_rows)


def larger_transport_gate(metrics: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    gate = manifest["transport_gate"]
    changed_min = int(gate["min_quantum_changed_cases_for_review"])
    different_min = int(gate["min_quantum_classical_different_cases_for_review"])
    changed = int(metrics["candidate_changed_questions"])
    different = int(metrics["candidate_control_different_questions"])
    passed = changed >= changed_min and different >= different_min
    return {
        "quantum_transport": "pass" if changed >= changed_min else "fail",
        "quantum_classical_separation": "pass" if different >= different_min else "fail",
        "advance_to_blind_review": bool(passed),
        "observed": {
            "quantum_changed_cases": changed,
            "quantum_classical_different_cases": different,
        },
        "required": {
            "quantum_changed_cases": changed_min,
            "quantum_classical_different_cases": different_min,
        },
        "reason": "both predeclared label-blind transport conditions met" if passed
        else "label-blind transport is too small or insufficiently distinct from the matched classical control",
    }


def build_trace(
    *,
    run_id: str,
    records: Sequence[Mapping[str, Any]],
    values: Mapping[str, np.ndarray],
    values_sha256: str,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    synthetic_only: bool = False,
) -> dict[str, Any]:
    if set(values) != set(VALUE_KEYS):
        raise ValueError("numeric value allowlist drift")
    questions = len(records)
    if any(np.asarray(values[key]).shape[:2] != (questions, 50) for key in VALUE_KEYS):
        raise ValueError("complete Top-50 coordinate coverage required")
    return {
        "schema_version": TRACE_SCHEMA,
        "artifact_type": "reader_larger_untouched_transport",
        "artifact_schema": ARTIFACT_SCHEMA,
        "run_id": run_id,
        "synthetic_only": bool(synthetic_only),
        "population": {"unseen_transport": questions, "training": 0, "evaluation": 0},
        "stages": ["data", "preprocess", "retrieval", "scoring", "selection", "diagnosis"],
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "samples": [
            {
                **dict(record),
                "identity_top5": stable_top5(values["identity_scores"][index]),
                "quantum_top5": stable_top5(values["quantum_scores"][index]),
                "classical_top5": stable_top5(values["classical_scores"][index]),
            }
            for index, record in enumerate(records)
        ],
        "values": {key: array_spec(np.asarray(values[key])) for key in VALUE_KEYS},
        "values_sha256": values_sha256,
        "label_boundary": dict(manifest["label_boundary"]),
        "transport_metrics": dict(metrics),
        "gate": dict(gate),
        "provenance": dict(provenance),
        "claim_ceiling": "fixture_only" if synthetic_only else "L0_label_blind_transport",
    }


def validate_trace(
    trace: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    expected_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if trace.get("schema_version") != TRACE_SCHEMA or trace.get("artifact_schema") != ARTIFACT_SCHEMA:
        raise ValueError("unexpected transport trace schema")
    if trace.get("manifest_identity_sha256") != manifest.get("manifest_identity_sha256"):
        raise ValueError("trace/manifest identity mismatch")
    if any(trace.get("label_boundary", {}).values()):
        raise ValueError("trace crossed the label boundary")
    if set(values) != set(VALUE_KEYS) or set(trace.get("values", {})) != set(VALUE_KEYS):
        raise ValueError("trace numeric allowlist mismatch")
    questions = int(trace["population"]["unseen_transport"])
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) != questions:
        raise ValueError("trace sample coverage mismatch")
    for key in VALUE_KEYS:
        value = np.asarray(values[key])
        if value.shape[:2] != (questions, 50) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"invalid trace numeric shape/dtype: {key}")
        if not bool(np.isfinite(value).all()) or trace["values"][key] != array_spec(value):
            raise ValueError(f"trace numeric coordinate drift: {key}")
    if values["cls"].shape[2] != 768 or values["pooled"].shape[2] != 3840:
        raise ValueError("frozen Reader hidden-size drift")
    for index, sample in enumerate(samples):
        expected_top5 = {
            "identity_top5": stable_top5(values["identity_scores"][index]),
            "quantum_top5": stable_top5(values["quantum_scores"][index]),
            "classical_top5": stable_top5(values["classical_scores"][index]),
        }
        if any(sample.get(name) != selected for name, selected in expected_top5.items()):
            raise ValueError(f"Top-5 replay drift at sample {index}")
        if expected_records is not None:
            for name in ("question_sha256", "candidate_id_sha256", "candidate_retrieved_ranks", "input_sha256"):
                if sample.get(name) != expected_records[index].get(name):
                    raise ValueError(f"sample identity drift at {index}: {name}")
    computed = transport_metrics(values["identity_scores"], values["quantum_scores"], values["classical_scores"])
    gate = larger_transport_gate(computed, manifest)
    if computed != trace.get("transport_metrics") or gate != trace.get("gate"):
        raise ValueError("transport metric/gate replay drift")
    return {
        "status": "passed",
        "cases_replayed": questions,
        "score_vectors_replayed": questions * 3,
        "complete_numeric_vectors": questions * len(VALUE_KEYS),
        "label_boundary_verified": True,
        "transport_metrics": computed,
        "gate": gate,
    }


def replay_head_vectors(
    *,
    values: Mapping[str, np.ndarray],
    sequential_values: Path,
    manifest: Mapping[str, Any],
    device: Any,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> dict[str, Any]:
    import torch

    cls = np.asarray(values["cls"], dtype=np.float32)
    pooled = np.asarray(values["pooled"], dtype=np.float32)
    identity = np.asarray(values["identity_scores"], dtype=np.float32)
    with np.load(sequential_values, allow_pickle=False) as checkpoint:
        weight = checkpoint["quantum__head__weight"].copy()
        bias = checkpoint["quantum__head__bias"].copy()
        quantum = restore_rotation_head(checkpoint, "quantum", manifest, weight, bias, device=device)
        classical = restore_rotation_head(checkpoint, "classical", manifest, weight, bias, device=device)
    linear_rows: list[np.ndarray] = []
    with torch.no_grad():
        weight_tensor = torch.from_numpy(weight).reshape(1, -1).to(device)
        bias_tensor = torch.from_numpy(bias).reshape(1).to(device)
        for row in cls:
            linear_rows.append(
                torch.nn.functional.linear(torch.from_numpy(row).to(device), weight_tensor, bias_tensor)
                .squeeze(-1).cpu().numpy()
            )
    linear = np.stack(linear_rows).astype(np.float32)
    if not np.allclose(linear, identity, rtol=rtol, atol=atol):
        raise ValueError("identity Reader classifier replay failed")
    quantum_scores, classical_scores = score_heads(cls, pooled, quantum, classical, device=device)
    comparisons = {}
    for family, observed, expected in (
        ("quantum", quantum_scores, values["quantum_scores"]),
        ("classical", classical_scores, values["classical_scores"]),
    ):
        expected_array = np.asarray(expected, dtype=np.float32)
        if not np.allclose(observed, expected_array, rtol=rtol, atol=atol):
            raise ValueError(f"{family} score-vector replay failed")
        if [stable_top5(row) for row in observed] != [stable_top5(row) for row in expected_array]:
            raise ValueError(f"{family} Top-5 replay failed")
        comparisons[family] = {
            "vectors": int(len(observed)),
            "max_abs_difference": float(np.max(np.abs(observed - expected_array))),
            "top5_exact": True,
        }
    return {
        "identity": {
            "vectors": int(len(identity)),
            "max_abs_difference": float(np.max(np.abs(linear - identity))),
            "top5_exact": True,
        },
        **comparisons,
    }


__all__ = [
    "ARTIFACT_SCHEMA",
    "VALUE_KEYS",
    "array_spec",
    "build_trace",
    "candidate_text",
    "case_record",
    "larger_transport_gate",
    "replay_head_vectors",
    "restore_rotation_head",
    "score_heads",
    "validate_trace",
    "verify_bound_inputs",
]
