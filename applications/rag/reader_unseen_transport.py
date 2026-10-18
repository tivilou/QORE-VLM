"""Label-blind transport utilities for the frozen unseen-question gate.

This module deliberately operates on numeric Reader features and immutable
question/candidate digests only.  It must not accept answers, support labels,
or generator outputs as inputs.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Mapping

import numpy as np


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_top5(scores: Iterable[float]) -> list[int]:
    values = [float(x) for x in scores]
    if len(values) != 50 or not all(math.isfinite(x) for x in values):
        raise ValueError("expected 50 finite scores")
    return sorted(range(50), key=lambda i: (-values[i], i))[:5]


def top5_sets(scores: np.ndarray) -> list[list[int]]:
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] != 50:
        raise ValueError("expected N x 50 score matrix")
    return [stable_top5(row) for row in values]


def symmetric_difference_size(left: Iterable[int], right: Iterable[int]) -> int:
    return len(set(left).symmetric_difference(right))


def transport_metrics(identity: np.ndarray, candidate: np.ndarray, control: np.ndarray) -> dict:
    """Return pre-label transport and matched-control diagnostics only."""
    identity_sets = top5_sets(identity)
    candidate_sets = top5_sets(candidate)
    control_sets = top5_sets(control)
    changed = [symmetric_difference_size(a, b) for a, b in zip(identity_sets, candidate_sets)]
    control_changed = [symmetric_difference_size(a, b) for a, b in zip(identity_sets, control_sets)]
    arm_difference = [symmetric_difference_size(a, b) for a, b in zip(candidate_sets, control_sets)]
    overlap = [len(set(a).intersection(b)) for a, b in zip(identity_sets, candidate_sets)]
    return {
        "questions": int(identity.shape[0]),
        "candidate_changed_questions": int(sum(x > 0 for x in changed)),
        "candidate_symmetric_difference_total": int(sum(changed)),
        "candidate_mean_top5_overlap_with_identity": float(np.mean(overlap)),
        "control_changed_questions": int(sum(x > 0 for x in control_changed)),
        "control_symmetric_difference_total": int(sum(control_changed)),
        "candidate_control_different_questions": int(sum(x > 0 for x in arm_difference)),
        "candidate_control_symmetric_difference_total": int(sum(arm_difference)),
        "candidate_changed_distribution": {str(k): int(changed.count(k)) for k in sorted(set(changed))},
        "control_changed_distribution": {str(k): int(control_changed.count(k)) for k in sorted(set(control_changed))},
        "candidate_control_difference_distribution": {str(k): int(arm_difference.count(k)) for k in sorted(set(arm_difference))},
    }


def evaluate_transport_gate(metrics: Mapping[str, int], config: Mapping[str, object]) -> dict:
    """Evaluate only the prospective label-blind transport gate."""
    gate = config["transport_gate"]
    changed_ok = metrics["candidate_changed_questions"] >= int(gate["min_candidate_changed_questions"])
    separation_ok = metrics["candidate_control_different_questions"] >= int(gate["min_candidate_control_different_questions"])
    return {
        "candidate_transport": "pass" if changed_ok else "fail",
        "quantum_classical_separation": "pass" if separation_ok else "fail",
        "advance_to_blind_review": bool(changed_ok and separation_ok),
        "reason": "both predeclared transport conditions met" if changed_ok and separation_ok
        else "label-blind transport is too small or indistinguishable from matched classical control",
    }
