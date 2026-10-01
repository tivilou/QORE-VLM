"""Reader-span/relevance feature fusion for a fixed Top-50 -> Top-5 screen.

The quantum arm is a small, fixed four-qubit feature map. Silver/gold labels,
generator outputs, and evaluator results are deliberately absent from this
module's inputs. Its exact statevector result has an independent classical
Born-probability implementation for parity checks.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math
import re

import numpy as np


FEATURE_NAMES = (
    "relevance",
    "best_span",
    "span_margin",
    "boundary_concentration",
)
N_QUBITS = 4
CNOT_RING = ((0, 1), (1, 2), (2, 3), (3, 0))
TOP5 = 5
REDUNDANCY_PENALTY = 0.12
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class SpanFusionError(ValueError):
    """Raised when an input violates the fixed-slice selector contract."""


def _finite(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise SpanFusionError("non-finite " + name)
    return array


def build_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Create four question-normalized online features from Reader outputs."""

    if len(rows) != 50:
        raise SpanFusionError("expected exactly 50 feature rows")
    raw_columns = []
    for key in ("relevance_logit", "span_logit", "span_margin"):
        raw_columns.append(_finite([row[key] for row in rows], key))
    start_entropy = _finite([row["start_entropy"] for row in rows], "start_entropy")
    end_entropy = _finite([row["end_entropy"] for row in rows], "end_entropy")
    raw_columns.append(-0.5 * (start_entropy + end_entropy))

    features = np.zeros((50, N_QUBITS), dtype=np.float64)
    for column_index, values in enumerate(raw_columns):
        spread = float(values.std())
        if spread > 1e-12:
            standardized = (values - float(values.mean())) / spread
            features[:, column_index] = np.clip(standardized / 3.0, -1.0, 1.0)
    return features


def classical_linear_scores(features: np.ndarray) -> np.ndarray:
    """Fixed 50:50 relevance/evidence fusion; weights are not fitted."""

    matrix = _features(features)
    evidence = matrix[:, 1:].mean(axis=1)
    scores = 0.5 * matrix[:, 0] + 0.5 * evidence
    return _finite(scores, "classical fusion scores")


def _features(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (50, N_QUBITS) or not np.all(np.isfinite(matrix)):
        raise SpanFusionError("features must be a finite 50x4 matrix")
    if np.any(matrix < -1.0000001) or np.any(matrix > 1.0000001):
        raise SpanFusionError("encoded features must be in [-1, 1]")
    return np.clip(matrix, -1.0, 1.0)


def _apply_ry(state: np.ndarray, theta: float, wire: int) -> np.ndarray:
    result = state.copy()
    bit = 1 << wire
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    for lower in range(state.size):
        if lower & bit:
            continue
        upper = lower | bit
        alpha = state[lower]
        beta = state[upper]
        result[lower] = cosine * alpha - sine * beta
        result[upper] = sine * alpha + cosine * beta
    return result


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    result = state.copy()
    control_bit = 1 << control
    target_bit = 1 << target
    for index in range(state.size):
        if index & control_bit and not index & target_bit:
            paired = index | target_bit
            result[index] = state[paired]
            result[paired] = state[index]
    return result


def _z_expectation(probabilities: np.ndarray, output_mask: int) -> float:
    total = 0.0
    for basis_state, probability in enumerate(probabilities):
        parity = bin(basis_state & output_mask).count("1") & 1
        total += float(probability) * (-1.0 if parity else 1.0)
    return total


def quantum_statevector_scores(features: np.ndarray) -> np.ndarray:
    """Evaluate RY feature encoding + one directed CNOT-ring block exactly."""

    matrix = _features(features)
    scores = np.empty(50, dtype=np.float64)
    for row_index, row in enumerate(matrix):
        state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
        state[0] = 1.0
        for wire, value in enumerate(row):
            state = _apply_ry(state, math.acos(float(value)), wire)
        for control, target in CNOT_RING:
            state = _apply_cnot(state, control, target)
        probabilities = np.abs(state) ** 2
        local = [_z_expectation(probabilities, 1 << wire) for wire in range(N_QUBITS)]
        pair = [
            _z_expectation(probabilities, (1 << left) | (1 << right))
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 0))
        ]
        scores[row_index] = 0.5 * float(np.mean(local)) + 0.5 * float(np.mean(pair))
    return _finite(scores, "quantum statevector scores")


def classical_born_scores(features: np.ndarray) -> np.ndarray:
    """Evaluate the circuit's Born expectations as an exact parity polynomial."""

    matrix = _features(features)

    def transform_bits(value: int) -> int:
        for control, target in CNOT_RING:
            if value & (1 << control):
                value ^= 1 << target
        return value

    columns = [transform_bits(1 << input_wire) for input_wire in range(N_QUBITS)]

    def pullback(output_mask: int) -> int:
        input_mask = 0
        for input_wire, output_column in enumerate(columns):
            if bin(output_column & output_mask).count("1") & 1:
                input_mask |= 1 << input_wire
        return input_mask

    local_masks = [pullback(1 << wire) for wire in range(N_QUBITS)]
    pair_masks = [
        pullback((1 << left) | (1 << right))
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 0))
    ]

    def expectation(row: np.ndarray, mask: int) -> float:
        value = 1.0
        for wire in range(N_QUBITS):
            if mask & (1 << wire):
                value *= float(row[wire])
        return value

    scores = np.empty(50, dtype=np.float64)
    for row_index, row in enumerate(matrix):
        local = [expectation(row, mask) for mask in local_masks]
        pair = [expectation(row, mask) for mask in pair_masks]
        scores[row_index] = 0.5 * float(np.mean(local)) + 0.5 * float(np.mean(pair))
    return _finite(scores, "classical Born-control scores")


def ranked_indices(scores: Sequence[float], rows: Sequence[Mapping[str, Any]], k: int = TOP5) -> list[int]:
    values = _finite(scores, "ranking scores")
    if len(rows) != 50 or values.size != 50 or k < 1 or k > 50:
        raise SpanFusionError("ranker requires 50 rows and 1 <= k <= 50")
    return sorted(
        range(50),
        key=lambda index: (
            -float(values[index]),
            int(rows[index]["retrieved_rank"]),
            str(rows[index]["candidate_id"]),
        ),
    )[:k]


def _token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _similarity_matrix(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    token_sets = [_token_set(str(candidate.get("text", ""))) for candidate in candidates]
    matrix = np.zeros((50, 50), dtype=np.float64)
    for left in range(50):
        for right in range(left + 1, 50):
            union = token_sets[left] | token_sets[right]
            similarity = len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
            matrix[left, right] = similarity
            matrix[right, left] = similarity
    return matrix


def redundancy_aware_indices(
    scores: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
    penalty: float = REDUNDANCY_PENALTY,
    k: int = TOP5,
) -> list[int]:
    """Greedily penalize overlap with already selected passages."""

    values = _finite(scores, "redundancy ranking scores")
    if len(candidates) != 50 or values.size != 50 or k < 1 or k > 50:
        raise SpanFusionError("diversity selector requires 50 candidates and 1 <= k <= 50")
    if not math.isfinite(penalty) or penalty < 0.0 or penalty > 1.0:
        raise SpanFusionError("redundancy penalty must be in [0, 1]")
    low = float(values.min())
    high = float(values.max())
    normalized = np.zeros(50, dtype=np.float64) if high - low <= 1e-12 else (values - low) / (high - low)
    similarity = _similarity_matrix(candidates)
    selected: list[int] = []
    remaining = set(range(50))
    for _ in range(k):
        def objective(index: int) -> tuple[float, int, str]:
            overlap = max((float(similarity[index, chosen]) for chosen in selected), default=0.0)
            adjusted = float(normalized[index]) - penalty * overlap
            return (
                -adjusted,
                int(candidates[index]["retrieved_rank"]),
                str(candidates[index]["candidate_id"]),
            )

        chosen = min(remaining, key=objective)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected
