"""Quantum Applicability-gated Residual Calibration for a DPR Reader.

Q-ARCG is a Reader-integrated *head contract*, not a post-hoc 50:50 span
fusion.  The frozen Reader relevance score remains the anchor.  A shallow
parameterized circuit consumes four Reader-side features and produces an
applicability gate plus a bounded residual::

    score = reader_relevance + gate * residual

The residual scale is initialized to zero, so the disabled intervention is
exactly the original Reader ordering.  The module is deliberately independent
of transformers and QORE: a collaborator can connect the tensor adapter to a
real DPRReader training step, while this file remains usable for deterministic
contract tests.  Silver labels, generated answers, and evaluator outputs are
not accepted by any online scoring function.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = (
    "relevance_anchor",
    "span_support",
    "span_margin",
    "relevance_span_disagreement",
)
N_QUBITS = 4
RING = ((0, 1), (1, 2), (2, 3), (3, 0))
OBSERVABLE_COUNT = N_QUBITS + len(RING)
DEFAULT_DEPTH = 2
DEFAULT_RESIDUAL_BOUND = 0.25


class QARCGError(ValueError):
    """Raised when the Reader-head contract is violated."""


def _finite_array(value: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise QARCGError(f"{name} must have ndim={ndim}")
    if not np.all(np.isfinite(array)):
        raise QARCGError(f"{name} contains non-finite values")
    return array


def _standardize(values: np.ndarray) -> np.ndarray:
    spread = float(values.std())
    if spread <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(values.mean())) / spread


def build_reader_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Project compact DPR Reader outputs into four bounded online features.

    ``rows`` contains only values emitted by the Reader.  The projection is
    question-local: the caller must pass the 50 candidates for one question,
    or another fixed candidate cohort.  No evidence, answer, or label fields
    are read.
    """

    if not rows:
        raise QARCGError("at least one Reader row is required")
    relevance = _finite_array([row["relevance_logit"] for row in rows], "relevance_logit", ndim=1)
    span = _finite_array([row["span_logit"] for row in rows], "span_logit", ndim=1)
    margin = _finite_array([row["span_margin"] for row in rows], "span_margin", ndim=1)
    start_entropy = _finite_array([row["start_entropy"] for row in rows], "start_entropy", ndim=1)
    end_entropy = _finite_array([row["end_entropy"] for row in rows], "end_entropy", ndim=1)
    concentrated_span = -0.5 * (start_entropy + end_entropy)

    relevance_z = _standardize(relevance)
    span_z = _standardize(span)
    margin_z = _standardize(margin)
    concentration_z = _standardize(concentrated_span)
    # Agreement is intentionally represented separately from span strength:
    # a sharp span that contradicts the relevance head should be gated rather
    # than allowed to hijack the anchor score.
    disagreement = span_z - relevance_z
    features = np.column_stack((relevance_z, span_z + 0.5 * concentration_z, margin_z, disagreement))
    return np.clip(features / 3.0, -1.0, 1.0)


def _features(value: Any) -> np.ndarray:
    array = _finite_array(value, "features", ndim=2)
    if array.shape[1] != N_QUBITS:
        raise QARCGError(f"features must have {N_QUBITS} columns")
    if np.any(array < -1.0000001) or np.any(array > 1.0000001):
        raise QARCGError("encoded Reader features must be in [-1, 1]")
    return np.clip(array, -1.0, 1.0)


@dataclass(frozen=True)
class QARCGConfig:
    """Immutable circuit and scoring contract."""

    n_qubits: int = N_QUBITS
    depth: int = DEFAULT_DEPTH
    residual_bound: float = DEFAULT_RESIDUAL_BOUND
    gate_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.n_qubits != N_QUBITS:
            raise QARCGError(f"Q-ARCG currently fixes n_qubits={N_QUBITS}")
        if self.depth < 1 or self.depth > 4:
            raise QARCGError("depth must be in [1, 4]")
        if not math.isfinite(self.residual_bound) or self.residual_bound <= 0.0:
            raise QARCGError("residual_bound must be positive and finite")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0.0:
            raise QARCGError("gate_temperature must be positive and finite")


@dataclass
class QARCGParameters:
    """Trainable circuit/readout parameters.

    ``residual_scale=0`` is the exact null initialization.  A non-zero
    ``residual_bias`` gives this scale a usable gradient in a Reader training
    step without changing the initial output.
    """

    circuit: np.ndarray
    gate_weights: np.ndarray
    gate_bias: float
    residual_weights: np.ndarray
    residual_bias: float
    residual_scale: float

    def copy(self) -> "QARCGParameters":
        return QARCGParameters(
            circuit=self.circuit.copy(),
            gate_weights=self.gate_weights.copy(),
            gate_bias=float(self.gate_bias),
            residual_weights=self.residual_weights.copy(),
            residual_bias=float(self.residual_bias),
            residual_scale=float(self.residual_scale),
        )


def parameter_count(config: QARCGConfig | None = None) -> int:
    """Return the trainable parameter budget for both Q and classical arms."""

    cfg = config or QARCGConfig()
    return (cfg.depth * N_QUBITS * 2) + OBSERVABLE_COUNT + 1 + OBSERVABLE_COUNT + 1 + 1


def initial_parameters(config: QARCGConfig | None = None) -> QARCGParameters:
    """Create identity-initialized parameters with deterministic values."""

    cfg = config or QARCGConfig()
    gate_initial = np.linspace(-0.03, 0.03, OBSERVABLE_COUNT, dtype=np.float64)
    residual_initial = np.linspace(0.02, -0.02, OBSERVABLE_COUNT, dtype=np.float64)
    return QARCGParameters(
        circuit=np.zeros((cfg.depth, N_QUBITS, 2), dtype=np.float64),
        gate_weights=gate_initial,
        gate_bias=-4.0,
        residual_weights=residual_initial,
        residual_bias=0.1,
        residual_scale=0.0,
    )


def _validate_parameters(parameters: QARCGParameters, config: QARCGConfig) -> None:
    circuit = _finite_array(parameters.circuit, "circuit", ndim=3)
    if circuit.shape != (config.depth, N_QUBITS, 2):
        raise QARCGError("circuit has an invalid shape")
    for name in ("gate_weights", "residual_weights"):
        value = _finite_array(getattr(parameters, name), name, ndim=1)
        if value.shape != (OBSERVABLE_COUNT,):
            raise QARCGError(f"{name} has an invalid shape")
    for name in ("gate_bias", "residual_bias", "residual_scale"):
        if not math.isfinite(float(getattr(parameters, name))):
            raise QARCGError(f"{name} is non-finite")


def _apply_ry(state: np.ndarray, theta: float, wire: int) -> np.ndarray:
    result = state.copy()
    bit = 1 << wire
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    for lower in range(state.size):
        if lower & bit:
            continue
        upper = lower | bit
        alpha, beta = state[lower], state[upper]
        result[lower] = cosine * alpha - sine * beta
        result[upper] = sine * alpha + cosine * beta
    return result


def _apply_rz(state: np.ndarray, theta: float, wire: int) -> np.ndarray:
    result = state.copy()
    bit = 1 << wire
    phase = math.cos(theta / 2.0) + 1j * math.sin(theta / 2.0)
    for index in range(state.size):
        result[index] = state[index] * (phase if index & bit else np.conjugate(phase))
    return result


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    result = state.copy()
    control_bit, target_bit = 1 << control, 1 << target
    for index in range(state.size):
        if index & control_bit and not index & target_bit:
            paired = index | target_bit
            result[index], result[paired] = state[paired], state[index]
    return result


def _z_expectation(probabilities: np.ndarray, mask: int) -> float:
    total = 0.0
    for basis, probability in enumerate(probabilities):
        parity = (basis & mask).bit_count() & 1
        total += float(probability) * (-1.0 if parity else 1.0)
    return total


def quantum_observables(features: np.ndarray, parameters: QARCGParameters, config: QARCGConfig | None = None) -> np.ndarray:
    """Evaluate local-Z and ring-ZZ observables with an exact statevector."""

    cfg = config or QARCGConfig()
    _validate_parameters(parameters, cfg)
    matrix = _features(features)
    rows = np.empty((len(matrix), OBSERVABLE_COUNT), dtype=np.float64)
    for row_index, row in enumerate(matrix):
        state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
        state[0] = 1.0
        for wire, value in enumerate(row):
            state = _apply_ry(state, math.acos(float(value)), wire)
        for layer in range(cfg.depth):
            for wire in range(N_QUBITS):
                state = _apply_rz(state, float(parameters.circuit[layer, wire, 0]), wire)
                state = _apply_ry(state, float(parameters.circuit[layer, wire, 1]), wire)
            for control, target in RING:
                state = _apply_cnot(state, control, target)
        probabilities = np.abs(state) ** 2
        if abs(float(probabilities.sum()) - 1.0) > 1.0e-10:
            raise QARCGError("statevector norm drifted")
        local = [_z_expectation(probabilities, 1 << wire) for wire in range(N_QUBITS)]
        pair = [_z_expectation(probabilities, (1 << left) | (1 << right)) for left, right in RING]
        rows[row_index] = local + pair
    return _finite_array(rows, "quantum observables", ndim=2)


def classical_matched_observables(features: np.ndarray, parameters: QARCGParameters, config: QARCGConfig | None = None) -> np.ndarray:
    """A same-budget classical control with diagonal feature rotations.

    The control has exactly the same number of trainable parameters and the
    same readout.  It replaces each quantum layer with a bounded diagonal
    affine map, then exposes the same local/pairwise feature channels.  It is
    not claimed to be the only classical competitor; it is the required
    parameter-budget control for the first integration screen.
    """

    cfg = config or QARCGConfig()
    _validate_parameters(parameters, cfg)
    matrix = _features(features)
    local = matrix.copy()
    for layer in range(cfg.depth):
        scale = 1.0 + np.tanh(parameters.circuit[layer, :, 0])
        shift = np.tanh(parameters.circuit[layer, :, 1])
        local = np.tanh(local * scale[None, :] + shift[None, :])
    pair = local * np.roll(local, -1, axis=1)
    # RING order is [0-1, 1-2, 2-3, 3-0], matching the quantum arm.
    pair = pair[:, [0, 1, 2, 3]]
    return _finite_array(np.column_stack((local, pair)), "classical observables", ndim=2)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_from_observables(
    base_scores: Any,
    observables: np.ndarray,
    parameters: QARCGParameters,
    config: QARCGConfig,
) -> dict[str, np.ndarray]:
    base = _finite_array(base_scores, "base_scores", ndim=1)
    if len(base) != len(observables):
        raise QARCGError("base_scores and features must have the same row count")
    temperature = config.gate_temperature
    gate_logits = (observables @ parameters.gate_weights + parameters.gate_bias) / temperature
    gate = _sigmoid(gate_logits)
    raw = observables @ parameters.residual_weights + parameters.residual_bias
    scale = math.tanh(float(parameters.residual_scale))
    score_spread = max(float(base.std()), 1.0e-6)
    residual = config.residual_bound * score_spread * scale * np.tanh(raw)
    final = base + gate * residual
    return {
        "scores": _finite_array(final, "final_scores", ndim=1),
        "gate": _finite_array(gate, "applicability_gate", ndim=1),
        "residual": _finite_array(residual, "bounded_residual", ndim=1),
        "gate_logits": _finite_array(gate_logits, "gate_logits", ndim=1),
        "observables": _finite_array(observables, "observables", ndim=2),
    }


def score_qarcg(base_scores: Any, features: Any, parameters: QARCGParameters | None = None, config: QARCGConfig | None = None) -> dict[str, np.ndarray]:
    """Score candidates with the Q-ARCG quantum arm."""

    cfg = config or QARCGConfig()
    params = parameters or initial_parameters(cfg)
    matrix = _features(features)
    return _score_from_observables(base_scores, quantum_observables(matrix, params, cfg), params, cfg)


def score_classical_control(base_scores: Any, features: Any, parameters: QARCGParameters | None = None, config: QARCGConfig | None = None) -> dict[str, np.ndarray]:
    """Score candidates with the matched classical control arm."""

    cfg = config or QARCGConfig()
    params = parameters or initial_parameters(cfg)
    matrix = _features(features)
    return _score_from_observables(base_scores, classical_matched_observables(matrix, params, cfg), params, cfg)


def rank_indices(scores: Any, *, k: int = 5, retrieved_ranks: Sequence[int] | None = None, candidate_ids: Sequence[str] | None = None) -> list[int]:
    """Stable Top-k ranking for a fixed candidate cohort."""

    values = _finite_array(scores, "scores", ndim=1)
    if k < 1 or k > len(values):
        raise QARCGError("k must be between 1 and the number of candidates")
    ranks = list(range(len(values))) if retrieved_ranks is None else list(retrieved_ranks)
    ids = [str(index) for index in range(len(values))] if candidate_ids is None else [str(value) for value in candidate_ids]
    if len(ranks) != len(values) or len(ids) != len(values):
        raise QARCGError("tie-break fields must match score length")
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), int(ranks[index]), ids[index]))[:k]


def torch_pairwise_rank_loss(scores: Any, positive_mask: Any, margin: float = 0.2) -> Any:
    """Training-only pairwise loss; labels never enter the online scoring API."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on collaborator env
        raise QARCGError("torch is required for the training loss") from exc
    values = scores.reshape(-1)
    mask = positive_mask.to(dtype=torch.bool).reshape(-1)
    positives, negatives = values[mask], values[~mask]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise QARCGError("pairwise loss needs both positive and negative candidates")
    differences = positives[:, None] - negatives[None, :]
    return torch.nn.functional.softplus(float(margin) - differences).mean()


def make_torch_qarcg(config: QARCGConfig | None = None) -> Any:
    """Build a differentiable Torch statevector head for Reader integration.

    Importing this module does not require Torch.  The returned module uses
    the same four-qubit circuit, readout, zero-residual initialization, and
    parameter count as the NumPy contract above.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on collaborator env
        raise QARCGError("torch is required for the Reader-integrated head") from exc
    cfg = config or QARCGConfig()

    class TorchQARCG(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.circuit = torch.nn.Parameter(torch.zeros(cfg.depth, N_QUBITS, 2))
            self.gate_weights = torch.nn.Parameter(torch.linspace(-0.03, 0.03, OBSERVABLE_COUNT))
            self.gate_bias = torch.nn.Parameter(torch.tensor(-4.0))
            self.residual_weights = torch.nn.Parameter(torch.linspace(0.02, -0.02, OBSERVABLE_COUNT))
            self.residual_bias = torch.nn.Parameter(torch.tensor(0.1))
            self.residual_scale = torch.nn.Parameter(torch.tensor(0.0))

        @staticmethod
        def _ry(state: Any, theta: Any, wire: int) -> Any:
            bit = 1 << wire
            low = [index for index in range(1 << N_QUBITS) if not index & bit]
            high = [index | bit for index in low]
            alpha, beta = state[:, low], state[:, high]
            result = state.clone()
            cosine, sine = torch.cos(theta / 2.0), torch.sin(theta / 2.0)
            result[:, low] = cosine[:, None] * alpha - sine[:, None] * beta
            result[:, high] = sine[:, None] * alpha + cosine[:, None] * beta
            return result

        @staticmethod
        def _rz(state: Any, theta: Any, wire: int) -> Any:
            bit = 1 << wire
            phase = torch.polar(torch.ones_like(theta), theta / 2.0)
            phases = torch.where(
                torch.tensor([(index & bit) != 0 for index in range(1 << N_QUBITS)], device=state.device),
                phase[:, None],
                torch.conj(phase)[:, None],
            )
            return state * phases

        @staticmethod
        def _cnot(state: Any, control: int, target: int) -> Any:
            permutation = list(range(1 << N_QUBITS))
            control_bit, target_bit = 1 << control, 1 << target
            for index in range(1 << N_QUBITS):
                if index & control_bit and not index & target_bit:
                    paired = index | target_bit
                    permutation[index], permutation[paired] = permutation[paired], permutation[index]
            return state[:, permutation]

        @staticmethod
        def _expectation(probabilities: Any, mask: int) -> Any:
            signs = torch.tensor(
                [(-1.0 if (index & mask).bit_count() & 1 else 1.0) for index in range(1 << N_QUBITS)],
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            return (probabilities * signs[None, :]).sum(dim=1)

        def _observables(self, features: Any) -> Any:
            values = torch.clamp(features.float(), -1.0, 1.0)
            state = torch.zeros((values.shape[0], 1 << N_QUBITS), dtype=torch.complex64, device=values.device)
            state[:, 0] = 1.0 + 0.0j
            for wire in range(N_QUBITS):
                state = self._ry(state, torch.acos(values[:, wire]), wire)
            for layer in range(cfg.depth):
                for wire in range(N_QUBITS):
                    state = self._rz(state, self.circuit[layer, wire, 0].expand(values.shape[0]), wire)
                    state = self._ry(state, self.circuit[layer, wire, 1].expand(values.shape[0]), wire)
                for control, target in RING:
                    state = self._cnot(state, control, target)
            probabilities = state.abs().square()
            local = [self._expectation(probabilities, 1 << wire) for wire in range(N_QUBITS)]
            pair = [self._expectation(probabilities, (1 << left) | (1 << right)) for left, right in RING]
            return torch.stack(local + pair, dim=1)

        def forward(self, base_scores: Any, features: Any) -> tuple[Any, Any, Any, Any]:
            base = base_scores.reshape(-1).float()
            matrix = features.float()
            if matrix.ndim != 2 or matrix.shape[1] != N_QUBITS or matrix.shape[0] != base.shape[0]:
                raise QARCGError("Torch Q-ARCG expects an N x 4 feature matrix")
            observables = self._observables(matrix)
            gate_logits = (observables @ self.gate_weights + self.gate_bias) / cfg.gate_temperature
            gate = torch.sigmoid(gate_logits)
            raw = observables @ self.residual_weights + self.residual_bias
            spread = torch.clamp(base.std(unbiased=False), min=1.0e-6)
            residual = cfg.residual_bound * spread * torch.tanh(self.residual_scale) * torch.tanh(raw)
            return base + gate * residual, gate, residual, observables

        def from_reader_logits(
            self,
            relevance_logits: Any,
            start_logits: Any,
            end_logits: Any,
            passage_mask: Any,
            *,
            max_answer_tokens: int = 10,
        ) -> tuple[Any, Any, Any, Any, Any]:
            features, span_summary = torch_reader_features(
                relevance_logits,
                start_logits,
                end_logits,
                passage_mask,
                max_answer_tokens=max_answer_tokens,
            )
            scores, gate, residual, observables = self(relevance_logits, features)
            return scores, gate, residual, observables, span_summary

    return TorchQARCG()


def make_torch_classical_control(config: QARCGConfig | None = None) -> Any:
    """Build a differentiable classical control with the exact same budget."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on collaborator env
        raise QARCGError("torch is required for the matched classical control") from exc
    cfg = config or QARCGConfig()

    class TorchClassicalARCG(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.circuit = torch.nn.Parameter(torch.zeros(cfg.depth, N_QUBITS, 2))
            self.gate_weights = torch.nn.Parameter(torch.linspace(-0.03, 0.03, OBSERVABLE_COUNT))
            self.gate_bias = torch.nn.Parameter(torch.tensor(-4.0))
            self.residual_weights = torch.nn.Parameter(torch.linspace(0.02, -0.02, OBSERVABLE_COUNT))
            self.residual_bias = torch.nn.Parameter(torch.tensor(0.1))
            self.residual_scale = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, base_scores: Any, features: Any) -> tuple[Any, Any, Any, Any]:
            base = base_scores.reshape(-1).float()
            local = torch.clamp(features.float(), -1.0, 1.0)
            if local.ndim != 2 or local.shape != (base.shape[0], N_QUBITS):
                raise QARCGError("classical control expects an N x 4 feature matrix")
            for layer in range(cfg.depth):
                scale = 1.0 + torch.tanh(self.circuit[layer, :, 0])
                shift = torch.tanh(self.circuit[layer, :, 1])
                local = torch.tanh(local * scale[None, :] + shift[None, :])
            pair = local * torch.roll(local, shifts=-1, dims=1)
            observables = torch.cat((local, pair), dim=1)
            gate_logits = (observables @ self.gate_weights + self.gate_bias) / cfg.gate_temperature
            gate = torch.sigmoid(gate_logits)
            raw = observables @ self.residual_weights + self.residual_bias
            spread = torch.clamp(base.std(unbiased=False), min=1.0e-6)
            residual = cfg.residual_bound * spread * torch.tanh(self.residual_scale) * torch.tanh(raw)
            return base + gate * residual, gate, residual, observables

    return TorchClassicalARCG()


def make_torch_dpr_reader_adapter(reader: Any, config: QARCGConfig | None = None) -> Any:
    """Wrap a Hugging Face DPRReader so Q-ARCG is in its forward score path.

    One batch row must correspond to one question-passage candidate. The
    caller supplies a passage-only token mask; it must exclude question and
    special tokens. The returned mapping keeps raw Reader outputs available
    for the original span task while exposing calibrated per-passage scores.
    Training labels are intentionally not accepted by this adapter. Compute a
    ranking loss separately with ``torch_pairwise_rank_loss``.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on collaborator env
        raise QARCGError("torch is required for the DPR Reader adapter") from exc
    cfg = config or QARCGConfig()
    head = make_torch_qarcg(cfg)

    class TorchDPRReaderARCG(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reader = reader
            self.head = head

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            passage_token_mask: Any,
            token_type_ids: Any = None,
        ) -> dict[str, Any]:
            if token_type_ids is None:
                outputs = self.reader(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            else:
                outputs = self.reader(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
            if any(not hasattr(outputs, name) for name in ("relevance_logits", "start_logits", "end_logits")):
                raise QARCGError("DPR Reader output is missing relevance/start/end logits")
            scores, gate, residual, observables, summary = self.head.from_reader_logits(
                outputs.relevance_logits,
                outputs.start_logits,
                outputs.end_logits,
                passage_token_mask,
            )
            return {
                "reader_outputs": outputs,
                "scores": scores,
                "applicability_gate": gate,
                "bounded_residual": residual,
                "quantum_observables": observables,
                "span_summary": summary,
            }

    return TorchDPRReaderARCG()


def torch_reader_features(
    relevance_logits: Any,
    start_logits: Any,
    end_logits: Any,
    passage_mask: Any,
    *,
    max_answer_tokens: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Differentiably derive candidate features from a batched DPR Reader pass.

    Each row is one question-passage pair. ``passage_mask`` must identify only
    tokens belonging to that passage, excluding question and special tokens.
    Rows in the call must belong to one question-level candidate group so the
    normalization cannot mix unrelated questions.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on collaborator env
        raise QARCGError("torch is required to derive features from Reader logits") from exc
    relevance = relevance_logits.reshape(-1).float()
    starts, ends = start_logits.float(), end_logits.float()
    mask = passage_mask.to(dtype=torch.bool)
    if starts.ndim != 2 or ends.shape != starts.shape or mask.shape != starts.shape:
        raise QARCGError("start/end logits and passage_mask must have matching N x L shapes")
    if starts.shape[0] != relevance.shape[0] or max_answer_tokens < 1:
        raise QARCGError("Reader rows or max_answer_tokens are invalid")
    if torch.any(mask.sum(dim=1) == 0):
        raise QARCGError("every passage must expose at least one passage token")

    length = starts.shape[1]
    positions = torch.arange(length, device=starts.device)
    span_mask = (
        mask[:, :, None]
        & mask[:, None, :]
        & (positions[None, :, None] <= positions[None, None, :])
        & ((positions[None, None, :] - positions[None, :, None]) < max_answer_tokens)
    )
    span_scores = starts[:, :, None] + ends[:, None, :]
    span_scores = span_scores.masked_fill(~span_mask, torch.finfo(span_scores.dtype).min)
    top_spans = torch.topk(span_scores.reshape(starts.shape[0], -1), k=min(2, length * length), dim=1).values
    best_span = top_spans[:, 0]
    second_span = top_spans[:, 1] if top_spans.shape[1] > 1 else best_span
    has_second = span_mask.reshape(starts.shape[0], -1).sum(dim=1) > 1
    span_margin = torch.where(has_second, best_span - second_span, torch.zeros_like(best_span))

    def entropy(logits: Any) -> Any:
        values = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(values, dim=1)
        return -(probabilities * torch.log(torch.clamp(probabilities, min=1.0e-12))).sum(dim=1)

    start_entropy, end_entropy = entropy(starts), entropy(ends)

    def standardize(values: Any) -> Any:
        spread = values.std(unbiased=False)
        return torch.where(spread > 1.0e-12, (values - values.mean()) / torch.clamp(spread, min=1.0e-12), torch.zeros_like(values))

    relevance_z = standardize(relevance)
    span_z = standardize(best_span)
    margin_z = standardize(span_margin)
    concentration_z = standardize(-0.5 * (start_entropy + end_entropy))
    features = torch.stack((relevance_z, span_z + 0.5 * concentration_z, margin_z, span_z - relevance_z), dim=1)
    features = torch.clamp(features / 3.0, -1.0, 1.0)
    return features, {
        "best_span": best_span,
        "span_margin": span_margin,
        "start_entropy": start_entropy,
        "end_entropy": end_entropy,
    }


__all__ = [
    "FEATURE_NAMES",
    "OBSERVABLE_COUNT",
    "QARCGConfig",
    "QARCGError",
    "QARCGParameters",
    "RING",
    "build_reader_features",
    "classical_matched_observables",
    "initial_parameters",
    "make_torch_qarcg",
    "make_torch_classical_control",
    "make_torch_dpr_reader_adapter",
    "parameter_count",
    "quantum_observables",
    "rank_indices",
    "score_classical_control",
    "score_qarcg",
    "torch_pairwise_rank_loss",
    "torch_reader_features",
]
