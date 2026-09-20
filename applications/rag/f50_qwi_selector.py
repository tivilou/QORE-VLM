"""Full-50 fixed-unitary quantum-walk interference selector.

This module implements the deliberately limited F50-QWI mechanism audit: a
single-particle state lives in a six-qubit (64-state) register, with exactly
50 passage basis states and 14 isolated invalid states.  It produces passage
marginal Born probabilities, then ranks them to select five passages.  It is
an exact classical linear-algebra emulator of the specified unitary; it does
not represent a joint five-passage quantum state or claim quantum advantage.

Selection accepts only ordinary frozen Top-50 fields.  Evidence labels, gold
answers, generations, and evaluator fields are rejected at the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any

import numpy as np


SELECTOR_ID = "f50_qwi"
SELECTOR_VERSION = "1.0"
DEFAULT_K = 5
FULL_TOP50 = 50
QUBITS = 6
REGISTER_DIMENSION = 1 << QUBITS
INVALID_STATES = REGISTER_DIMENSION - FULL_TOP50
VARIANTS = (
    "born_exact",
    "real_diffusion_control",
    "dephased_control",
    "phase_scramble_control",
)

_FORBIDDEN_FIELDS = {
    "evidence", "gold", "gold_answers", "answer", "answers", "prediction",
    "metrics", "generation", "generation_panel", "generation_consensus",
    "evaluator", "evaluation", "label", "labels", "selectors", "diagnostics",
}
_ALLOWED_FIELDS = {
    "id", "passage_id", "retrieved_rank", "rank",
    "retrieval_score", "answer_scorer_score", "answer_scorer", "dpr_passage_embedding",
    "passage_embedding", "embedding",
}


class F50QWIError(ValueError):
    """Raised when a F50-QWI input or numerical invariant is invalid."""


@dataclass(frozen=True)
class F50QWIConfig:
    """Frozen, label-free mechanism parameters.

    The parameter values are design constants, not values tuned on the Silver
    oracle.  Retrieval and Answer Scorer scores are independently min-max
    normalized within the fixed Top-50, averaged, then used both for the
    initial population weights and the diagonal potential.
    """

    n_qubits: int = QUBITS
    valid_states: int = FULL_TOP50
    diagonal_strength: float = 1.0
    interaction_strength: float = 0.85
    evolution_time: float = 1.0
    initial_temperature: float = 1.0
    retrieval_weight: float = 0.5
    answer_scorer_weight: float = 0.5
    phase_seed: int = 42
    tolerance: float = 1.0e-10


@dataclass(frozen=True)
class OnlineCandidate:
    passage_id: str
    retrieved_rank: int
    retrieval_score: float
    answer_scorer_score: float
    embedding: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    variant: str
    selected_indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    initial_weights: tuple[float, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MechanismAudit:
    passed: bool
    checks: dict[str, Any]
    variants: dict[str, SelectionResult]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F50QWIError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise F50QWIError(f"{field} must be a finite number")
    return result


def _embedding(value: Any) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise F50QWIError("passage embedding must be a one-dimensional numeric sequence")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise F50QWIError("passage embedding must be numeric") from exc
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise F50QWIError("passage embedding must be non-empty, one-dimensional, and finite")
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise F50QWIError("passage embedding must have non-zero norm")
    return result / norm


def _candidate(value: OnlineCandidate | Mapping[str, Any]) -> OnlineCandidate:
    if isinstance(value, OnlineCandidate):
        return value
    if not isinstance(value, Mapping):
        raise F50QWIError("candidate must be an object")
    keys = {str(key) for key in value}
    forbidden = sorted(keys & _FORBIDDEN_FIELDS)
    if forbidden:
        raise F50QWIError(f"forbidden selection fields: {', '.join(forbidden)}")
    unknown = sorted(keys - _ALLOWED_FIELDS)
    if unknown:
        raise F50QWIError(f"unknown selection fields: {', '.join(unknown)}")
    identifier = value.get("id", value.get("passage_id"))
    if not isinstance(identifier, (str, int)) or not str(identifier).strip():
        raise F50QWIError("candidate id must be non-empty")
    rank = value.get("retrieved_rank", value.get("rank"))
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise F50QWIError("retrieved_rank must be a positive integer")
    answer_score = value.get("answer_scorer_score", value.get("answer_scorer"))
    if answer_score is None:
        raise F50QWIError("answer_scorer_score is required")
    embedding = value.get("dpr_passage_embedding")
    if embedding is None:
        embedding = value.get("passage_embedding", value.get("embedding"))
    if embedding is None:
        raise F50QWIError("dpr_passage_embedding is required")
    return OnlineCandidate(
        passage_id=str(identifier),
        retrieved_rank=rank,
        retrieval_score=_finite(value.get("retrieval_score"), "retrieval_score"),
        answer_scorer_score=_finite(answer_score, "answer_scorer_score"),
        embedding=_embedding(embedding),
    )


def coerce_candidates(values: Sequence[OnlineCandidate | Mapping[str, Any]]) -> tuple[OnlineCandidate, ...]:
    candidates = tuple(_candidate(value) for value in values)
    if len(candidates) != FULL_TOP50:
        raise F50QWIError(f"F50-QWI requires exactly {FULL_TOP50} candidates")
    identifiers = [candidate.passage_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise F50QWIError("candidate passage IDs must be unique")
    ranks = [candidate.retrieved_rank for candidate in candidates]
    if ranks != list(range(1, FULL_TOP50 + 1)):
        raise F50QWIError("retrieved ranks must be the ordered sequence 1 through 50")
    dimensions = {candidate.embedding.size for candidate in candidates}
    if len(dimensions) != 1:
        raise F50QWIError("all passage embeddings must have the same dimension")
    return candidates


def _validate_config(config: F50QWIConfig) -> None:
    if config.n_qubits != QUBITS:
        raise F50QWIError(f"F50-QWI requires exactly {QUBITS} qubits")
    if config.valid_states != FULL_TOP50:
        raise F50QWIError(f"F50-QWI requires exactly {FULL_TOP50} valid basis states")
    if (1 << config.n_qubits) != REGISTER_DIMENSION:
        raise F50QWIError("six-qubit register dimension mismatch")
    for field in (
        "diagonal_strength", "interaction_strength", "evolution_time", "initial_temperature",
        "retrieval_weight", "answer_scorer_weight", "tolerance",
    ):
        value = _finite(getattr(config, field), field)
        if field in {"initial_temperature", "tolerance"} and value <= 0.0:
            raise F50QWIError(f"{field} must be positive")
    if isinstance(config.phase_seed, bool) or not isinstance(config.phase_seed, int):
        raise F50QWIError("phase_seed must be an integer")
    if config.retrieval_weight < 0.0 or config.answer_scorer_weight < 0.0:
        raise F50QWIError("score weights must be non-negative")
    if config.retrieval_weight + config.answer_scorer_weight <= 0.0:
        raise F50QWIError("at least one score weight must be positive")


def _minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= np.finfo(np.float64).eps:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def _score_terms(candidates: Sequence[OnlineCandidate], config: F50QWIConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retrieval = _minmax(np.asarray([candidate.retrieval_score for candidate in candidates], dtype=np.float64))
    answer = _minmax(np.asarray([candidate.answer_scorer_score for candidate in candidates], dtype=np.float64))
    weight_sum = config.retrieval_weight + config.answer_scorer_weight
    relevance = (
        config.retrieval_weight * retrieval + config.answer_scorer_weight * answer
    ) / weight_sum
    logits = relevance / config.initial_temperature
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights /= float(np.sum(weights))
    potential = relevance - float(np.mean(relevance))
    potential_scale = float(np.max(np.abs(potential)))
    if potential_scale > config.tolerance:
        potential /= potential_scale
    else:
        potential.fill(0.0)
    return relevance, weights, potential


def _centered_similarity(candidates: Sequence[OnlineCandidate], config: F50QWIConfig) -> np.ndarray:
    embeddings = np.stack([candidate.embedding for candidate in candidates], axis=0)
    similarity = embeddings @ embeddings.T
    similarity = 0.5 * (similarity + similarity.T)
    off_diagonal = ~np.eye(FULL_TOP50, dtype=bool)
    centered = np.zeros_like(similarity)
    centered[off_diagonal] = similarity[off_diagonal] - float(np.mean(similarity[off_diagonal]))
    centered = 0.5 * (centered + centered.T)
    eigenvalues = np.linalg.eigvalsh(centered)
    norm = float(np.max(np.abs(eigenvalues)))
    if norm > config.tolerance:
        centered /= norm
    else:
        centered.fill(0.0)
    return centered


def build_hamiltonian(
    values: Sequence[OnlineCandidate | Mapping[str, Any]],
    *,
    config: F50QWIConfig = F50QWIConfig(),
) -> tuple[np.ndarray, tuple[OnlineCandidate, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Build the fixed block-preserving 64 x 64 Hermitian Hamiltonian."""

    _validate_config(config)
    candidates = coerce_candidates(values)
    relevance, weights, potential = _score_terms(candidates, config)
    coupling = _centered_similarity(candidates, config)
    valid_block = (
        config.diagonal_strength * np.diag(potential)
        + config.interaction_strength * coupling
    )
    valid_block = 0.5 * (valid_block + valid_block.T)
    hamiltonian = np.zeros((REGISTER_DIMENSION, REGISTER_DIMENSION), dtype=np.complex128)
    hamiltonian[:FULL_TOP50, :FULL_TOP50] = valid_block
    return hamiltonian, candidates, relevance, weights, coupling


def _unitary(hamiltonian: np.ndarray, evolution_time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * evolution_time * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _phase(candidate_id: str, seed: int) -> float:
    payload = f"{seed}:{candidate_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)
    return 2.0 * math.pi * (integer / float(1 << 64))


def _real_diffusion(weights: np.ndarray, coupling: np.ndarray, config: F50QWIConfig) -> np.ndarray:
    """Matched classical control on the same normalized passage graph."""

    affinity = np.maximum(coupling, 0.0)
    np.fill_diagonal(affinity, 0.0)
    laplacian = np.diag(np.sum(affinity, axis=1)) - affinity
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    diffusion = (eigenvectors * np.exp(-config.evolution_time * eigenvalues)) @ eigenvectors.T
    probabilities = np.asarray(diffusion @ weights, dtype=np.float64)
    if float(np.min(probabilities)) < -config.tolerance:
        raise F50QWIError("real diffusion produced a negative population")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise F50QWIError("real diffusion lost all population")
    return probabilities / total


def _probabilities(
    variant: str,
    candidates: Sequence[OnlineCandidate],
    hamiltonian: np.ndarray,
    weights: np.ndarray,
    coupling: np.ndarray,
    config: F50QWIConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    unitary = _unitary(hamiltonian, config.evolution_time)
    state = np.zeros(REGISTER_DIMENSION, dtype=np.complex128)
    state[:FULL_TOP50] = np.sqrt(weights)
    if variant == "born_exact":
        evolved = unitary @ state
        full_probabilities = np.abs(evolved) ** 2
    elif variant == "dephased_control":
        initial_population = np.zeros(REGISTER_DIMENSION, dtype=np.float64)
        initial_population[:FULL_TOP50] = weights
        full_probabilities = (np.abs(unitary) ** 2) @ initial_population
    elif variant == "phase_scramble_control":
        state[:FULL_TOP50] *= np.exp(1j * np.asarray([_phase(item.passage_id, config.phase_seed) for item in candidates]))
        evolved = unitary @ state
        full_probabilities = np.abs(evolved) ** 2
    elif variant == "real_diffusion_control":
        full_probabilities = np.zeros(REGISTER_DIMENSION, dtype=np.float64)
        full_probabilities[:FULL_TOP50] = _real_diffusion(weights, coupling, config)
    else:
        raise F50QWIError(f"unsupported F50-QWI variant: {variant}")
    full_probabilities = np.real_if_close(full_probabilities, tol=1000).astype(np.float64)
    if float(np.min(full_probabilities)) < -config.tolerance:
        raise F50QWIError("probability distribution contains a material negative value")
    full_probabilities = np.maximum(full_probabilities, 0.0)
    return full_probabilities, {
        "unitary_residual": float(np.max(np.abs(unitary.conj().T @ unitary - np.eye(REGISTER_DIMENSION)))),
        "initial_state_normalization_error": abs(float(np.vdot(state, state).real) - 1.0),
    }


def select(
    values: Sequence[OnlineCandidate | Mapping[str, Any]],
    *,
    k: int = DEFAULT_K,
    variant: str = "born_exact",
    config: F50QWIConfig = F50QWIConfig(),
) -> SelectionResult:
    """Return the deterministic Top-k passage marginals for one fixed Top-50."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > FULL_TOP50:
        raise F50QWIError(f"k must be an integer in [1, {FULL_TOP50}]")
    hamiltonian, candidates, relevance, weights, coupling = build_hamiltonian(values, config=config)
    probabilities, numerical = _probabilities(variant, candidates, hamiltonian, weights, coupling, config)
    valid_probabilities = probabilities[:FULL_TOP50]
    selected = sorted(
        range(FULL_TOP50),
        key=lambda index: (-float(valid_probabilities[index]), candidates[index].retrieved_rank, candidates[index].passage_id),
    )[:k]
    hermitian_residual = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
    invalid_mass = float(np.sum(probabilities[FULL_TOP50:]))
    block_leakage = float(
        max(
            np.max(np.abs(hamiltonian[:FULL_TOP50, FULL_TOP50:])),
            np.max(np.abs(hamiltonian[FULL_TOP50:, :FULL_TOP50])),
        )
    )
    return SelectionResult(
        variant=variant,
        selected_indices=tuple(selected),
        probabilities=tuple(float(value) for value in valid_probabilities),
        initial_weights=tuple(float(value) for value in weights),
        diagnostics={
            "register_dimension": REGISTER_DIMENSION,
            "valid_state_count": FULL_TOP50,
            "invalid_state_count": INVALID_STATES,
            "single_particle_marginal_ranking": True,
            "classically_exactly_simulated": True,
            "relevance_min": float(np.min(relevance)),
            "relevance_max": float(np.max(relevance)),
            "coupling_spectral_norm": float(np.max(np.abs(np.linalg.eigvalsh(coupling)))),
            "hamiltonian_hermitian_residual": hermitian_residual,
            "block_leakage_max_abs": block_leakage,
            "invalid_state_probability_mass": invalid_mass,
            "probability_normalization_error": abs(float(np.sum(probabilities)) - 1.0),
            **numerical,
        },
    )


def mechanism_audit(
    values: Sequence[OnlineCandidate | Mapping[str, Any]],
    *,
    config: F50QWIConfig = F50QWIConfig(),
) -> MechanismAudit:
    """Evaluate numerical, identity, and matched-control checks without labels."""

    results = {variant: select(values, variant=variant, config=config) for variant in VARIANTS}
    tau_zero = select(values, variant="born_exact", config=replace(config, evolution_time=0.0))
    born = results["born_exact"]
    identity_error = max(
        abs(probability - weight)
        for probability, weight in zip(tau_zero.probabilities, tau_zero.initial_weights)
    )
    tolerance = config.tolerance
    checks = {
        "hamiltonian_hermitian": born.diagnostics["hamiltonian_hermitian_residual"] <= tolerance,
        "block_preserving": born.diagnostics["block_leakage_max_abs"] <= tolerance,
        "invalid_state_isolation": born.diagnostics["invalid_state_probability_mass"] <= tolerance,
        "probability_normalization": born.diagnostics["probability_normalization_error"] <= tolerance,
        "unitarity": born.diagnostics["unitary_residual"] <= tolerance * 10.0,
        "tau_zero_identity": identity_error <= tolerance * 10.0,
        "no_label_input": True,
        "born_differs_from_real_diffusion": born.selected_indices != results["real_diffusion_control"].selected_indices,
        "born_differs_from_dephased": born.selected_indices != results["dephased_control"].selected_indices,
        "born_differs_from_phase_scramble": born.selected_indices != results["phase_scramble_control"].selected_indices,
        "tau_zero_max_probability_error": identity_error,
    }
    passed = all(value for key, value in checks.items() if isinstance(value, bool) and key not in {"born_differs_from_real_diffusion", "born_differs_from_dephased", "born_differs_from_phase_scramble"})
    return MechanismAudit(passed=passed, checks=checks, variants=results)


__all__ = [
    "DEFAULT_K", "F50QWIConfig", "F50QWIError", "FULL_TOP50", "INVALID_STATES",
    "MechanismAudit", "OnlineCandidate", "QUBITS", "REGISTER_DIMENSION", "SELECTOR_ID",
    "SELECTOR_VERSION", "SelectionResult", "VARIANTS", "build_hamiltonian", "coerce_candidates",
    "mechanism_audit", "select",
]
