"""EnhancerPipeline: validate and compose QUBO enhancer plugins."""

import warnings
from time import perf_counter
from typing import Any

import numpy as np

from .base import QUBOEnhancer


class EnhancerPipeline:
    """
    Chain multiple enhancers in sequence.

    Each enhancer receives the output of the previous one, allowing ideas to be
    composed. For example:
        - Enhancer 1 (baseline): w = gamma * b
        - Enhancer 2 (idea4): w = w + alpha * integrity
        - Final: w = gamma * b + alpha * integrity

    The pipeline ensures enhancers are applied in the specified order.
    """

    def __init__(
        self,
        enhancers: list[QUBOEnhancer],
        *,
        strict_composition: bool = False,
    ):
        """
        Initialize the pipeline with a list of enhancers.

        Args:
            enhancers: List of QUBOEnhancer instances to apply in order.
        """
        if not enhancers:
            raise ValueError("Pipeline requires at least one enhancer")
        self.enhancers = enhancers
        self.strict_composition = strict_composition
        self._validate_composition()

    def _validate_composition(self) -> None:
        """Reject ambiguous roots in strict mode; warn for legacy callers."""
        root_positions = [
            index
            for index, enhancer in enumerate(self.enhancers)
            if enhancer.composition_mode == "replace"
        ]
        invalid_modes = [
            enhancer.name
            for enhancer in self.enhancers
            if enhancer.composition_mode not in {"add", "replace"}
        ]
        if invalid_modes:
            raise ValueError(
                "Enhancers have invalid composition_mode values: "
                + ", ".join(invalid_modes)
            )

        ambiguous = len(root_positions) > 1 or (
            root_positions and root_positions[0] != 0
        )
        if not ambiguous:
            return

        message = (
            "Ambiguous enhancer composition: a replace-mode enhancer discards "
            "terms produced before it. Use one root objective first, followed "
            "only by add-mode enhancers."
        )
        if self.strict_composition:
            raise ValueError(message)
        warnings.warn(message, DeprecationWarning, stacklevel=3)

    def enhance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply all enhancers in sequence.

        Args:
            a: (N,) quality scores.
            b: (N, N) redundancy matrix.
            context: Context dict for enhancers.

        Returns:
            w: (N, N) final interaction matrix after all enhancers.
        """
        w, _ = self._run(a, b, context, collect_diagnostics=False)
        return w

    def enhance_with_diagnostics(
        self,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Apply plugins and return their measured objective contributions."""
        return self._run(a, b, context, collect_diagnostics=True)

    def _run(
        self,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
        *,
        collect_diagnostics: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        N = len(a)
        if b.shape != (N, N):
            raise ValueError(f"b must have shape {(N, N)}, got {b.shape}")

        w = np.zeros((N, N), dtype=np.float64)
        trace: list[dict[str, Any]] = []

        for enhancer in self.enhancers:
            enhancer.validate_context(context, list(enhancer.required_context_keys))
            before = w.copy() if collect_diagnostics else None
            started = perf_counter() if collect_diagnostics else 0.0
            candidate = np.asarray(
                enhancer.enhance(w, a, b, context), dtype=np.float64
            )
            self._validate_matrix(enhancer.name, candidate, N)
            w = candidate
            if collect_diagnostics:
                assert before is not None
                trace.append({
                    "name": enhancer.name,
                    "mode": enhancer.composition_mode,
                    "elapsed_ms": (perf_counter() - started) * 1000.0,
                    "input_norm": float(np.linalg.norm(before)),
                    "output_norm": float(np.linalg.norm(w)),
                    "delta_norm": float(np.linalg.norm(w - before)),
                    "overwrote_nonzero_input": bool(
                        enhancer.composition_mode == "replace"
                        and np.any(before != 0.0)
                    ),
                })

        return w, trace

    @staticmethod
    def _validate_matrix(name: str, w: np.ndarray, N: int) -> None:
        if w.shape != (N, N):
            raise ValueError(
                f"Enhancer '{name}' returned shape {w.shape}; expected {(N, N)}"
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(f"Enhancer '{name}' returned non-finite values")
        if not np.allclose(w, w.T, atol=1e-10, rtol=1e-10):
            raise ValueError(f"Enhancer '{name}' returned a non-symmetric matrix")
        if not np.allclose(np.diag(w), 0.0, atol=1e-12, rtol=0.0):
            raise ValueError(f"Enhancer '{name}' returned a non-zero diagonal")

    def required_context_keys(self) -> tuple[str, ...]:
        keys = {
            key
            for enhancer in self.enhancers
            for key in enhancer.required_context_keys
        }
        return tuple(sorted(keys))

    def describe(self) -> str:
        """Return a readable, ASCII-safe description of the pipeline."""
        descriptions = [e.description() for e in self.enhancers]
        return " -> ".join(descriptions)

    def names(self) -> list[str]:
        """
        Get the names of all enhancers in this pipeline.

        Returns:
            List of enhancer names in order.
        """
        return [e.name for e in self.enhancers]
