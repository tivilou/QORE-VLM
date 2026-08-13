"""Cohesion-calibrated redundancy correction.

The plugin keeps the baseline QORE objective intact and adds a deterministic,
query-local redundancy term.  The calibrated weight follows the development
plan's normalized ``delta * (K - 1)`` scale law:

    delta_q = eta * m_q / ((K - 1) * (c_q + eps))

where ``m_q`` is the relevance gap around the K-th candidate and ``c_q`` is
the mean pairwise cohesion of the candidate pool.  It is intentionally an
additive plugin so it can be ablated without changing selector core logic.
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("cohesion")
class CohesionCalibrationEnhancer(QUBOEnhancer):
    """Add a fixed or query/pool-calibrated redundancy correction.

    Config keys:
        mode: ``"calibrated"`` (default), ``"fixed"``, or ``"disabled"``.
        eta: Scale for calibrated mode. Default ``0.1``.
        weight: Fixed additive weight. Default ``0.0``.
        eps: Denominator stabilizer. Default ``1e-6``.

    The plugin only uses deterministic values already available to the
    selector.  It does not inspect gold labels, answer text, or evaluation
    metrics.
    """

    VERSION = "1.0"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = str(self.config.get("mode", "calibrated")).lower()
        if self.mode not in {"calibrated", "fixed", "disabled"}:
            raise ValueError(
                "cohesion mode must be one of calibrated, fixed, disabled"
            )
        self.eta = float(self.config.get("eta", 0.1))
        self.weight = float(self.config.get("weight", 0.0))
        self.eps = float(self.config.get("eps", 1e-6))
        if self.eta < 0.0 or self.weight < 0.0:
            raise ValueError("cohesion eta and weight must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("cohesion eps must be positive")
        self._last_diagnostics: dict[str, Any] = {}

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64)
        N = len(a)

        if self.mode == "disabled" or N < 2:
            self._last_diagnostics = {
                "mode": self.mode,
                "delta_q": 0.0,
                "relevance_margin": 0.0,
                "pool_cohesion": 0.0,
                "K": int(context.get("selection_K", min(N, 1))),
            }
            return np.array(w, dtype=np.float64, copy=True)

        K = int(context.get("selection_K", min(N, 1)))
        K = min(max(K, 1), N)
        sorted_quality = np.sort(a)[::-1]
        if K < N:
            relevance_margin = max(float(sorted_quality[K - 1] - sorted_quality[K]), 0.0)
        else:
            relevance_margin = 0.0

        upper = b[np.triu_indices(N, k=1)]
        pool_cohesion = float(np.mean(np.clip(upper, 0.0, None))) if upper.size else 0.0

        if self.mode == "fixed":
            delta_q = self.weight
        else:
            denominator = max((K - 1) * (pool_cohesion + self.eps), self.eps)
            delta_q = self.eta * relevance_margin / denominator

        correction = delta_q * b
        self._last_diagnostics = {
            "mode": self.mode,
            "version": self.VERSION,
            "delta_q": float(delta_q),
            "eta": float(self.eta),
            "weight": float(self.weight),
            "relevance_margin": float(relevance_margin),
            "pool_cohesion": float(pool_cohesion),
            "K": int(K),
            "correction_norm": float(np.linalg.norm(correction)),
        }
        return np.asarray(w, dtype=np.float64) + correction

    @property
    def name(self) -> str:
        return "cohesion"

    @property
    def required_context_keys(self) -> tuple[str, ...]:
        return ("selection_K",)

    def description(self) -> str:
        if self.mode == "fixed":
            return f"Cohesion calibration (fixed={self.weight})"
        if self.mode == "disabled":
            return "Cohesion calibration (disabled)"
        return f"Cohesion calibration (eta={self.eta})"

    def diagnostic_summary(self) -> dict[str, Any]:
        """Return the last deterministic calibration statistics."""
        return dict(self._last_diagnostics)
