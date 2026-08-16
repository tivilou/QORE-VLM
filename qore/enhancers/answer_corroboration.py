"""Answer-identity corroboration enhancer for the Phase 7C ablation.

This plugin is deliberately narrower than the rejected signed conflict
objective.  It adds a reward for answer agreement or independent corroboration
to an existing baseline interaction matrix.  The feature matrix is produced
once by the reader and passed through selector context; the plugin never
changes retrieval, labels, or evaluation.
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("answer_corroboration")
class AnswerCorroborationEnhancer(QUBOEnhancer):
    """Add a normalized answer-agreement/corroboration reward.

    Config keys:
        mode: ``agreement`` or ``corroboration``. Default ``corroboration``.
        strength: non-negative fraction of baseline pair RMS. Default ``0.25``.
        normalization: ``baseline_rms`` (default) or ``none``.
    """

    VERSION = "0.1.0"
    MODES = {"agreement", "corroboration"}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = str(self.config.get("mode", "corroboration")).lower()
        if self.mode not in self.MODES:
            raise ValueError("answer_corroboration mode must be agreement or corroboration")
        self.strength = float(self.config.get("strength", 0.25))
        if self.strength < 0.0:
            raise ValueError("answer_corroboration strength must be non-negative")
        self.normalization = str(
            self.config.get("normalization", "baseline_rms")
        ).lower()
        if self.normalization not in {"baseline_rms", "none"}:
            raise ValueError(
                "answer_corroboration normalization must be baseline_rms or none"
            )
        self._last_diagnostics: dict[str, Any] = {}

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        del a
        w = np.asarray(w, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != w.shape[1] or b.shape != w.shape:
            raise ValueError("answer_corroboration requires square w and matching b")
        if self.strength == 0.0:
            self._last_diagnostics = self._diagnostics(0.0, 0.0, 0.0)
            return np.array(w, dtype=np.float64, copy=True)

        self.validate_context(context, ["answer_evidence_matrices"])
        matrices = context["answer_evidence_matrices"]
        if not isinstance(matrices, dict) or self.mode not in matrices:
            raise ValueError(
                f"answer_evidence_matrices must contain '{self.mode}'"
            )
        feature = np.asarray(matrices[self.mode], dtype=np.float64)
        self._validate_feature(feature, w.shape[0])

        feature_rms = self._pair_rms(feature)
        baseline_rms = self._pair_rms(b)
        if feature_rms == 0.0 or self.normalization == "none":
            coefficient = self.strength
        else:
            coefficient = self.strength * baseline_rms / feature_rms
        correction = -coefficient * feature
        output = w + correction
        np.fill_diagonal(output, 0.0)
        self._last_diagnostics = self._diagnostics(
            coefficient, baseline_rms, feature_rms
        )
        return output

    @staticmethod
    def _pair_rms(matrix: np.ndarray) -> float:
        values = matrix[np.triu_indices(matrix.shape[0], k=1)]
        return float(np.sqrt(np.mean(values * values))) if values.size else 0.0

    @staticmethod
    def _validate_feature(feature: np.ndarray, size: int) -> None:
        if feature.shape != (size, size):
            raise ValueError(
                f"answer evidence feature has shape {feature.shape}; expected {(size, size)}"
            )
        if not np.all(np.isfinite(feature)):
            raise ValueError("answer evidence feature contains non-finite values")
        if not np.allclose(feature, feature.T, atol=1e-10, rtol=1e-10):
            raise ValueError("answer evidence feature must be symmetric")
        if not np.allclose(np.diag(feature), 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("answer evidence feature must have zero diagonal")
        if np.min(feature) < -1e-12 or np.max(feature) > 1.0 + 1e-12:
            raise ValueError("answer evidence feature must be bounded in [0, 1]")

    def _diagnostics(
        self, coefficient: float, baseline_rms: float, feature_rms: float
    ) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "mode": self.mode,
            "strength": self.strength,
            "normalization": self.normalization,
            "coefficient": float(coefficient),
            "baseline_pair_rms": float(baseline_rms),
            "feature_pair_rms": float(feature_rms),
        }

    @property
    def name(self) -> str:
        return "answer_corroboration"

    @property
    def required_context_keys(self) -> tuple[str, ...]:
        return () if self.strength == 0.0 else ("answer_evidence_matrices",)

    def description(self) -> str:
        return (
            f"Answer {self.mode} reward "
            f"(strength={self.strength}, normalization={self.normalization})"
        )

    def diagnostic_summary(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)
