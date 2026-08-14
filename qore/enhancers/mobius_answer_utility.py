"""Second-order answer-utility Mobius correction.

The plugin estimates the second-order set-function coefficient

    m_ij = u({i, j}) - u({i}) - u({j})

from the configured answer scorer.  Positive pair surplus is a useful joint
answer signal, so it enters the minimization objective as ``-beta * m_ij``.
It is additive by design and therefore composes with the stable baseline root.
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("mobius")
class MobiusAnswerUtilityEnhancer(QUBOEnhancer):
    """Add a pairwise answer-utility Mobius term to the current objective.

    Config keys:
        mode: ``"pairwise"`` (default) or ``"disabled"``.
        beta: Non-negative strength of the Mobius interaction. Default 0.0.
        clip: Optional non-negative absolute surplus clip. Default 1.0.

    The scorer is expected to return bounded answer utilities (the built-in
    scorers return sigmoid values).  A zero empty-set utility is used, so this
    is the literal second-order coefficient of the bounded scorer utility.
    """

    VERSION = "1.0"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.mode = str(self.config.get("mode", "pairwise")).lower()
        if self.mode not in {"pairwise", "disabled"}:
            raise ValueError("mobius mode must be one of pairwise, disabled")
        self.beta = float(self.config.get("beta", 0.0))
        self.clip = float(self.config.get("clip", 1.0))
        if self.beta < 0.0:
            raise ValueError("mobius beta must be non-negative")
        if self.clip < 0.0:
            raise ValueError("mobius clip must be non-negative")
        self._last_diagnostics: dict[str, Any] = {}

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        del a, b
        w = np.asarray(w, dtype=np.float64)
        if self.mode == "disabled" or self.beta == 0.0:
            self._last_diagnostics = {
                "mode": self.mode,
                "version": self.VERSION,
                "beta": float(self.beta),
                "n_pairs": 0,
                "correction_norm": 0.0,
            }
            return np.array(w, dtype=np.float64, copy=True)

        self.validate_context(context, ["question", "passages", "answer_scorer"])
        question = context["question"]
        passages = list(context["passages"])
        scorer = context["answer_scorer"]
        n = len(passages)
        if w.shape != (n, n):
            raise ValueError(
                f"mobius received {n} passages but interaction matrix has shape {w.shape}"
            )
        if n < 2:
            self._last_diagnostics = {
                "mode": self.mode,
                "version": self.VERSION,
                "beta": float(self.beta),
                "n_pairs": 0,
                "correction_norm": 0.0,
            }
            return np.array(w, dtype=np.float64, copy=True)

        singles = np.asarray(scorer.score_passages(question, passages), dtype=np.float64)
        if singles.shape != (n,) or not np.all(np.isfinite(singles)):
            raise ValueError("answer_scorer returned invalid single-passage utilities")

        pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_texts = [passages[i] + " " + passages[j] for i, j in pair_indices]
        pair_scores = np.asarray(
            scorer.score_passages(question, pair_texts), dtype=np.float64
        )
        if pair_scores.shape != (len(pair_indices),) or not np.all(np.isfinite(pair_scores)):
            raise ValueError("answer_scorer returned invalid pair utilities")

        surplus = pair_scores - np.asarray(
            [singles[i] + singles[j] for i, j in pair_indices], dtype=np.float64
        )
        if self.clip > 0.0:
            surplus = np.clip(surplus, -self.clip, self.clip)

        correction = np.zeros((n, n), dtype=np.float64)
        for (i, j), value in zip(pair_indices, surplus):
            correction[i, j] = correction[j, i] = -self.beta * float(value)

        self._last_diagnostics = {
            "mode": self.mode,
            "version": self.VERSION,
            "beta": float(self.beta),
            "clip": float(self.clip),
            "n_pairs": len(pair_indices),
            "single_min": float(np.min(singles)),
            "single_max": float(np.max(singles)),
            "pair_min": float(np.min(pair_scores)),
            "pair_max": float(np.max(pair_scores)),
            "surplus_mean": float(np.mean(surplus)),
            "surplus_min": float(np.min(surplus)),
            "surplus_max": float(np.max(surplus)),
            "correction_norm": float(np.linalg.norm(correction)),
        }
        return w + correction

    @property
    def name(self) -> str:
        return "mobius"

    @property
    def required_context_keys(self) -> tuple[str, ...]:
        if self.mode == "disabled" or self.beta == 0.0:
            return ()
        return ("question", "passages", "answer_scorer")

    def description(self) -> str:
        if self.mode == "disabled" or self.beta == 0.0:
            return "Mobius answer utility (disabled)"
        return f"Mobius answer utility (beta={self.beta})"

    def diagnostic_summary(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)
