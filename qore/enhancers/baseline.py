"""Baseline enhancer: standard redundancy penalty.

This is the default QUBO formulation: w = gamma * b
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("baseline")
class BaselineEnhancer(QUBOEnhancer):
    """
    Baseline: w = gamma * b (standard redundancy penalty).

    This is the original QORE formulation where the interaction matrix is simply
    the redundancy matrix scaled by gamma. Higher gamma increases the penalty for
    selecting redundant passages.

    Config:
        gamma (float): Redundancy weight. Default 1.0.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.gamma = self.config.get("gamma", 1.0)

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Return gamma * b (ignores input w).

        Args:
            w: Ignored (baseline starts fresh).
            a: Ignored.
            b: (N, N) redundancy matrix.
            context: Ignored.

        Returns:
            w = gamma * b
        """
        return self.gamma * b

    @property
    def name(self) -> str:
        return "baseline"

    @property
    def composition_mode(self) -> str:
        return "replace"

    def description(self) -> str:
        return f"Baseline (γ={self.gamma})"
