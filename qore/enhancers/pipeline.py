"""EnhancerPipeline: chain multiple enhancers in sequence."""

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

    def __init__(self, enhancers: list[QUBOEnhancer]):
        """
        Initialize the pipeline with a list of enhancers.

        Args:
            enhancers: List of QUBOEnhancer instances to apply in order.
        """
        if not enhancers:
            raise ValueError("Pipeline requires at least one enhancer")
        self.enhancers = enhancers

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
        N = len(a)
        w = np.zeros((N, N), dtype=np.float64)  # Start with zero matrix

        for enhancer in self.enhancers:
            w = enhancer.enhance(w, a, b, context)

        return w

    def describe(self) -> str:
        """
        Human-readable pipeline description.

        Returns:
            String showing the sequence of enhancers (e.g., "baseline → idea6").
        """
        descriptions = [e.description() for e in self.enhancers]
        return " → ".join(descriptions)

    def names(self) -> list[str]:
        """
        Get the names of all enhancers in this pipeline.

        Returns:
            List of enhancer names in order.
        """
        return [e.name for e in self.enhancers]
