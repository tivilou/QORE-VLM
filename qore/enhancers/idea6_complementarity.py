"""Idea 6: Complementarity enhancer.

w = gamma * b - delta * c

Where c_ij is the complementarity between passages i and j, measuring whether
they together provide more answer support than either alone.
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("idea6")
class ComplementarityEnhancer(QUBOEnhancer):
    """
    Idea 6: w = gamma * b - delta * c

    Adds complementarity reward using answer scorer. High complementarity means
    two passages together provide more answer support than either alone.

    Config:
        gamma (float): Redundancy weight. Default 1.0.
        delta (float): Complementarity weight. Default 0.0 (disabled).
        method (str): Complementarity computation method. Default "dpr".

    Context requirements:
        - question (str): Query text
        - passages (list[str]): Passage texts
        - answer_scorer: Instance with score_passages(question, texts) method
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.gamma = self.config.get("gamma", 1.0)
        self.delta = self.config.get("delta", 0.0)
        self.method = self.config.get("method", "dpr")

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Compute w = gamma * b - delta * c.

        Args:
            w: Ignored (Idea 6 computes w from scratch).
            a: Ignored.
            b: (N, N) redundancy matrix.
            context: Must contain "question", "passages", "answer_scorer".

        Returns:
            w = gamma * b - delta * c
        """
        if self.delta == 0.0:
            # No complementarity, just return gamma * b
            return self.gamma * b

        # Validate required context
        self.validate_context(context, ["question", "passages", "answer_scorer"])

        question = context["question"]
        passages = context["passages"]
        answer_scorer = context["answer_scorer"]

        # Compute complementarity matrix
        if self.method == "dpr":
            from applications.rag.signals_rag import passage_complementarity_dpr
            c = passage_complementarity_dpr(question, passages, answer_scorer)
        else:
            raise ValueError(f"Unknown complementarity method: {self.method}")

        # Combine: w = gamma * b - delta * c
        # Positive w_ij penalizes selecting i,j together (redundancy)
        # Negative w_ij rewards selecting i,j together (complementarity)
        return self.gamma * b - self.delta * c

    @property
    def name(self) -> str:
        return "idea6"

    @property
    def composition_mode(self) -> str:
        return "replace"

    @property
    def required_context_keys(self) -> tuple[str, ...]:
        if self.delta == 0.0:
            return ()
        return ("question", "passages", "answer_scorer")

    def description(self) -> str:
        return f"Idea 6 Complementarity (γ={self.gamma}, δ={self.delta})"
