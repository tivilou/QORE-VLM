"""Base class for QUBO enhancers.

Each idea (Idea 6, Idea 7, etc.) implements this interface to inject its logic
into the QUBO construction. Multiple enhancers can be chained to combine ideas.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class QUBOEnhancer(ABC):
    """
    QUBO enhancer interface: modify the interaction matrix w.

    Each idea implements this interface to inject its logic. Multiple enhancers
    can be chained using EnhancerPipeline (composition pattern).

    The enhancer receives:
    - w: Current interaction matrix (output from previous enhancer, or zeros)
    - a: Quality scores (normalized)
    - b: Redundancy matrix (cosine similarity between passages)
    - context: Additional data (embeddings, texts, question, answer_scorer, etc.)

    And returns:
    - w_new: Modified interaction matrix

    The final w is used to build QUBO: Q_ij = w_ij + 2*lam (off-diagonal)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the enhancer with configuration.

        Args:
            config: Enhancer-specific configuration dict (e.g., gamma, delta, alpha).
                   Each enhancer defines its own config schema.
        """
        self.config = config or {}

    @abstractmethod
    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Modify the interaction matrix w.

        Args:
            w: (N, N) current interaction matrix from previous enhancers.
               Initial w is zeros. Positive values penalize selecting pairs
               (redundancy), negative values reward selecting pairs (complementarity).
            a: (N,) quality scores, min-max normalized to [0, 1].
            b: (N, N) redundancy matrix (cosine similarity), symmetric with zero diagonal.
            context: Additional context dict with keys:
                - "embeddings": (N, d) passage embeddings
                - "query_embedding": (d,) query embedding
                - "passages": list[str] of N passage texts (optional)
                - "question": str query text (optional)
                - "answer_scorer": answer scorer instance (optional)
                - "passages_meta": list[dict] of passage metadata (optional)

        Returns:
            w_new: (N, N) modified interaction matrix. Must be symmetric with zero diagonal.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this enhancer (e.g., "baseline", "idea6").

        Used for registry lookup and config keys.
        """
        pass

    def description(self) -> str:
        """
        Human-readable description of this enhancer (for logging and display).

        Default implementation returns the name. Override to include config info.
        """
        return self.name

    @property
    def composition_mode(self) -> str:
        """How this enhancer composes with the current interaction matrix.

        ``"add"`` enhancers preserve the current objective and add a term.
        ``"replace"`` enhancers define a complete root objective.  The default
        remains additive so existing third-party enhancers keep working.
        """
        return "add"

    @property
    def required_context_keys(self) -> tuple[str, ...]:
        """Context fields required before this enhancer can run."""
        return ()

    def validate_context(self, context: dict[str, Any], required_keys: list[str]) -> None:
        """
        Helper to validate required context keys.

        Args:
            context: Context dict to validate.
            required_keys: List of keys that must be present and non-None.

        Raises:
            ValueError: If any required key is missing or None.
        """
        missing = [key for key in required_keys if context.get(key) is None]
        if missing:
            raise ValueError(
                f"{self.name} requires context keys: {required_keys}. "
                f"Missing: {missing}"
            )
