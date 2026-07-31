"""Idea 4: Context integrity enhancer (placeholder).

Rewards selecting passages that maintain contextual coherence (e.g., consecutive
passages from the same document).

NOTE: This is a placeholder implementation. The actual logic needs to be
implemented based on Idea 4's full specification.
"""

from typing import Any

import numpy as np

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("idea4")
class ContextIntegrityEnhancer(QUBOEnhancer):
    """
    Idea 4: Add context integrity bonus.

    Rewards selecting passages that maintain contextual coherence. For example,
    consecutive passages from the same document get a negative w_ij (reward).

    Config:
        alpha (float): Integrity bonus weight. Default 0.1.

    Context requirements:
        - passages_meta (list[dict]): Passage metadata with "doc_id" and "rank" fields.

    NOTE: This is a placeholder. The actual implementation needs to be completed
    when Idea 4 is fully specified.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.alpha = self.config.get("alpha", 0.1)

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Add context integrity bonus to w.

        Args:
            w: (N, N) current interaction matrix.
            a: Ignored.
            b: Ignored.
            context: Must contain "passages_meta".

        Returns:
            w_new = w + alpha * integrity_matrix
        """
        passages_meta = context.get("passages_meta")
        if passages_meta is None:
            # No metadata available, return w unchanged
            return w

        N = len(w)
        integrity = np.zeros((N, N), dtype=np.float64)

        # Compute integrity bonus matrix
        for i in range(N):
            for j in range(i + 1, N):
                # Check if passages i, j are consecutive in the same document
                if self._are_consecutive(passages_meta[i], passages_meta[j]):
                    integrity[i, j] = -1.0  # Negative = reward
                    integrity[j, i] = -1.0

        return w + self.alpha * integrity

    def _are_consecutive(self, meta_i: dict, meta_j: dict) -> bool:
        """
        Check if two passages are consecutive in the same document.

        Args:
            meta_i: Metadata for passage i (must have "doc_id" and "rank").
            meta_j: Metadata for passage j (must have "doc_id" and "rank").

        Returns:
            True if passages are from the same doc and have adjacent ranks.
        """
        if not all(k in meta_i for k in ["doc_id", "rank"]):
            return False
        if not all(k in meta_j for k in ["doc_id", "rank"]):
            return False

        same_doc = meta_i["doc_id"] == meta_j["doc_id"]
        consecutive = abs(meta_i["rank"] - meta_j["rank"]) == 1

        return same_doc and consecutive

    @property
    def name(self) -> str:
        return "idea4"

    def description(self) -> str:
        return f"Idea 4 Context Integrity (α={self.alpha})"
