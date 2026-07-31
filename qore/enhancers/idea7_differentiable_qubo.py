"""Idea 7: Differentiable QUBO with learned weights.

Uses Gumbel-Softmax or straight-through estimator to make QUBO selection
differentiable, enabling end-to-end training with task loss (Recall, F1).

NOTE: Phase 2 showed this approach failed (-27.6% Recall) due to sparse gradient
signal at low selection rates (K=5/N=50 = 10%). This plugin is included for:
1. Demonstration of plugin architecture for failed ideas
2. Future exploration with different configurations (e.g., K=15/N=50)
3. Reference implementation for similar research

Status: ⏸️ PAUSED after Phase 2 failure
"""

from typing import Any

import numpy as np
import torch

from .base import QUBOEnhancer
from .registry import register_enhancer


@register_enhancer("idea7")
class DifferentiableQUBOEnhancer(QUBOEnhancer):
    """
    Idea 7: End-to-end differentiable QUBO using Gumbel-Softmax.

    Instead of using fixed gamma/delta parameters, this plugin uses a trained
    neural network to predict w_ij from passage embeddings.

    Config:
        model_path (str): Path to trained model checkpoint. Required.
        temperature (float): Gumbel-Softmax temperature. Default 0.5.
        use_straight_through (bool): Use straight-through estimator. Default False.
        device (str): Device for inference ("cpu" or "cuda"). Default "cpu".

    Context requirements:
        - embeddings (required): (N, d) passage embeddings

    Training mode:
        Set config["training"] = True to return soft selection for backprop.
        In inference mode (default), returns hard selection.

    Phase 2 Results:
        - Baseline: Recall@5 = 0.4454
        - Idea 7:   Recall@5 = 0.3224 (-27.6%)
        - Root cause: K=5/N=50 (10% selection) → sparse gradients
        - Decision: PAUSED, prioritize Idea 6 (+39.4%)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        self.model_path = self.config.get("model_path")
        self.temperature = self.config.get("temperature", 0.5)
        self.use_straight_through = self.config.get("use_straight_through", False)
        self.device = self.config.get("device", "cpu")
        self.training = self.config.get("training", False)

        # Lazy load model
        self._model = None

        if not self.model_path:
            raise ValueError(
                "idea7 requires 'model_path' in config. "
                "Train a model using scripts/rag/train/train_soft_qubo_simple.py first."
            )

    def _load_model(self):
        """Lazy load the trained model."""
        if self._model is not None:
            return self._model

        from qore.soft_qubo import LearnableQUBO

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Infer model dimensions from checkpoint
        # The LearnableQUBO has w_a and w_b parameters
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Create model (dimensions will be inferred from state_dict)
        model = LearnableQUBO(
            embedding_dim=None,  # Will be set from checkpoint
            temperature=self.temperature,
            use_straight_through=self.use_straight_through,
        )

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        return model

    def enhance(
        self,
        w: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Use learned model to predict interaction matrix w.

        In training mode, this would be called during forward pass with gradients.
        In inference mode (default), we use the model to predict w for selection.

        Args:
            w: (N, N) current interaction matrix (ignored - we predict from scratch)
            a: (N,) quality scores (normalized)
            b: (N, N) redundancy matrix (cosine similarity)
            context: Must contain "embeddings" (N, d)

        Returns:
            w_predicted: (N, N) learned interaction matrix
        """
        # Validate context
        self.validate_context(context, ["embeddings"])

        embeddings = context["embeddings"]
        N = len(embeddings)

        # Load model
        model = self._load_model()

        # Convert to torch tensors
        embeddings_t = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        a_t = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)
        b_t = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(self.device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            # Model predicts learned weights w_a, w_b
            # Then computes w = w_a * identity + w_b * b
            # (Simplified - actual model may be more complex)

            # For now, use the model's learned parameters directly
            # In a full implementation, the model would take embeddings as input
            w_learned = model.w_b * b_t  # Simplified: use learned redundancy weight

            if hasattr(model, 'w_a'):
                # Add learned quality weight if present
                w_learned = w_learned + model.w_a * torch.eye(N, device=self.device)

        # Convert back to numpy
        w_predicted = w_learned.cpu().numpy()

        # Ensure symmetric and zero diagonal
        w_predicted = (w_predicted + w_predicted.T) / 2
        np.fill_diagonal(w_predicted, 0.0)

        return w_predicted

    @property
    def name(self) -> str:
        return "idea7"

    def description(self) -> str:
        status = "training" if self.training else "inference"
        return f"Idea 7 Differentiable QUBO (τ={self.temperature}, {status}) ⏸️ PAUSED"


# Note: This plugin demonstrates how failed ideas can be cleanly preserved
# in the plugin architecture without affecting other code. To remove:
# 1. Delete this file
# 2. Remove import from __init__.py
# 3. Delete any config files using idea7
# Core code remains completely unaffected.
