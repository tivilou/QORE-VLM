"""DPR encoder wrapper for query and passage encoding.

Provides a unified interface to encode text into embeddings using DPR or
sentence transformers. Used by corpus builders (aligned mode) and the eval loop
(query encoding).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class DPREncoder:
    """Dual encoder wrapper (query encoder + passage encoder for DPR).

    DPR uses separate models for queries and passages. If you want symmetric
    encoding (e.g. sentence-transformers all-mpnet), use SentenceEncoder instead.
    """

    def __init__(
        self,
        query_encoder: str = "facebook/dpr-question_encoder-single-nq-base",
        passage_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
        device: Optional[str] = None,
    ):
        from transformers import (
            DPRQuestionEncoder, DPRContextEncoder,
            DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
        )

        self.query_model = DPRQuestionEncoder.from_pretrained(query_encoder)
        self.passage_model = DPRContextEncoder.from_pretrained(passage_encoder)
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_encoder)
        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_encoder)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_model.to(self.device).eval()
        self.passage_model.to(self.device).eval()

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode queries into (N, d) embeddings."""
        return self._encode_batch(
            queries, self.query_model, self.query_tokenizer, batch_size
        )

    def encode_passages(self, passages: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode passages into (N, d) embeddings."""
        return self._encode_batch(
            passages, self.passage_model, self.passage_tokenizer, batch_size
        )

    def _encode_batch(self, texts, model, tokenizer, batch_size):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embs = model(**inputs).pooler_output.cpu().numpy()
                all_embs.append(embs)
        return np.vstack(all_embs) if all_embs else np.empty((0, 768), np.float32)


class SentenceEncoder:
    """Symmetric encoder via sentence-transformers (same model for query & passage).

    Simpler than DPR for tasks where query/passage distinction isn't needed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into (N, d) embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # Alias for compatibility
    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(queries, batch_size)

    def encode_passages(self, passages: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(passages, batch_size)


def make_encoder(encoder_type: str = "dpr", **kwargs):
    """Factory: create DPREncoder or SentenceEncoder by name."""
    if encoder_type == "dpr":
        return DPREncoder(**kwargs)
    if encoder_type == "sentence":
        return SentenceEncoder(**kwargs)
    raise ValueError(f"Unknown encoder_type: {encoder_type} (dpr|sentence)")
