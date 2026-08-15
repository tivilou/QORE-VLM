"""Answer likelihood scorer for RAG passage selection.

Computes how likely each retrieved passage contains an answer to the query.
This signal replaces (or augments) DPR retrieval scores as the quality term
a_i in the QUBO objective, directly aligning QORE's optimization target with
the downstream QA goal.

Two backends are available:
- DPRAnswerScorer: uses facebook/dpr-reader-single-nq-base relevance_logits.
  Each logit represents "does this passage contain the answer?" — the most
  semantically correct signal for RAG selection.
- CrossEncoderAnswerScorer: uses a MiniLM cross-encoder as a lightweight
  alternative when DPR reader is too slow or unavailable.

Usage:
    scorer = DPRAnswerScorer()                    # load once
    scores = scorer.score_passages(question, passages)  # (N,) float in [0,1]
    selected = select_passages(..., relevance_scores=scores)
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


class DPRAnswerScorer:
    """Answer likelihood scorer based on DPR reader relevance logits.

    Uses facebook/dpr-reader-single-nq-base, which outputs a per-passage
    logit representing whether the passage contains the answer. This is
    more directly aligned with the RAG goal than the retriever inner product.

    Args:
        model_name: HuggingFace model path for DPR reader.
        device: 'cuda', 'cpu', or None (auto-detect).
        batch_size: Number of passages scored per forward pass.
    """

    def __init__(
        self,
        model_name: str = "facebook/dpr-reader-single-nq-base",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        import torch
        from transformers import DPRReader, DPRReaderTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        print(f"  [AnswerScorer] Loading DPR reader: {model_name} on {self.device}")
        self.reader = DPRReader.from_pretrained(model_name).to(self.device)
        self.tokenizer = DPRReaderTokenizer.from_pretrained(model_name)
        self.reader.eval()

    def score_passages(self, question: str, passages: list[str]) -> np.ndarray:
        """Score each passage by answer likelihood.

        Args:
            question: The query string.
            passages: List of N passage texts.

        Returns:
            scores: (N,) float32 array in [0, 1]; higher = more likely contains answer.
        """
        import torch

        all_scores = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = self.tokenizer(
                questions=[question] * len(batch),
                texts=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=350,  # DPR reader max
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.reader(**encoded)
                # relevance_logits: (batch_size,) — passage-level answer presence
                logits = outputs.relevance_logits.cpu().float().numpy()
                # Sigmoid to get probabilities in [0, 1]
                batch_scores = 1.0 / (1.0 + np.exp(-logits))
            all_scores.append(batch_scores)

        return np.concatenate(all_scores, axis=0)

    def score_passages_with_hypotheses(
        self,
        question: str,
        passages: list[str],
        *,
        top_m: int = 3,
        max_answer_tokens: int = 10,
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        """Return baseline scores plus diagnostic-only top answer spans.

        This method does not modify the selector. It exposes start/end logits
        already produced by the DPR reader so Phase 7A can measure answer
        agreement before implementing an answer-conditioned objective.
        """
        import torch

        from applications.rag.answer_evidence import extract_top_answer_spans

        if not passages:
            return np.empty(0, dtype=np.float32), []

        all_scores: list[np.ndarray] = []
        all_hypotheses: list[list[dict[str, Any]]] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            encoded = self.tokenizer(
                questions=[question] * len(batch),
                texts=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=350,
            )
            encoded_cpu = {
                key: value.detach().cpu()
                for key, value in encoded.items()
            }
            encoded_device = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.reader(**encoded_device)
                logits = outputs.relevance_logits.detach().cpu().float().numpy()
                batch_scores = 1.0 / (1.0 + np.exp(-logits))
            batch_hypotheses = extract_top_answer_spans(
                encoded_cpu["input_ids"].numpy(),
                outputs.start_logits.detach().cpu().float().numpy(),
                outputs.end_logits.detach().cpu().float().numpy(),
                self.tokenizer,
                attention_mask=encoded_cpu.get("attention_mask").numpy()
                if encoded_cpu.get("attention_mask") is not None
                else None,
                top_m=top_m,
                max_answer_tokens=max_answer_tokens,
            )
            all_scores.append(batch_scores)
            all_hypotheses.extend(batch_hypotheses)

        return np.concatenate(all_scores, axis=0), all_hypotheses


class CrossEncoderAnswerScorer:
    """Lightweight answer scorer using a cross-encoder reranker.

    Uses sentence-transformers CrossEncoder. Faster than DPR reader but
    outputs relevance score rather than true answer likelihood.
    Suitable when latency is critical or DPR reader is unavailable.

    Args:
        model_name: CrossEncoder model from sentence-transformers.
        device: 'cuda', 'cpu', or None (auto-detect).
        batch_size: Passages per forward pass.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import CrossEncoder

        self.batch_size = batch_size
        print(f"  [AnswerScorer] Loading CrossEncoder: {model_name}")
        self.model = CrossEncoder(model_name, device=device or "cuda")

    def score_passages(self, question: str, passages: list[str]) -> np.ndarray:
        """Score each passage by question-passage relevance.

        Args:
            question: The query string.
            passages: List of N passage texts.

        Returns:
            scores: (N,) float32 array (logit scale); higher = more relevant.
        """
        pairs = [[question, p] for p in passages]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        # Normalize to [0, 1] via sigmoid for compatibility with QORE
        scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
        return scores


def make_answer_scorer(backend: str = "dpr", **kwargs):
    """Factory: create an answer scorer by backend name.

    Args:
        backend: 'dpr' (DPRAnswerScorer) or 'cross_encoder' (CrossEncoderAnswerScorer).
        **kwargs: Forwarded to the scorer constructor.
    """
    backend = backend.lower()
    if backend == "dpr":
        return DPRAnswerScorer(**kwargs)
    if backend in ("cross_encoder", "ce"):
        return CrossEncoderAnswerScorer(**kwargs)
    raise ValueError(f"Unknown answer scorer backend '{backend}'. Choose 'dpr' or 'cross_encoder'.")
