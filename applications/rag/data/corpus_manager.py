"""Unified corpus management for RAG evaluation.

Three interchangeable corpus modes solve the same problem — "give each query a
candidate passage pool that provably contains its gold evidence" — with
different cost/fidelity tradeoffs:

- precomputed: use a dataset's own retrieval results (cheapest; gold coverage
  depends on the dataset). Ideal when the benchmark ships DPR/BM25 top-k.
- aligned: build a controlled pool of (all gold passages) + random distractors.
  Guarantees gold coverage; corpus size is tunable (default ~40k). Recommended
  default for QORE experiments — every query is answerable, so EM/F1/Recall@K
  are meaningful without a full 21M-passage download.
- faiss: index the full corpus and retrieve live. Most realistic; highest cost
  (disk + build time). Gold coverage is probabilistic (whatever retrieval finds).

All three expose the same interface (build / retrieve / gold mapping), so the
eval loop is corpus-mode agnostic. See docs/rag_corpus_modes.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Corpus:
    """A candidate passage pool with embeddings and gold alignment.

    Attributes:
        passages: list of passage texts, length N.
        embeddings: (N, d) passage embedding matrix, row i embeds passages[i].
        gold_mapping: question_id -> set of corpus indices that are gold for it.
            Empty set means "no gold recorded" (e.g. precomputed mode without
            alignment); Recall@K is then undefined and reported as None.
        metadata: free-form provenance (mode, sizes, source ids) for the results
            file so a run is reproducible.
    """

    passages: list[str]
    embeddings: np.ndarray
    gold_mapping: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.passages)

    def gold_for(self, question_id) -> set:
        """Gold corpus indices for a question, or empty set if none recorded."""
        return self.gold_mapping.get(question_id, set())


class CorpusManager(ABC):
    """Abstract corpus backend: build once, then retrieve per query.

    Subclasses implement one mode. The eval loop only sees this interface, so
    switching modes is a config change, not a code change.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._corpus: Optional[Corpus] = None

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            raise RuntimeError("build() must be called before accessing corpus.")
        return self._corpus

    @abstractmethod
    def build(self, questions: list[dict]) -> Corpus:
        """Construct (or load) the corpus for these questions.

        Args:
            questions: list of dicts, each with at least 'id' and 'question';
                'gold_passages' / 'context' used by modes that align gold.

        Returns:
            The built Corpus (also cached on self._corpus).
        """
        raise NotImplementedError

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) of the top_k passages for a query.

        Default: dense inner-product over the full embedding matrix. DPR uses
        inner product (not cosine), matching wiki_dpr's space. FAISS mode
        overrides this with an index lookup.
        """
        emb = self.corpus.embeddings
        scores = emb @ np.asarray(query_embedding, dtype=emb.dtype)
        k = min(top_k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return top, scores[top]

    def gold_mapping(self) -> dict:
        return self.corpus.gold_mapping


def make_corpus_manager(mode: str, config: Optional[dict] = None) -> CorpusManager:
    """Factory: resolve a mode string to a concrete CorpusManager.

    Lazy imports keep optional heavy deps (faiss) out of the import path unless
    that mode is actually requested.
    """
    mode = (mode or "aligned").lower()
    if mode == "aligned":
        from .aligned import AlignedCorpusManager
        return AlignedCorpusManager(config)
    if mode == "precomputed":
        from .precomputed import PrecomputedCorpusManager
        return PrecomputedCorpusManager(config)
    if mode == "faiss":
        from .faiss_corpus import FaissCorpusManager
        return FaissCorpusManager(config)
    raise ValueError(
        f"Unknown corpus mode '{mode}'. Choose 'aligned', 'precomputed', or 'faiss'."
    )
