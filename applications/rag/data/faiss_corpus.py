"""FAISS corpus mode: full-corpus retrieval over the complete embedding set.

This is the most realistic (and most expensive) mode: index the FULL DPR corpus
(21M passages) with FAISS and retrieve per query, exactly like a production RAG
pipeline. Gold coverage is probabilistic — a question's gold passage is only in
the candidate set if retrieval surfaces it, so Recall@retrieval is itself a
measured quantity, not a guarantee (unlike aligned mode).

Requires `faiss` (faiss-cpu or faiss-gpu). Building the index over 21M x 768
float32 is heavy (~65GB); intended for machines provisioned for it, not the
24GB-GPU smoke path. The aligned mode is the tractable default.
"""

from __future__ import annotations

import numpy as np

from .corpus_manager import Corpus, CorpusManager


class FaissCorpusManager(CorpusManager):
    """Full-corpus retrieval via a FAISS index.

    config keys:
        embeddings: (N, d) float32 corpus embeddings (or a memmap).
        passages: list[str] corpus texts (or a lazy accessor).
        gold_resolver: callable(question) -> set[int] mapping a question to the
            global corpus indices of its gold passages (for Recall scoring).
        index_type: "flat" (exact, default) or "ivf" (approximate, faster).
        nlist / nprobe: IVF params (ignored for flat).
    """

    def build(self, questions: list[dict]) -> Corpus:
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "FAISS mode needs faiss-cpu or faiss-gpu: pip install faiss-cpu"
            ) from e

        raw = self.config["embeddings"]
        # Preserve memmap: if already float32, asarray returns a view (no copy).
        # Forcing dtype=float32 only copies when the source dtype differs.
        if isinstance(raw, np.ndarray) and raw.dtype == np.float32:
            embeddings = raw  # memmap or plain ndarray — no extra copy
        else:
            embeddings = np.asarray(raw, dtype=np.float32)
        passages = self.config["passages"]
        n, d = embeddings.shape

        index_type = self.config.get("index_type", "flat")
        if index_type == "flat":
            index = faiss.IndexFlatIP(d)  # inner product (DPR convention)
            index.add(embeddings)
        elif index_type == "ivf":
            nlist = self.config.get("nlist", 4096)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = self.config.get("nprobe", 64)
        else:
            raise ValueError(f"Unknown index_type {index_type!r} (flat|ivf)")

        self._index = index

        # Resolve gold to GLOBAL corpus indices for Recall scoring.
        gold_mapping = {}
        resolver = self.config.get("gold_resolver")
        if resolver is not None:
            for q in questions:
                gold_mapping[q.get("id")] = set(resolver(q))

        metadata = {
            "mode": "faiss",
            "corpus_size": n,
            "embed_dim": d,
            "index_type": index_type,
            "n_questions_with_gold": sum(1 for v in gold_mapping.values() if v),
        }
        self._corpus = Corpus(passages, embeddings, gold_mapping, metadata)
        return self._corpus

    def retrieve(self, query_embedding, top_k):
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, idx = self._index.search(q, top_k)
        return idx[0], scores[0]
