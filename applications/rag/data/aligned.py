"""Aligned corpus mode: gold passages + random distractors.

Guarantees every question's gold evidence is in the candidate pool, so
end-to-end EM/F1 and Recall@K are meaningful without downloading the full
~80GB / 21M-passage wiki_dpr corpus. Corpus size is tunable via
`n_distractors`; default ~36k distractors + all gold ≈ 40k passages.

Build is cached to disk (passages + embeddings + gold mapping); a second run
with the same output_dir loads instead of rebuilding.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .corpus_manager import Corpus, CorpusManager


class AlignedCorpusManager(CorpusManager):
    """Build/load a gold-aligned corpus.

    config keys:
        output_dir: where to cache the built corpus (required for caching).
        n_distractors: number of random distractor passages (default 36000).
        embed_dim: passage embedding dim (default 768, DPR).
        wiki_dpr_config: HF config name for the corpus source
            (default 'psgs_w100.nq.exact').
        seed: RNG seed for distractor sampling (default 42).
        rebuild: if True, ignore any cache and rebuild.
    """

    DEFAULT_N_DISTRACTORS = 36000
    DEFAULT_EMBED_DIM = 768

    def build(self, questions: list[dict]) -> Corpus:
        out_dir = self.config.get("output_dir")
        rebuild = self.config.get("rebuild", False)

        if out_dir and not rebuild and self._cache_exists(out_dir):
            self._corpus = self._load_cache(out_dir)
            return self._corpus

        self._corpus = self._build_fresh(questions)
        if out_dir:
            self._save_cache(out_dir, self._corpus)
        return self._corpus

    # ------------------------------------------------------------------ build

    def _build_fresh(self, questions: list[dict]) -> Corpus:
        gold_passages, gold_embeddings, gold_mapping = self._collect_gold(questions)
        n_distractors = int(self.config.get("n_distractors", self.DEFAULT_N_DISTRACTORS))
        distractor_passages, distractor_embeddings = self._sample_distractors(
            n_distractors, exclude_texts=set(gold_passages)
        )

        passages = list(gold_passages) + list(distractor_passages)
        if len(gold_embeddings) and len(distractor_embeddings):
            embeddings = np.vstack([gold_embeddings, distractor_embeddings])
        elif len(gold_embeddings):
            embeddings = gold_embeddings
        else:
            embeddings = distractor_embeddings

        metadata = {
            "mode": "aligned",
            "n_gold": len(gold_passages),
            "n_distractors": len(distractor_passages),
            "n_total": len(passages),
            "n_questions": len(questions),
            "n_questions_with_gold": sum(1 for v in gold_mapping.values() if v),
            "embed_dim": int(embeddings.shape[1]) if embeddings.size else None,
        }
        return Corpus(passages, embeddings, gold_mapping, metadata)

    def _collect_gold(self, questions: list[dict]):
        """Extract gold passages, dedup, and map question_id -> corpus indices.

        A gold passage may be shared by multiple questions; we store it once and
        point every owning question at that single index. Each question's
        'gold_passages' is expected to be a list of {'text', 'embedding'} dicts
        (embedding optional — if absent it is encoded on the fly by the caller's
        embedder, provided via config['embedder']).
        """
        embedder = self.config.get("embedder")  # callable(list[str]) -> (n,d) or None
        text_to_idx: dict[str, int] = {}
        gold_passages: list[str] = []
        gold_embeddings: list[np.ndarray] = []
        gold_mapping: dict = {}

        pending_texts: list[str] = []  # texts needing on-the-fly embedding

        for q in questions:
            qid = q.get("id")
            gold_items = q.get("gold_passages") or []
            idx_set = set()
            for item in gold_items:
                text = item["text"] if isinstance(item, dict) else item
                if text not in text_to_idx:
                    idx = len(gold_passages)
                    text_to_idx[text] = idx
                    gold_passages.append(text)
                    emb = item.get("embedding") if isinstance(item, dict) else None
                    if emb is not None:
                        gold_embeddings.append(np.asarray(emb, dtype=np.float32))
                    else:
                        gold_embeddings.append(None)  # placeholder, fill below
                        pending_texts.append(text)
                idx_set.add(text_to_idx[text])
            gold_mapping[qid] = idx_set

        # Fill missing embeddings in one batched encode call if an embedder given.
        if pending_texts:
            if embedder is None:
                raise ValueError(
                    "Some gold passages have no embedding and no config['embedder'] "
                    "was provided to encode them."
                )
            encoded = np.asarray(embedder(pending_texts), dtype=np.float32)
            it = iter(encoded)
            gold_embeddings = [
                next(it) if e is None else e for e in gold_embeddings
            ]

        emb_arr = (
            np.vstack([e.reshape(1, -1) for e in gold_embeddings])
            if gold_embeddings else np.empty((0, self.DEFAULT_EMBED_DIM), np.float32)
        )
        return gold_passages, emb_arr, gold_mapping

    def _sample_distractors(self, n: int, exclude_texts: set):
        """Stream wiki_dpr and randomly keep n passages not in exclude_texts.

        Reservoir-free approach: we shuffle stream indices deterministically by
        pulling more than n and subsampling. To avoid loading 21M rows we stream
        a bounded window (n * oversample) and sample within it — good enough for
        distractors, which only need to be plausible negatives.
        """
        if n <= 0:
            return [], np.empty((0, self.DEFAULT_EMBED_DIM), np.float32)

        from datasets import load_dataset

        cfg = self.config.get("wiki_dpr_config", "psgs_w100.nq.exact")
        seed = int(self.config.get("seed", 42))
        oversample = float(self.config.get("distractor_oversample", 3.0))
        window = int(n * oversample)

        stream = load_dataset(
            "facebook/wiki_dpr", cfg, split="train",
            streaming=True, trust_remote_code=True,
        )
        texts, embs = [], []
        for item in stream:
            if len(texts) >= window:
                break
            if item["text"] in exclude_texts:
                continue
            texts.append(item["text"])
            embs.append(item["embeddings"])

        rng = np.random.default_rng(seed)
        if len(texts) > n:
            keep = rng.choice(len(texts), size=n, replace=False)
            keep.sort()
            texts = [texts[i] for i in keep]
            embs = [embs[i] for i in keep]

        return texts, np.asarray(embs, dtype=np.float32)

    # ------------------------------------------------------------------ cache

    @staticmethod
    def _paths(out_dir: str):
        d = Path(out_dir)
        return (
            d / "corpus_passages.pkl",
            d / "corpus_embeddings.npy",
            d / "gold_mapping.json",
            d / "corpus_meta.json",
        )

    def _cache_exists(self, out_dir: str) -> bool:
        return all(p.exists() for p in self._paths(out_dir))

    def _save_cache(self, out_dir: str, corpus: Corpus):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        p_txt, p_emb, p_gold, p_meta = self._paths(out_dir)
        with open(p_txt, "wb") as f:
            pickle.dump(corpus.passages, f)
        np.save(p_emb, corpus.embeddings)
        # JSON needs list, not set; keep question_id as string key.
        with open(p_gold, "w") as f:
            json.dump({str(k): sorted(v) for k, v in corpus.gold_mapping.items()}, f)
        with open(p_meta, "w") as f:
            json.dump(corpus.metadata, f, indent=2)

    def _load_cache(self, out_dir: str) -> Corpus:
        p_txt, p_emb, p_gold, p_meta = self._paths(out_dir)
        with open(p_txt, "rb") as f:
            passages = pickle.load(f)
        embeddings = np.load(p_emb)
        with open(p_gold) as f:
            raw = json.load(f)
        gold_mapping = {k: set(v) for k, v in raw.items()}
        metadata = json.load(open(p_meta)) if p_meta.exists() else {}
        metadata["loaded_from_cache"] = True
        return Corpus(passages, embeddings, gold_mapping, metadata)
