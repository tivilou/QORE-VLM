"""Precomputed corpus mode: use a dataset's own retrieved candidates.

Some QA datasets ship per-question candidate passages (e.g. HotpotQA
distractor's 10 context paragraphs, or a DPR-retrieved top-100). In that case
there is no shared corpus to build — each question carries its own candidate
pool and gold labels. This manager just wraps those per-question candidates
behind the common interface.

Because candidates are per-question, `passages`/`embeddings` on the returned
Corpus are left empty; `retrieve()` is a no-op that expects the caller to pass
the question's own candidate embeddings. The value here is the UNIFORM API +
gold mapping, so eval code is identical across modes.
"""

from __future__ import annotations

import numpy as np

from .corpus_manager import Corpus, CorpusManager


class PrecomputedCorpusManager(CorpusManager):
    """Per-question candidate pools (no shared corpus).

    config keys:
        embedder: callable(list[str]) -> (n, d) to encode candidate texts
            when the dataset provides text-only candidates.
    """

    def build(self, questions: list[dict]) -> Corpus:
        # Per-question candidates live on each question dict under
        # 'candidates' (list of {'text', 'embedding'?}) and 'gold_passages'.
        gold_mapping = {}
        n_with_gold = 0
        for q in questions:
            qid = q.get("id")
            # Gold indices are LOCAL to this question's candidate list.
            gold_local = set(q.get("gold_local_indices", []))
            gold_mapping[qid] = gold_local
            if gold_local:
                n_with_gold += 1

        metadata = {
            "mode": "precomputed",
            "n_questions": len(questions),
            "n_questions_with_gold": n_with_gold,
            "per_question_candidates": True,
        }
        # Empty shared arrays — candidates are carried per question.
        self._corpus = Corpus([], np.empty((0, 0), np.float32), gold_mapping, metadata)
        return self._corpus

    def retrieve(self, query_embedding, top_k, candidate_embeddings=None):
        """Rank this question's own candidates by dot product.

        Unlike the shared-corpus modes, the caller passes the question's
        candidate_embeddings (N_cand, d); we return indices LOCAL to that list.
        """
        if candidate_embeddings is None:
            raise ValueError(
                "precomputed mode needs per-question candidate_embeddings "
                "passed to retrieve()."
            )
        scores = np.asarray(candidate_embeddings) @ np.asarray(query_embedding)
        k = min(top_k, len(scores))
        top = np.argsort(scores)[::-1][:k]
        return top, scores[top]
