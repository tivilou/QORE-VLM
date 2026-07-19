"""wiki_dpr compressed mode: retrieval via HuggingFace datasets built-in FAISS index.

Uses facebook/wiki_dpr psgs_w100.nq.compressed which ships with a prebuilt
IVFPQ index. Retrieval calls ds.get_nearest_examples() — the full 21M-passage
embedding matrix never lands in system RAM; only the IVFPQ index (~few GB) is
loaded.

This is the recommended mode for full-corpus NQ evaluation when disk/RAM is
limited: no need to download and store embeddings.npy (~65GB), and no need to
hold 65GB in RAM for a flat FAISS index. The compressed index trades a small
recall penalty (~1-2%) for massive memory savings.

config keys:
    wiki_dpr_config: dataset config name (default: "psgs_w100.nq.compressed")
    cache_dir: optional HF cache dir
    nprobe: IVFPQ search breadth (default: 64; higher = more accurate but slower)
"""

from __future__ import annotations

import numpy as np

from .corpus_manager import Corpus, CorpusManager


class WikiDPRCorpusManager(CorpusManager):
    """Full-corpus retrieval via HuggingFace datasets' built-in FAISS index.

    Loads facebook/wiki_dpr with a prebuilt compressed FAISS index. Retrieval
    is done via ds.get_nearest_examples() which queries the index without
    loading the full embedding matrix into RAM.

    Unlike the regular faiss mode, this does not require pre-downloading
    embeddings.npy or passages.pkl files.
    """

    def build(self, questions: list[dict]) -> Corpus:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "wiki_dpr mode needs datasets: pip install datasets"
            ) from e

        wiki_config = self.config.get("wiki_dpr_config", "psgs_w100.nq.compressed")
        cache_dir = self.config.get("cache_dir")
        nprobe = self.config.get("nprobe", 64)

        print(f"  Loading facebook/wiki_dpr [{wiki_config}]...")
        print(f"  (This downloads the dataset + prebuilt FAISS index on first run)")

        self._dataset = load_dataset(
            "facebook/wiki_dpr",
            wiki_config,
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # Load the built-in FAISS index
        if not self._dataset.list_indexes():
            print("  Building FAISS index (one-time setup)...")
            self._dataset.add_faiss_index(column="embeddings")

        # Set nprobe for IVFPQ index (controls search quality/speed tradeoff)
        if hasattr(self._dataset.get_index("embeddings"), "faiss_index"):
            faiss_index = self._dataset.get_index("embeddings").faiss_index
            if hasattr(faiss_index, "nprobe"):
                faiss_index.nprobe = nprobe
                print(f"  Set FAISS nprobe={nprobe}")

        # For wiki_dpr mode, we don't store embeddings or passages in memory.
        # They are accessed on-demand from the HF dataset during retrieval.
        # Create a minimal Corpus object to satisfy the interface.
        metadata = {
            "mode": "wiki_dpr",
            "corpus_size": len(self._dataset),
            "wiki_config": wiki_config,
            "nprobe": nprobe,
        }
        # Pass empty lists for passages/embeddings since they're not used
        self._corpus = Corpus([], np.array([]), {}, metadata)
        return self._corpus

    def retrieve_with_embeddings(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """Retrieve top-k passages with their embeddings and texts.

        Returns:
            indices: np.array of shape (top_k,) — row indices in the dataset
            embeddings: np.array of shape (top_k, d) — passage embeddings
            texts: list of top_k passage texts (with title prefix)
            scores: np.array of shape (top_k,) — DPR retrieval scores (inner product)
        """
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        # get_nearest_examples returns (scores, retrieved_examples)
        # retrieved_examples is a dict with keys: 'id', 'text', 'title', 'embeddings', etc.
        scores, retrieved = self._dataset.get_nearest_examples(
            "embeddings", q[0], k=top_k
        )

        # Build passage texts (title + text, matching build_faiss_corpus.py format)
        texts = []
        for title, text in zip(retrieved["title"], retrieved["text"]):
            passage = f"{title}. {text}" if title else text
            texts.append(passage)

        # Extract embeddings
        embeddings = np.array(retrieved["embeddings"], dtype=np.float32)

        # Indices are just 0..k-1 (local indices within the retrieved set)
        # Note: retrieved["id"] contains the global dataset row IDs, but we
        # don't need them since we pass texts directly in the eval loop
        indices = np.arange(top_k, dtype=np.int32)

        # Convert scores to numpy array (DPR uses inner product as similarity)
        scores = np.array(scores, dtype=np.float32)

        return indices, embeddings, texts, scores
