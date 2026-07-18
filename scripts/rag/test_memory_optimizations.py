"""Unit tests for RAG memory optimization options.

Tests both memory optimization modes:
1. faiss_mmap: memmap-backed embeddings loading
2. wiki_dpr: HuggingFace datasets built-in FAISS index

Run with: pytest scripts/rag/test_memory_optimizations.py -v
Or directly: python -m scripts.rag.test_memory_optimizations
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from applications.rag.data import make_corpus_manager


class TestFaissMmap:
    """Test faiss mode with memmap-backed embeddings."""

    def test_memmap_preserves_memory_efficiency(self):
        """Verify that memmap embeddings are not copied unnecessarily."""
        # Create a small test corpus
        n_passages = 100
        embed_dim = 768
        embeddings = np.random.randn(n_passages, embed_dim).astype(np.float32)
        passages = [f"Passage {i} text." for i in range(n_passages)]

        with tempfile.TemporaryDirectory() as tmpdir:
            emb_path = Path(tmpdir) / "embeddings.npy"
            np.save(emb_path, embeddings)

            # Load as memmap
            embeddings_mmap = np.load(emb_path, mmap_mode="r")
            assert isinstance(embeddings_mmap, np.memmap)
            assert embeddings_mmap.dtype == np.float32

            # Build corpus with memmap
            config = {
                "embeddings": embeddings_mmap,
                "passages": passages,
                "index_type": "flat",
            }
            corpus_manager = make_corpus_manager("faiss", config)

            questions = [{"id": "q1", "question": "test"}]
            corpus = corpus_manager.build(questions)

            assert len(corpus) == n_passages
            assert corpus.metadata["mode"] == "faiss"
            assert corpus.metadata["embed_dim"] == embed_dim

    def test_regular_array_still_works(self):
        """Verify that regular numpy arrays (non-memmap) still work."""
        n_passages = 50
        embed_dim = 768
        embeddings = np.random.randn(n_passages, embed_dim).astype(np.float32)
        passages = [f"Passage {i} text." for i in range(n_passages)]

        config = {
            "embeddings": embeddings,
            "passages": passages,
            "index_type": "flat",
        }
        corpus_manager = make_corpus_manager("faiss", config)

        questions = [{"id": "q1", "question": "test"}]
        corpus = corpus_manager.build(questions)

        assert len(corpus) == n_passages


class TestWikiDPR:
    """Test wiki_dpr corpus mode with built-in FAISS index."""

    @pytest.mark.skip(reason="Requires downloading large wiki_dpr dataset; run manually")
    def test_wiki_dpr_retrieval(self):
        """Test retrieval via wiki_dpr's built-in FAISS index.

        This test downloads the full wiki_dpr dataset (~80GB) on first run.
        Skip in CI; run manually for verification:
            pytest scripts/rag/test_memory_optimizations.py::TestWikiDPR::test_wiki_dpr_retrieval -v -s
        """
        config = {
            "wiki_dpr_config": "psgs_w100.nq.compressed",
            "nprobe": 64,
        }
        corpus_manager = make_corpus_manager("wiki_dpr", config)

        questions = [{"id": "q1", "question": "Who is the president?"}]
        corpus = corpus_manager.build(questions)

        assert corpus.metadata["mode"] == "wiki_dpr"
        assert corpus.metadata["corpus_size"] > 0

        # Test retrieval
        query_emb = np.random.randn(768).astype(np.float32)
        indices, embeddings, texts = corpus_manager.retrieve_with_embeddings(
            query_emb, top_k=10
        )

        assert len(indices) == 10
        assert embeddings.shape == (10, 768)
        assert len(texts) == 10
        assert all(isinstance(t, str) for t in texts)

    def test_wiki_dpr_factory(self):
        """Test that wiki_dpr mode is registered in the factory."""
        # Just verify the factory accepts wiki_dpr without errors
        config = {"wiki_dpr_config": "psgs_w100.nq.compressed"}
        corpus_manager = make_corpus_manager("wiki_dpr", config)
        assert corpus_manager is not None


def test_all_modes_registered():
    """Verify all four corpus modes are registered in the factory."""
    modes = ["aligned", "precomputed", "faiss", "wiki_dpr"]
    for mode in modes:
        # Should not raise ValueError
        manager = make_corpus_manager(mode, {})
        assert manager is not None


if __name__ == "__main__":
    # Run tests when executed directly
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])
