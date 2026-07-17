"""Build a FAISS-mode corpus for full-corpus RAG evaluation.

Streams facebook/wiki_dpr (21M passages with precomputed DPR embeddings) and
saves a subset (or all) as:
  - <out>/embeddings.npy   (N, 768) float32   — DPR passage embeddings
  - <out>/passages.pkl     list[str]          — passage texts (with title prefix)

The eval script (eval_rag_refactored.py --corpus_mode faiss) loads these and
builds a FAISS IndexFlatIP for retrieval. Gold is resolved at eval time by
answer-string matching over the retrieved passages (the DPR answer-recall
convention), so no gold labels are needed here.

Usage:
    # Small subset for verification
    python -m scripts.rag.build_faiss_corpus --corpus_size 50000 --out data/wiki_dpr_50k

    # Full corpus (~80GB download, ~65GB embeddings in RAM)
    python -m scripts.rag.build_faiss_corpus --corpus_size 0 --out data/wiki_dpr_full
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Build FAISS corpus from wiki_dpr")
    p.add_argument("--corpus_size", type=int, default=50000,
                   help="Number of passages to stream (0 = full 21M corpus)")
    p.add_argument("--out", default="data/wiki_dpr_50k",
                   help="Output directory")
    p.add_argument("--wiki_config", default="psgs_w100.nq.exact",
                   help="wiki_dpr config (nq.exact matches DPR-NQ embedding space)")
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_desc = "full corpus (~21M)" if args.corpus_size == 0 else f"{args.corpus_size} passages"
    print(f"Streaming facebook/wiki_dpr [{args.wiki_config}], {size_desc}...")

    stream = load_dataset(
        "facebook/wiki_dpr", args.wiki_config,
        split="train", streaming=True, trust_remote_code=True,
    )

    passages, embeddings = [], []
    for i, item in enumerate(stream):
        if args.corpus_size and i >= args.corpus_size:
            break
        # Prefix title so the passage text is self-contained for the LLM.
        title = item.get("title", "")
        text = item["text"]
        passages.append(f"{title}. {text}" if title else text)
        embeddings.append(item["embeddings"])
        if (i + 1) % 10000 == 0:
            print(f"  streamed {i + 1} passages...")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"  collected {len(passages)} passages, dim={embeddings.shape[1]}")

    emb_path = out_dir / "embeddings.npy"
    psg_path = out_dir / "passages.pkl"
    np.save(emb_path, embeddings)
    with open(psg_path, "wb") as f:
        pickle.dump(passages, f)

    print(f"  saved embeddings -> {emb_path} ({embeddings.nbytes / 1e9:.2f} GB)")
    print(f"  saved passages   -> {psg_path}")
    print("Done. Run eval with:")
    print(f"  python -m scripts.rag.eval_rag_refactored --corpus_mode faiss \\")
    print(f"      --faiss_embeddings_path {emb_path} --faiss_passages_path {psg_path} \\")
    print(f"      --dataset nq_open --method qore")


if __name__ == "__main__":
    main()
