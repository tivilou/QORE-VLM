"""Build an aligned corpus for RAG evaluation.

Standalone script to construct a gold-aligned corpus (all test questions' gold
passages + random distractors) and cache it to disk. Once built, eval_rag can
load it instantly instead of rebuilding on every run.

Usage:
    # Build for NQ validation set (default)
    python -m scripts.rag.build_corpus \\
        --dataset nq_open \\
        --output_dir data/nq_aligned_corpus \\
        --n_distractors 36000

    # Build for a custom JSONL dataset
    python -m scripts.rag.build_corpus \\
        --dataset jsonl \\
        --custom_path my_questions.jsonl \\
        --output_dir data/custom_corpus

After building, run eval_rag with:
    --corpus_mode aligned --corpus_output_dir data/nq_aligned_corpus
"""

import argparse
from pathlib import Path

from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.retrieval import make_encoder


def parse_args():
    p = argparse.ArgumentParser(description="Build aligned corpus")
    p.add_argument("--dataset", default="nq_open",
                   help="Dataset: nq_open, hotpotqa_fullwiki, jsonl")
    p.add_argument("--split", default="validation")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Limit questions (0=all)")
    p.add_argument("--custom_path", help="Path for jsonl dataset")

    p.add_argument("--output_dir", required=True,
                   help="Output directory for corpus cache")
    p.add_argument("--n_distractors", type=int, default=36000,
                   help="Number of random distractor passages")
    p.add_argument("--encoder_type", default="sentence",
                   choices=["dpr", "sentence"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild", action="store_true",
                   help="Force rebuild even if cache exists")

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Building Aligned Corpus")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Distractors: {args.n_distractors}")
    print(f"Seed: {args.seed}")
    print()

    # Check if cache exists
    output_dir = Path(args.output_dir)
    cache_files = [
        output_dir / "corpus_passages.pkl",
        output_dir / "corpus_embeddings.npy",
        output_dir / "gold_mapping.json",
    ]
    if not args.rebuild and all(f.exists() for f in cache_files):
        print("✓ Corpus cache already exists. Use --rebuild to force rebuild.")
        print(f"  Location: {output_dir}")
        return

    # Load questions
    print("Loading dataset...")
    questions = load_dataset_for_rag(
        args.dataset, args.split, args.max_samples, args.custom_path
    )
    print(f"  Loaded {len(questions)} questions")

    # TODO: Extract gold passages from questions
    # For NQ, we need to resolve gold passages from the dataset or wiki_dpr
    # This is dataset-specific logic that needs implementation per dataset

    print("\nNote: Gold passage extraction is dataset-specific.")
    print("For NQ, you need to:")
    print("  1. Match NQ questions to wiki_dpr passages, OR")
    print("  2. Use DPR retrieval to find candidate gold passages")
    print("\nThis is a complex step that requires domain knowledge.")
    print("See docs/rag_corpus_building.md for detailed instructions.")

    # Load encoder
    print(f"\nLoading encoder ({args.encoder_type})...")
    encoder = make_encoder(args.encoder_type)

    # Build corpus
    print("Building corpus...")
    config = {
        "output_dir": args.output_dir,
        "n_distractors": args.n_distractors,
        "seed": args.seed,
        "embedder": encoder.encode_passages,
        "rebuild": args.rebuild,
    }
    corpus_manager = make_corpus_manager("aligned", config)

    try:
        corpus = corpus_manager.build(questions)
        print("\n" + "=" * 70)
        print("✓ Corpus built successfully")
        print("=" * 70)
        print(f"Total passages: {len(corpus)}")
        if corpus.metadata:
            for k, v in corpus.metadata.items():
                print(f"  {k}: {v}")
        print(f"\nCached to: {args.output_dir}")
    except Exception as e:
        print(f"\n✗ Build failed: {e}")
        print("\nThis likely means the questions don't have 'gold_passages' field.")
        print("You need to preprocess your dataset to extract/resolve gold passages.")
        raise


if __name__ == "__main__":
    main()
