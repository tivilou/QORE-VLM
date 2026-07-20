"""End-to-end RAG evaluation: retrieval → selection → generation → scoring.

Unified evaluation script supporting three corpus modes (aligned, precomputed,
faiss) and three selection methods (qore, mmr, topk). The refactored architecture
decouples corpus management, retrieval, selection, generation, and evaluation so
each can be swapped or tested independently.

Usage:
    # Aligned corpus mode (recommended default, gold-guaranteed)
    python -m scripts.rag.eval_rag \\
        --dataset nq_open \\
        --corpus_mode aligned \\
        --corpus_output_dir data/nq_corpus \\
        --method qore \\
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \\
        --max_samples 100

    # Precomputed mode (use dataset's own candidates, e.g. HotpotQA distractor)
    python -m scripts.rag.eval_rag \\
        --dataset hotpotqa_distractor \\
        --corpus_mode precomputed \\
        --method qore \\
        --skip_generation

    # FAISS mode (full 21M corpus, requires FAISS + embeddings)
    python -m scripts.rag.eval_rag \\
        --dataset nq_open \\
        --corpus_mode faiss \\
        --faiss_embeddings_path data/wiki_dpr_embeddings.npy \\
        --method qore

See docs/rag_corpus_modes.md for mode details.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from applications.rag.data import load_dataset_for_rag, make_corpus_manager
from applications.rag.evaluation import Evaluator
from applications.rag.generation import Generator
from applications.rag.retrieval import make_encoder
from applications.rag.selector import select_passages


def answer_has_match_in_text(answer: str, text: str) -> bool:
    """Check if answer appears in text with token boundaries.

    More strict than simple substring match to avoid false positives like
    "2012" matching "2012-2013" when the gold answer is specifically "2012".

    Uses word boundaries (\b in regex) or whitespace/punctuation delimiters.
    """
    import re
    # Normalize
    answer_norm = answer.lower().strip()
    text_norm = text.lower()

    # Simple substring check first (fast path)
    if answer_norm not in text_norm:
        return False

    # Token boundary check: require word boundaries around the answer
    # \b matches position between \w and \W (word/non-word boundary)
    pattern = r'\b' + re.escape(answer_norm) + r'\b'
    return re.search(pattern, text_norm) is not None


def parse_args():
    p = argparse.ArgumentParser(description="RAG end-to-end evaluation")

    # Dataset
    p.add_argument("--dataset", default="nq_open",
                   help="Dataset name: nq_open, hotpotqa_distractor, hotpotqa_fullwiki, jsonl")
    p.add_argument("--split", default="validation", help="Dataset split")
    p.add_argument("--max_samples", type=int, default=0, help="Limit samples (0=all)")
    p.add_argument("--custom_path", help="Path for jsonl dataset")

    # Corpus mode
    p.add_argument("--corpus_mode", default="aligned",
                   choices=["aligned", "precomputed", "faiss", "wiki_dpr"],
                   help="Corpus management mode")
    p.add_argument("--corpus_output_dir", help="Cache dir for aligned corpus")
    p.add_argument("--n_distractors", type=int, default=36000,
                   help="Distractors for aligned mode")
    p.add_argument("--faiss_embeddings_path", help="Embeddings .npy for faiss mode")
    p.add_argument("--faiss_passages_path", help="Passages list for faiss mode")
    p.add_argument("--faiss_mmap", action="store_true",
                   help="Load embeddings as memmap (np.load mmap_mode='r'). "
                        "Halves peak RAM: Python holds no separate copy of the "
                        "embedding array — only the FAISS index stays in RAM. "
                        "Use for large corpora (e.g. full 21M wiki_dpr).")
    # wiki_dpr compressed mode
    p.add_argument("--wiki_dpr_config", default="psgs_w100.nq.compressed",
                   help="facebook/wiki_dpr config for wiki_dpr corpus mode")
    p.add_argument("--wiki_dpr_cache_dir", default=None,
                   help="HuggingFace cache dir for wiki_dpr dataset")
    p.add_argument("--wiki_dpr_nprobe", type=int, default=64,
                   help="IVFPQ search breadth for wiki_dpr mode (higher=more accurate)")

    # Retrieval
    p.add_argument("--encoder_type", default="sentence",
                   choices=["dpr", "sentence"],
                   help="Encoder type")
    p.add_argument("--top_k_retrieval", type=int, default=50,
                   help="Retrieve top-K candidates per query")

    # Selection
    p.add_argument("--method", default="qore",
                   choices=["qore", "mmr", "topk"],
                   help="Selection method")
    p.add_argument("--K", type=int, default=5, help="Select K passages")
    p.add_argument("--num_reads", type=int, default=100,
                   help="SA reads for QORE (ignored for mmr/topk)")
    p.add_argument("--lam", type=float, default=2.0,
                   help="QUBO cardinality penalty weight (keep at ~2.0 for proper constraint enforcement)")
    p.add_argument("--gamma", type=float, default=None,
                   help="QUBO redundancy weight (None=auto-tune to 1.0; "
                        "lower=favor relevance, higher=favor diversity; "
                        "try 0.05-0.5 for better answer coverage)")
    p.add_argument("--qore_prefilter_size", type=int, default=None,
                   help="QORE relevance-first candidate pool size for large N "
                        "(default: max(K*3, 15); try 15-20 to reduce low-relevance risk)")
    p.add_argument("--direct_solve_max_n", type=int, default=20,
                   help="QORE: if N <= this, solve QUBO directly without prefilter. "
                        "Keep at 20 (below Top-K retrieval size) so prefilter always runs. "
                        "Increase only for small-N demos.")
    p.add_argument("--lambda_mmr", type=float, default=0.7,
                   help="MMR lambda (1=relevance, 0=diversity)")

    # Generation
    p.add_argument("--model_path",
                   default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HF model for answer generation")
    p.add_argument("--skip_generation", action="store_true",
                   help="Skip LLM generation (selection-only eval)")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max tokens for answer generation")

    # Output
    p.add_argument("--output_dir", default="results/rag",
                   help="Directory for results JSON")
    p.add_argument("--output_file", help="Result filename (auto if not set)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("=" * 70)
    print("RAG End-to-End Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Corpus mode: {args.corpus_mode}")
    print(f"Selection: {args.method}, K={args.K}")
    print(f"Seed: {args.seed}")
    print()

    # ──────────────────────────────────────────────────────────────────────
    # 1. Load dataset
    # ──────────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    questions = load_dataset_for_rag(
        args.dataset, args.split, args.max_samples, args.custom_path
    )
    print(f"  Loaded {len(questions)} questions")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Build/load corpus
    # ──────────────────────────────────────────────────────────────────────
    print(f"Building corpus (mode={args.corpus_mode})...")
    corpus_config = {
        "output_dir": args.corpus_output_dir,
        "n_distractors": args.n_distractors,
        "seed": args.seed,
    }
    if args.corpus_mode == "faiss":
        if not args.faiss_embeddings_path:
            raise ValueError("faiss mode requires --faiss_embeddings_path")
        import pickle
        mmap_mode = "r" if args.faiss_mmap else None
        if args.faiss_mmap:
            print("  [faiss] loading embeddings as memmap (peak RAM ~halved)")
        corpus_config["embeddings"] = np.load(args.faiss_embeddings_path, mmap_mode=mmap_mode)
        with open(args.faiss_passages_path, "rb") as f:
            corpus_config["passages"] = pickle.load(f)
    elif args.corpus_mode == "wiki_dpr":
        corpus_config["wiki_dpr_config"] = args.wiki_dpr_config
        corpus_config["cache_dir"] = args.wiki_dpr_cache_dir
        corpus_config["nprobe"] = args.wiki_dpr_nprobe

    # FAISS / wiki_dpr modes use wiki_dpr's DPR passage embeddings, so queries
    # MUST be encoded with the DPR question encoder to share that vector space.
    # A sentence encoder would put queries in a different space → garbage retrieval.
    if args.corpus_mode in ("faiss", "wiki_dpr") and args.encoder_type != "dpr":
        print(f"  [{args.corpus_mode}] forcing --encoder_type dpr "
              f"(was {args.encoder_type!r}; corpus is DPR space)")
        args.encoder_type = "dpr"

    # Encoder for aligned mode (needed to embed gold passages if missing)
    encoder = make_encoder(args.encoder_type)
    if args.corpus_mode == "aligned":
        corpus_config["embedder"] = encoder.encode_passages

    corpus_manager = make_corpus_manager(args.corpus_mode, corpus_config)
    corpus = corpus_manager.build(questions)
    print(f"  Corpus ready: {len(corpus)} passages")
    if corpus.metadata:
        for k, v in corpus.metadata.items():
            print(f"    {k}: {v}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Load generator (if needed)
    # ──────────────────────────────────────────────────────────────────────
    generator = None
    if not args.skip_generation:
        print(f"Loading generator: {args.model_path}...")
        generator = Generator(
            args.model_path,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=True,
        )
        print("  Generator ready")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Evaluation loop
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nEvaluating {len(questions)} questions...")
    evaluator = Evaluator()

    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(questions)}")

        question = q["question"]
        qid = q["id"]
        gold_answers = q.get("answers", [])

        # Encode query
        query_emb = encoder.encode_queries([question])[0]

        # Retrieve
        if args.corpus_mode == "precomputed":
            # Per-question candidates
            candidates = q.get("candidates", [])
            cand_texts = [c["text"] for c in candidates]
            cand_embs = encoder.encode_passages(cand_texts)
            retrieved_idx, retrieval_scores = corpus_manager.retrieve(
                query_emb, args.top_k_retrieval, candidate_embeddings=cand_embs
            )
            retrieved_embs = cand_embs[retrieved_idx]
            retrieved_texts = None
        elif args.corpus_mode == "wiki_dpr":
            # wiki_dpr mode: retrieval returns embeddings, texts, and scores
            retrieved_idx, retrieved_embs, retrieved_texts, retrieval_scores = corpus_manager.retrieve_with_embeddings(
                query_emb, args.top_k_retrieval
            )
        else:
            # Shared corpus (aligned or faiss)
            retrieved_idx, retrieval_scores = corpus_manager.retrieve(query_emb, args.top_k_retrieval)
            retrieved_embs = corpus.embeddings[retrieved_idx]
            retrieved_texts = None

        # Select
        t0 = time.perf_counter()
        selected_local = select_passages(
            query_emb,
            retrieved_embs,
            K=args.K,
            method=args.method,
            num_reads=args.num_reads,
            lam=args.lam,
            gamma=args.gamma,
            lambda_mmr=args.lambda_mmr,
            seed=args.seed,
            relevance_scores=retrieval_scores,
            qore_prefilter_size=args.qore_prefilter_size,
            direct_solve_max_n=args.direct_solve_max_n,
        )
        selection_time_ms = (time.perf_counter() - t0) * 1000

        selected_global = retrieved_idx[selected_local]
        selected_embs = retrieved_embs[selected_local]

        # Gold indices (mode-dependent)
        if args.corpus_mode == "precomputed":
            gold_local = set(q.get("gold_local_indices", []))
            # Map to retrieved space
            gold_in_retrieved = gold_local & set(range(len(retrieved_idx)))
        elif args.corpus_mode in ("faiss", "wiki_dpr"):
            # No gold labels for open-domain NQ: use the DPR answer-recall
            # convention — a retrieved passage is "gold" if it contains any
            # gold answer string with token boundaries (stricter than substring).
            gold_in_retrieved = set()
            if args.corpus_mode == "wiki_dpr":
                # Use retrieved_texts directly
                for j, text in enumerate(retrieved_texts):
                    if any(answer_has_match_in_text(ans, text) for ans in gold_answers if ans):
                        gold_in_retrieved.add(j)
            else:
                # faiss mode: access corpus.passages
                for j, gidx in enumerate(retrieved_idx):
                    passage_text = corpus.passages[gidx]
                    if any(answer_has_match_in_text(ans, passage_text) for ans in gold_answers if ans):
                        gold_in_retrieved.add(j)
        else:
            gold_global = corpus.gold_for(qid)
            # Map global gold to retrieved local indices
            gold_in_retrieved = set()
            for j, idx in enumerate(retrieved_idx):
                if idx in gold_global:
                    gold_in_retrieved.add(j)

        # Generate answer
        prediction = None
        generation_time_ms = 0.0
        if generator:
            if args.corpus_mode == "precomputed":
                selected_texts = [candidates[j]["text"] for j in selected_global]
            elif args.corpus_mode == "wiki_dpr":
                selected_texts = [retrieved_texts[j] for j in selected_local]
            else:
                selected_texts = [corpus.passages[j] for j in selected_global]
            t0 = time.perf_counter()
            prediction = generator.generate(question, selected_texts)
            generation_time_ms = (time.perf_counter() - t0) * 1000

        # Evaluate
        # answer_hit_at_retrieved: for open-domain modes, whether ANY retrieved
        # candidate contains a gold answer string — this is the retrieval upper bound.
        # For aligned/precomputed, retrieval guarantees gold so we skip this.
        if args.corpus_mode in ("faiss", "wiki_dpr"):
            answer_hit = len(gold_in_retrieved) > 0
        else:
            answer_hit = None

        evaluator.evaluate_sample(
            question_id=qid,
            selected_indices=set(selected_local),
            selected_embeddings=selected_embs,
            gold_indices=gold_in_retrieved,
            prediction=prediction,
            gold_answers=gold_answers,
            selection_time_ms=selection_time_ms,
            generation_time_ms=generation_time_ms,
            answer_hit_at_retrieved=answer_hit,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 5. Aggregate and save results
    # ──────────────────────────────────────────────────────────────────────
    agg = evaluator.aggregate()
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Samples: {agg.get('n_samples', 0)}")
    if "n_with_gold" in agg:
        n_total = agg.get('n_samples', 0)
        n_with = agg.get('n_with_gold', 0)
        n_fail = agg.get('n_retrieval_failure', 0)
        print(f"Retrieval: {n_with}/{n_total} hit gold in Top-{args.top_k_retrieval}"
              + (f" | {n_fail} retrieval failures" if n_fail > 0 else ""))
    if "mean_recall" in agg:
        print(f"Recall@{args.K}: {agg['mean_recall']:.4f} ± {agg.get('std_recall', 0):.4f}"
              " (conditional on retrieval hit)")
    if "mean_redundancy" in agg:
        print(f"Redundancy: {agg['mean_redundancy']:.4f} ± {agg.get('std_redundancy', 0):.4f}")
    if "mean_em" in agg:
        print(f"EM: {agg['mean_em']:.4f} ± {agg.get('std_em', 0):.4f}")
    if "mean_f1" in agg:
        print(f"F1: {agg['mean_f1']:.4f} ± {agg.get('std_f1', 0):.4f}")
    print()

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.output_file:
        args.output_file = f"{args.method}_K{args.K}_seed{args.seed}.json"
    output_path = output_dir / args.output_file

    result = {
        "config": vars(args),
        "corpus_metadata": corpus.metadata,
        "metrics": agg,
        "samples": evaluator.samples,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
