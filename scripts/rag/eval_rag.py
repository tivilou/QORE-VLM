"""
RAG evaluation script: run passage selection + LLM generation on QA benchmarks.

Usage:
    python -m scripts.rag.eval_rag \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset natural_questions \
        --method qore \
        --K 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="QORE-RAG Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dataset", type=str, default="natural_questions",
                        choices=["natural_questions", "hotpotqa", "multihop_rag"])
    parser.add_argument("--method", type=str, default="qore",
                        choices=["qore", "topk", "mmr"])
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--solver", type=str, default="anneal")
    parser.add_argument("--num_reads", type=int, default=50)
    parser.add_argument("--lambda_mmr", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit samples (0 = all)")
    parser.add_argument("--output_dir", type=str, default="results/rag/nq")
    parser.add_argument("--output_file", type=str, default="results.json")
    parser.add_argument("--num_passages", type=int, default=100,
                        help="Number of candidate passages to retrieve per query")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset_splits(dataset_name, max_samples=0):
    """Load QA dataset with passages. Returns list of dicts."""
    from datasets import load_dataset

    if dataset_name == "natural_questions":
        ds = load_dataset("nq_open", split="validation")
    elif dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = list(ds)
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def load_passage_corpus(dataset_name="natural_questions"):
    """
    Load passage corpus with pre-computed embeddings.

    For NQ: uses facebook/wiki_dpr (21M Wikipedia passages + DPR embeddings).
    Loaded via HuggingFace datasets — no manual download needed.

    Returns:
        passages: list of passage texts
        embeddings: (N_corpus, d) numpy array of passage embeddings
    """
    from datasets import load_dataset

    if dataset_name in ("natural_questions", "hotpotqa"):
        # facebook/wiki_dpr: 21M Wikipedia passages with DPR embeddings
        # Using the NQ subset with exact search vectors
        print("  Loading facebook/wiki_dpr corpus (this may take a while on first run)...")
        corpus = load_dataset(
            "facebook/wiki_dpr", "psgs_w100.nq.exact",
            split="train",
        )
        passages = corpus["text"]
        embeddings = np.array(corpus["embeddings"])
        print(f"  Loaded {len(passages)} passages, embedding dim={embeddings.shape[1]}")
        return passages, embeddings
    else:
        raise ValueError(f"No pre-built corpus for dataset: {dataset_name}")


def embed_texts(texts, model_name, batch_size=64):
    """Embed a list of texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings


def retrieve_passages(query_emb, corpus_embs, top_n=100):
    """Retrieve top-N passages by cosine similarity."""
    scores = corpus_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices, scores[top_indices]


def select_passages_for_method(method, query_emb, passage_embs, relevance_scores, K, args):
    """Select K passages using the specified method."""
    from applications.rag.selector import select_passages

    kwargs = {}
    if method == "qore":
        kwargs = {"num_reads": args.num_reads, "seed": args.seed}
    elif method == "mmr":
        kwargs = {"lambda_mmr": args.lambda_mmr}

    indices = select_passages(
        query_embedding=query_emb,
        passage_embeddings=passage_embs,
        K=K,
        method=method,
        relevance_scores=relevance_scores,
        lam=args.lam,
        **kwargs,
    )
    return indices


def generate_answer(model, tokenizer, question, passages, max_new_tokens=128):
    """Generate answer given question + selected passages."""
    # Format context
    context = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
    prompt = (
        f"Answer the question based on the given passages.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def compute_metrics(predictions, references):
    """Compute accuracy and F1."""
    from collections import Counter

    def normalize_answer(s):
        """Normalize answer for comparison."""
        import re
        import string
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        s = ' '.join(s.split())
        return s

    def f1_score(prediction, ground_truth):
        pred_tokens = normalize_answer(prediction).split()
        gt_tokens = normalize_answer(ground_truth).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(gt_tokens) if gt_tokens else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def exact_match(prediction, ground_truths):
        norm_pred = normalize_answer(prediction)
        return float(any(normalize_answer(gt) == norm_pred for gt in ground_truths))

    em_scores = []
    f1_scores = []

    for pred, refs in zip(predictions, references):
        if isinstance(refs, str):
            refs = [refs]
        em_scores.append(exact_match(pred, refs))
        f1_scores.append(max(f1_score(pred, ref) for ref in refs))

    return {
        "exact_match": np.mean(em_scores),
        "f1": np.mean(f1_scores),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Loading dataset: {args.dataset}")
    samples = load_dataset_splits(args.dataset, args.max_samples)
    print(f"  {len(samples)} samples loaded")

    # Load passage corpus with pre-computed embeddings
    print(f"Loading passage corpus...")
    passages, corpus_embeddings = load_passage_corpus(args.dataset)

    # Output setup
    output_path = Path(args.output_dir) / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # VQC training (only for qore method)
    train_log = None
    if args.method == "qore":
        print(f"\nTraining VQC encoder...")
        train_log_path = Path(args.output_dir) / "train_log.json"
        train_log = _train_vqc_encoder(
            corpus_embeddings, args, train_log_path
        )

    # Save results
    result = {
        "experiment": f"RAG-{args.dataset}",
        "method": args.method,
        "model": args.model_path,
        "dataset": args.dataset,
        "K": args.K,
        "solver": args.solver,
        "config": {
            "num_reads": args.num_reads,
            "lam": args.lam,
            "lambda_mmr": args.lambda_mmr,
            "num_passages": args.num_passages,
            "seed": args.seed,
        },
        "compression": {
            "tokens_total": args.num_passages,
            "tokens_kept": args.K,
            "compression_ratio": args.K / args.num_passages,
        },
        "num_samples": len(samples),
        "corpus_size": len(passages),
        "status": "ready",
        "train_log": train_log,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to {output_path}")
    print(f"Compression: {args.num_passages} → {args.K} "
          f"({args.K/args.num_passages:.1%} kept)")


def _train_vqc_encoder(corpus_embeddings, args, log_path):
    """Train VQC encoder on a small sample of corpus embeddings."""
    from qore.vqc.encoder import VQCEncoder
    from qore.vqc.train import train_encoder, energy_loss

    # Use a small random subset for training
    n_train = min(50, len(corpus_embeddings))
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(corpus_embeddings), n_train, replace=False)
    train_features = corpus_embeddings[train_idx]

    encoder = VQCEncoder(n_qubits=6, n_layers=2, backend="tensorcircuit", seed=args.seed)

    losses = train_encoder(
        encoder, train_features, K=args.K,
        loss_fn=energy_loss,
        n_steps=20, lr=0.2,
        solver="anneal", num_reads=20,
        verbose=True,
    )

    # Save training log
    train_log = {
        "n_train_samples": n_train,
        "n_steps": 20,
        "losses": [float(l) for l in losses],
        "final_loss": float(losses[-1]),
    }

    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"  Training log saved to {log_path}")

    return train_log


if __name__ == "__main__":
    main()
