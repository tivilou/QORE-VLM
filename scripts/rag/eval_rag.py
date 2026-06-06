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
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="QORE-RAG Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dataset", type=str, default="natural_questions",
                        choices=["natural_questions", "hotpotqa", "multihop_rag"])
    parser.add_argument("--method", type=str, default="qore",
                        choices=["qore", "vqc", "topk", "mmr"])
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
    parser.add_argument("--vqc_encoder_path", type=str, default=None,
                        help="Path to pre-trained VQC encoder (.npz). If None, trains a new one.")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip LLM generation (only evaluate retrieval/selection quality)")
    return parser.parse_args()


def load_dataset_splits(dataset_name, max_samples=0):
    """Load QA dataset. Returns list of dicts with 'question' and 'answer' keys."""
    from datasets import load_dataset

    if dataset_name == "natural_questions":
        ds = load_dataset("nq_open", split="validation")
        samples = [{"question": item["question"], "answers": item["answer"]} for item in ds]
    elif dataset_name == "hotpotqa":
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
        samples = [{"question": item["question"], "answers": [item["answer"]]} for item in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def load_passage_corpus(dataset_name="natural_questions"):
    """
    Load passage corpus with pre-computed embeddings.

    Uses facebook/wiki_dpr (21M Wikipedia passages + DPR embeddings).
    Both queries and passages use DPR embeddings for consistency.

    Returns:
        passages: list of passage texts
        embeddings: (N_corpus, d) numpy array of passage embeddings
    """
    from datasets import load_dataset

    if dataset_name in ("natural_questions", "hotpotqa"):
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


def embed_queries(questions, embed_model_name, batch_size=64):
    """
    Embed queries using the DPR question encoder (matches corpus embeddings).

    Uses the DPR question encoder to ensure query/passage embeddings are in the
    same vector space.
    """
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    model = DPRQuestionEncoder.from_pretrained(
        "facebook/dpr-question_encoder-single-nq-base"
    )
    model.eval()

    embeddings = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.pooler_output.numpy())

    return np.vstack(embeddings)


def retrieve_passages(query_emb, corpus_embs, top_n=100):
    """Retrieve top-N passages by dot product (DPR uses inner product, not cosine)."""
    scores = corpus_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices, scores[top_indices]


def select_passages_for_method(method, query_emb, passage_embs, relevance_scores, K, args, vqc_encoder=None):
    """Select K passages using the specified method."""
    from applications.rag.selector import select_passages

    kwargs = {}
    if method == "qore":
        kwargs = {"num_reads": args.num_reads, "seed": args.seed}
    elif method == "mmr":
        kwargs = {"lambda_mmr": args.lambda_mmr}
    elif method == "vqc":
        kwargs = {"vqc_encoder": vqc_encoder, "seed": args.seed}

    indices = select_passages(
        query_embedding=query_emb,
        passage_embeddings=passage_embs,
        K=K,
        method=method,
        relevance_scores=relevance_scores,
        lam=args.lam,
        num_reads=args.num_reads,
        **kwargs,
    )
    return indices


def load_llm(model_path):
    """Load LLM for answer generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading LLM: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_answer(model, tokenizer, question, passages, max_new_tokens=128):
    """Generate answer given question + selected passages."""
    context = "\n\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])
    prompt = (
        f"Answer the question based on the given passages.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


def compute_metrics(predictions, references):
    """Compute exact match and F1."""
    from collections import Counter
    import re
    import string

    def normalize(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        return ' '.join(s.split())

    def f1(pred, ref):
        pred_toks = normalize(pred).split()
        ref_toks = normalize(ref).split()
        common = Counter(pred_toks) & Counter(ref_toks)
        n = sum(common.values())
        if n == 0:
            return 0.0
        p = n / len(pred_toks) if pred_toks else 0
        r = n / len(ref_toks) if ref_toks else 0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def exact_match(pred, refs):
        norm_pred = normalize(pred)
        return float(any(normalize(r) == norm_pred for r in refs))

    em_scores, f1_scores = [], []
    for pred, refs in zip(predictions, references):
        em_scores.append(exact_match(pred, refs))
        f1_scores.append(max(f1(pred, ref) for ref in refs))

    return {"exact_match": float(np.mean(em_scores)), "f1": float(np.mean(f1_scores))}


def train_and_save_vqc(corpus_embeddings, args, save_path):
    """Train VQC encoder and save parameters."""
    from qore.vqc.encoder import VQCEncoder
    from qore.vqc.train import train_encoder, energy_loss

    n_train = min(50, len(corpus_embeddings))
    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(corpus_embeddings), n_train, replace=False)
    train_features = corpus_embeddings[train_idx]

    encoder = VQCEncoder(n_qubits=6, n_layers=2, backend="tensorcircuit", seed=args.seed)

    print(f"  Training VQC encoder ({encoder.n_params} params, {n_train} samples)...")
    losses = train_encoder(
        encoder, train_features, K=args.K,
        loss_fn=energy_loss,
        n_steps=20, lr=0.2,
        solver="anneal", num_reads=20,
        verbose=True,
    )

    # Save trained parameters
    encoder.save(save_path)
    print(f"  Encoder saved to {save_path}")

    return encoder, losses


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    # --- Load data ---
    print(f"Loading dataset: {args.dataset}")
    samples = load_dataset_splits(args.dataset, args.max_samples)
    print(f"  {len(samples)} samples loaded")

    print(f"Loading passage corpus...")
    passages, corpus_embeddings = load_passage_corpus(args.dataset)

    # --- Embed queries (using DPR question encoder to match corpus space) ---
    print(f"Embedding queries with DPR question encoder...")
    questions = [s["question"] for s in samples]
    query_embeddings = embed_queries(questions, args.embed_model)

    # --- VQC training (for vqc method) ---
    vqc_encoder = None
    train_log = None
    if args.method == "vqc":
        encoder_path = args.vqc_encoder_path
        if encoder_path and Path(encoder_path).exists():
            from qore.vqc.encoder import VQCEncoder
            vqc_encoder = VQCEncoder.load(encoder_path)
            print(f"  Loaded pre-trained VQC encoder from {encoder_path}")
        else:
            encoder_save_path = str(output_dir / "vqc_encoder.npz")
            vqc_encoder, losses = train_and_save_vqc(
                corpus_embeddings[:10000], args, encoder_save_path
            )
            train_log = {"losses": [float(l) for l in losses], "n_steps": len(losses)}
            # Save training log
            train_log_path = output_dir / "train_log.json"
            with open(train_log_path, "w") as f:
                json.dump(train_log, f, indent=2)

    # --- Evaluation loop ---
    print(f"\nRunning evaluation: method={args.method}, K={args.K}")
    predictions = []
    references = []
    selection_times = []
    redundancy_scores = []

    # Load LLM (skip if --skip_generation)
    model, tokenizer = None, None
    if not args.skip_generation:
        model, tokenizer = load_llm(args.model_path)

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        # Retrieve top-N passages for this query
        retrieved_idx, retrieved_scores = retrieve_passages(
            query_embeddings[i], corpus_embeddings, top_n=args.num_passages
        )
        retrieved_embs = corpus_embeddings[retrieved_idx]

        # Select K passages
        t0 = time.perf_counter()
        selected_local = select_passages_for_method(
            args.method, query_embeddings[i], retrieved_embs,
            retrieved_scores, args.K, args, vqc_encoder=vqc_encoder
        )
        selection_times.append((time.perf_counter() - t0) * 1000)

        # Compute selection redundancy
        from applications.rag.selector import evaluate_selection
        sel_metrics = evaluate_selection(selected_local, set(), retrieved_embs)
        redundancy_scores.append(sel_metrics["redundancy_ratio"])

        # Map selected indices back to corpus
        selected_corpus_idx = retrieved_idx[selected_local]
        selected_passages = [passages[j] for j in selected_corpus_idx]

        # Generate answer (or skip)
        if model is not None:
            answer = generate_answer(model, tokenizer, sample["question"], selected_passages)
        else:
            answer = ""

        predictions.append(answer)
        references.append(sample["answers"])

    # --- Compute metrics ---
    metrics = {}
    if not args.skip_generation:
        metrics = compute_metrics(predictions, references)
        print(f"\n  Results: EM={metrics['exact_match']:.4f}, F1={metrics['f1']:.4f}")

    metrics["avg_selection_time_ms"] = float(np.mean(selection_times))
    metrics["avg_redundancy_ratio"] = float(np.mean(redundancy_scores))
    metrics["avg_diversity_score"] = 1.0 - metrics["avg_redundancy_ratio"]

    # --- Save results ---
    result = {
        "experiment": f"RAG-{args.dataset}",
        "method": args.method,
        "model": args.model_path,
        "dataset": args.dataset,
        "K": args.K,
        "num_samples": len(samples),
        "metrics": metrics,
        "compression": {
            "tokens_total": args.num_passages,
            "tokens_kept": args.K,
            "compression_ratio": args.K / args.num_passages,
        },
        "config": {
            "solver": args.solver,
            "num_reads": args.num_reads,
            "lam": args.lam,
            "lambda_mmr": args.lambda_mmr,
            "num_passages": args.num_passages,
            "seed": args.seed,
        },
        "train_log": train_log,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Compression: {args.num_passages} → {args.K} ({args.K/args.num_passages:.1%} kept)")


if __name__ == "__main__":
    main()
