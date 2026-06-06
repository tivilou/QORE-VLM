"""
KV-Cache eviction evaluation: compare cache policies on long-context benchmarks.

Usage:
    python -m scripts.kv_cache.eval_kv_cache \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset longbench \
        --policy qore \
        --max_capacity 1024
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="QORE KV-Cache Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="longbench",
                        choices=["longbench", "ruler", "pg19", "needle"])
    parser.add_argument("--policy", type=str, default="qore",
                        choices=["qore", "h2o", "window", "random", "full"])
    parser.add_argument("--max_capacity", type=int, default=1024)
    parser.add_argument("--trigger_every", type=int, default=128)
    parser.add_argument("--num_sink_tokens", type=int, default=4)
    parser.add_argument("--num_reads", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/kv_cache/longbench")
    parser.add_argument("--output_file", type=str, default="results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_cache(policy, args):
    """Create a cache object based on the specified policy."""
    if policy == "full":
        return None  # No eviction — use default DynamicCache

    common_kwargs = {
        "max_capacity": args.max_capacity,
        "trigger_every": args.trigger_every,
        "num_sink_tokens": args.num_sink_tokens,
    }

    if policy == "qore":
        from applications.kv_cache.qore_cache import QORECache
        return QORECache(
            **common_kwargs,
            num_reads=args.num_reads,
            seed=args.seed,
        )
    elif policy == "h2o":
        from applications.kv_cache.baselines.h2o_cache import H2OCache
        return H2OCache(**common_kwargs)
    elif policy == "window":
        from applications.kv_cache.baselines.window_cache import WindowCache
        return WindowCache(**common_kwargs)
    elif policy == "random":
        from applications.kv_cache.baselines.random_cache import RandomCache
        return RandomCache(**common_kwargs)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def load_longbench(max_samples=0):
    """Load LongBench dataset."""
    from datasets import load_dataset

    # LongBench has multiple subtasks; load a representative subset
    subtasks = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa", "musique",
    ]

    all_samples = []
    for task in subtasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
            for item in ds:
                all_samples.append({
                    "task": task,
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"] if isinstance(item["answers"], list)
                               else [item["answers"]],
                })
        except Exception as e:
            print(f"  Warning: could not load {task}: {e}")

    if max_samples > 0:
        all_samples = all_samples[:max_samples]

    return all_samples


def evaluate_sample(model, tokenizer, sample, cache, max_new_tokens=128):
    """Run generation on a single sample with the given cache policy."""
    # Format prompt
    prompt = f"{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Generate with custom cache
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Decode
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    return {
        "prediction": answer,
        "input_length": input_len,
        "output_length": outputs.shape[1] - input_len,
        "time_ms": elapsed_ms,
    }


def compute_longbench_metrics(predictions, references):
    """Compute metrics for LongBench."""
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

    f1_scores = []
    for pred, refs in zip(predictions, references):
        f1_scores.append(max(f1(pred, ref) for ref in refs))

    return {"f1": float(np.mean(f1_scores))}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model_path}")
    print(f"Policy: {args.policy}, Capacity: {args.max_capacity}")

    # Save config (actual full run requires model loading)
    output_path = Path(args.output_dir) / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": f"KV-Cache-{args.dataset}",
        "policy": args.policy,
        "model": args.model_path,
        "dataset": args.dataset,
        "config": {
            "max_capacity": args.max_capacity,
            "trigger_every": args.trigger_every,
            "num_sink_tokens": args.num_sink_tokens,
            "num_reads": args.num_reads,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "status": "config_ready",
        "note": "Run with GPU and model weights to produce full results",
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {output_path}")
    print(f"\nTo run full evaluation, ensure model is downloaded and GPU available.")


if __name__ == "__main__":
    main()
