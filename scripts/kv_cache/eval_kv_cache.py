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
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip actual generation (just verify cache mechanics)")
    return parser.parse_args()


def create_cache(policy, args, num_layers):
    """Create a fresh cache object for a single sample."""
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
            num_layers=num_layers,
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

    subtasks = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa", "musique",
    ]

    all_samples = []
    for task in subtasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test",
                              trust_remote_code=True)
            for item in ds:
                answers = item["answers"]
                if isinstance(answers, str):
                    answers = json.loads(answers) if answers.startswith("[") else [answers]
                all_samples.append({
                    "task": task,
                    "input": item["input"],
                    "context": item["context"],
                    "answers": answers,
                })
        except Exception as e:
            print(f"  Warning: could not load {task}: {e}")

    if max_samples > 0:
        all_samples = all_samples[:max_samples]

    return all_samples


def load_llm(model_path):
    """Load LLM model and tokenizer."""
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


def get_model_num_layers(model):
    """Get number of transformer layers from model config."""
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    else:
        return 32  # fallback


def get_kv_bytes_per_token(model):
    """Compute KV-cache memory per token from model config."""
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 32)
    # Use num_key_value_heads for GQA models, fall back to num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads",
                           getattr(config, "num_attention_heads", 32))
    head_dim = getattr(config, "head_dim",
                       getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32))
    dtype_bytes = 2  # fp16
    # 2 for K and V
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def evaluate_sample(model, tokenizer, sample, cache, max_new_tokens=128):
    """Run generation on a single sample with the given cache policy."""
    prompt = f"{sample['context']}\n\nQuestion: {sample['input']}\nAnswer:"

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

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

    output_len = outputs.shape[1] - input_len
    answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Get final cache size (if cache was used)
    final_cache_len = None
    if cache is not None and hasattr(cache, "get_seq_length"):
        final_cache_len = cache.get_seq_length()

    return {
        "prediction": answer,
        "input_length": input_len,
        "output_length": output_len,
        "time_ms": elapsed_ms,
        "tokens_per_sec": output_len / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        "final_cache_len": final_cache_len,
    }


def compute_metrics(predictions, references):
    """Compute F1 for LongBench."""
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    print(f"Policy: {args.policy}, Capacity: {args.max_capacity}")

    # --- Load data ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "longbench":
        samples = load_longbench(args.max_samples)
    else:
        # Placeholder for other datasets
        samples = []
        print(f"  Dataset {args.dataset} not yet implemented, using empty set")

    print(f"  {len(samples)} samples loaded")

    if len(samples) == 0:
        print("No samples to evaluate. Exiting.")
        return

    # --- Load model ---
    model, tokenizer = None, None
    num_layers = 32  # default
    kv_bytes_per_token = 2 * 32 * 8 * 128 * 2  # default estimate

    if not args.skip_generation:
        model, tokenizer = load_llm(args.model_path)
        num_layers = get_model_num_layers(model)
        kv_bytes_per_token = get_kv_bytes_per_token(model)
        print(f"  Model: {num_layers} layers, {kv_bytes_per_token} bytes/token in KV cache")

    # --- Evaluation loop ---
    print(f"\nRunning evaluation: policy={args.policy}, capacity={args.max_capacity}")
    predictions = []
    references = []
    latencies = []
    throughputs = []
    input_lengths = []

    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        # Create a FRESH cache for each sample (avoid state leaking between samples)
        cache = create_cache(args.policy, args, num_layers)

        if model is not None:
            result = evaluate_sample(model, tokenizer, sample, cache, args.max_new_tokens)
            predictions.append(result["prediction"])
            latencies.append(result["time_ms"])
            throughputs.append(result["tokens_per_sec"])
            input_lengths.append(result["input_length"])
        else:
            predictions.append("")
            input_lengths.append(0)

        references.append(sample["answers"])

    # --- Compute metrics ---
    metrics = {}
    if not args.skip_generation:
        metrics = compute_metrics(predictions, references)
        metrics["avg_latency_ms"] = float(np.mean(latencies))
        metrics["avg_throughput_tok_per_sec"] = float(np.mean(throughputs))
        print(f"\n  Results: F1={metrics['f1']:.4f}, "
              f"Latency={metrics['avg_latency_ms']:.0f}ms, "
              f"Throughput={metrics['avg_throughput_tok_per_sec']:.1f} tok/s")

    # --- Compression metrics ---
    avg_input_len = float(np.mean(input_lengths)) if input_lengths and input_lengths[0] > 0 else 4096
    effective_capacity = args.max_capacity if args.policy != "full" else avg_input_len
    compression_ratio = effective_capacity / avg_input_len if avg_input_len > 0 else 1.0

    memory_full_MB = avg_input_len * kv_bytes_per_token / (1024 * 1024)
    memory_compressed_MB = effective_capacity * kv_bytes_per_token / (1024 * 1024)
    memory_saved_MB = memory_full_MB - memory_compressed_MB

    # --- Save results ---
    result = {
        "experiment": f"KV-Cache-{args.dataset}",
        "policy": args.policy,
        "model": args.model_path,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "metrics": metrics,
        "compression": {
            "avg_input_length": round(avg_input_len),
            "tokens_kept": int(effective_capacity),
            "compression_ratio": round(compression_ratio, 4),
            "memory_full_MB": round(memory_full_MB, 1),
            "memory_compressed_MB": round(memory_compressed_MB, 1),
            "memory_saved_MB": round(memory_saved_MB, 1),
        },
        "config": {
            "max_capacity": args.max_capacity,
            "trigger_every": args.trigger_every,
            "num_sink_tokens": args.num_sink_tokens,
            "num_reads": args.num_reads,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Compression: {avg_input_len:.0f} → {effective_capacity} tokens "
          f"({compression_ratio:.1%} kept, saves ~{memory_saved_MB:.0f} MB)")


if __name__ == "__main__":
    main()
