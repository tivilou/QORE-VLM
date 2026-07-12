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
                        choices=["qore", "h2o", "snapkv", "pyramidkv",
                                 "window", "random", "full"])
    parser.add_argument("--quality", type=str, default="attention",
                        choices=["attention", "keynorm", "vqc"],
                        help="QORE quality signal: real cumulative attention "
                             "(default), key-norm proxy, or vqc (quantum encoder "
                             "produces both aᵢ and bᵢⱼ — ablation).")
    parser.add_argument("--window", type=int, default=32,
                        help="SnapKV observation window size.")
    # --- Quantum ablation knobs (QORE only) ---
    parser.add_argument("--solver_method", type=str, default="anneal",
                        choices=["anneal", "greedy", "qaoa_tc", "qaoa_pl", "qaoa_qk"],
                        help="QUBO solver. anneal=SA (default); qaoa_*=QAOA ablation.")
    parser.add_argument("--redundancy_method", type=str, default="cosine",
                        choices=["cosine", "rbf", "quantum"],
                        help="bᵢⱼ signal. quantum=fidelity kernel (ablation).")
    parser.add_argument("--qaoa_p", type=int, default=2, help="QAOA depth p.")
    parser.add_argument("--qaoa_maxiter", type=int, default=50,
                        help="QAOA optimizer iterations.")
    parser.add_argument("--quantum_backend", type=str, default="tensorcircuit",
                        choices=["tensorcircuit", "pennylane", "qiskit"])
    parser.add_argument("--quantum_n_qubits", type=int, default=6)
    parser.add_argument("--quantum_n_layers", type=int, default=2)
    parser.add_argument("--max_capacity", type=int, default=1024)
    parser.add_argument("--trigger_every", type=int, default=128)
    parser.add_argument("--num_sink_tokens", type=int, default=4)
    parser.add_argument("--num_reads", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=7900,
                        help="Max input tokens; longer prompts are middle-truncated "
                             "(keeps context head + trailing question)")
    parser.add_argument("--output_dir", type=str, default="results/kv_cache/longbench")
    parser.add_argument("--output_file", type=str, default="results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip actual generation (just verify cache mechanics)")
    return parser.parse_args()


# Policies whose eviction uses real cumulative attention → need forward hooks
# + eager attention. Window/Random/Full select by position/chance, no attention.
ATTENTION_POLICIES = {"qore", "h2o", "snapkv", "pyramidkv"}


def policy_needs_attention(policy, quality="attention"):
    """Whether this policy requires attention capture for its eviction signal."""
    if policy == "qore":
        return quality == "attention"
    return policy in {"h2o", "snapkv", "pyramidkv"}


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
            quality=args.quality,
            solver_method=args.solver_method,
            redundancy_method=args.redundancy_method,
            qaoa_p=args.qaoa_p,
            qaoa_maxiter=args.qaoa_maxiter,
            quantum_backend=args.quantum_backend,
            quantum_n_qubits=args.quantum_n_qubits,
            quantum_n_layers=args.quantum_n_layers,
            seed=args.seed,
        )
    elif policy == "h2o":
        from applications.kv_cache.baselines.h2o_cache import H2OCache
        return H2OCache(**common_kwargs, num_layers=num_layers)
    elif policy == "snapkv":
        from applications.kv_cache.baselines.snapkv_cache import SnapKVCache
        return SnapKVCache(**common_kwargs, num_layers=num_layers, window=args.window)
    elif policy == "pyramidkv":
        from applications.kv_cache.baselines.pyramidkv_cache import PyramidKVCache
        return PyramidKVCache(**common_kwargs, num_layers=num_layers)
    elif policy == "window":
        from applications.kv_cache.baselines.window_cache import WindowCache
        return WindowCache(**common_kwargs, num_layers=num_layers)
    elif policy == "random":
        from applications.kv_cache.baselines.random_cache import RandomCache
        return RandomCache(**common_kwargs, num_layers=num_layers)
    else:
        raise ValueError(f"Unknown policy: {policy}")


def load_longbench(max_samples=0):
    """Load LongBench dataset."""
    from datasets import load_dataset

    subtasks = [
        "narrativeqa", "qasper", "multifieldqa_en",
        "hotpotqa", "2wikimqa", "musique",
    ]

    # Load each subtask separately so we can sample evenly across all of them.
    per_task = {}
    for task in subtasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test",
                              trust_remote_code=True)
            items = []
            for item in ds:
                answers = item["answers"]
                if isinstance(answers, str):
                    answers = json.loads(answers) if answers.startswith("[") else [answers]
                items.append({
                    "task": task,
                    "input": item["input"],
                    "context": item["context"],
                    "answers": answers,
                })
            per_task[task] = items
        except Exception as e:
            print(f"  Warning: could not load {task}: {e}")

    if max_samples <= 0:
        # Full run: use everything, grouped by task.
        all_samples = [s for task in subtasks if task in per_task for s in per_task[task]]
        return all_samples

    # Stratified sampling: take an even share from each available subtask so the
    # first N samples aren't all from one task (e.g. narrativeqa).
    tasks_available = [t for t in subtasks if per_task.get(t)]
    per_task_quota = max(1, max_samples // len(tasks_available))
    all_samples = []
    for task in tasks_available:
        all_samples.extend(per_task[task][:per_task_quota])
    # Top up to exactly max_samples if integer division left a shortfall.
    # Distribute the remainder round-robin across tasks (one extra each pass)
    # so it doesn't all come from the first task (e.g. narrativeqa).
    depth = per_task_quota
    while len(all_samples) < max_samples:
        added_this_round = False
        for task in tasks_available:
            if depth < len(per_task[task]):
                all_samples.append(per_task[task][depth])
                added_this_round = True
                if len(all_samples) >= max_samples:
                    break
        if not added_this_round:
            break  # every task exhausted
        depth += 1
    all_samples = all_samples[:max_samples]

    return all_samples


def load_llm(model_path, attn_implementation="eager"):
    """Load LLM model and tokenizer.

    Attention-based eviction (H2O/SnapKV/PyramidKV/QORE) needs the attention
    weight matrices, which only the "eager" implementation exposes. SDPA and
    FlashAttention fuse the softmax and never materialize the weights, so hooks
    would see None. We therefore default to eager for these experiments.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading LLM: {model_path} (attn={attn_implementation})")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # bfloat16, NOT float16: with attn_implementation="eager" + output_attentions
    # (needed to capture the cumulative-attention quality signal), fp16's limited
    # mantissa makes the returned attention weights imprecise. Top-K baselines
    # (H2O) tolerate it since they only need a rough ranking, but QORE's QUBO
    # weights those exact values and produces degenerate keep-sets → garbage
    # generation on larger models (verified on Qwen2.5-7B: fp16 garbles, bf16 is
    # coherent, identical otherwise). bf16 has the same 2-byte footprint and
    # dynamic range as fp16 with more usable precision here, and is the dtype
    # Qwen2.5 / Llama-3 are trained and released in.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_implementation,
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


# Official LongBench task-specific prompt templates (dataset2prompt). Using one
# generic prompt for every task depresses scores and diverges from the published
# protocol; these match the official per-task instructions.
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a "
        "question. Answer the question as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer "
        "the question based on the story as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", "
        "or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be answered "
        "based on the information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide "
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
        "following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do "
        "not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}

_GENERIC_PROMPT = (
    "{context}\n\nAnswer the question based on the passages above. Only give the "
    "answer, no explanation.\nQuestion: {input}\nAnswer:"
)


def build_inputs(tokenizer, sample, max_length, device):
    """
    Build model inputs with the official LongBench per-task prompt template.

    The question sits at the END of the prompt. If the prompt exceeds max_length,
    we truncate the CONTEXT from the MIDDLE (keep head and tail token blocks) —
    the LongBench-official strategy — so the trailing instruction + question
    always survive, unlike default right-truncation which drops them.

    Uses the tokenizer's chat template for Instruct models.
    """
    context = sample["context"]
    question = sample["input"]
    template = DATASET2PROMPT.get(sample.get("task"), _GENERIC_PROMPT)

    # Middle-truncate the CONTEXT only, so template instructions never get cut.
    fixed = template.format(context="", input=question)
    fixed_ids = tokenizer(fixed, add_special_tokens=False)["input_ids"]
    use_chat = getattr(tokenizer, "chat_template", None) is not None
    reserve = (64 if use_chat else 0) + len(fixed_ids)
    ctx_budget = max(1, max_length - reserve)

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(ctx_ids) > ctx_budget:
        half = ctx_budget // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[-(ctx_budget - half):]
        context = tokenizer.decode(ctx_ids, skip_special_tokens=True)

    user_msg = template.format(context=context, input=question)

    use_chat = getattr(tokenizer, "chat_template", None) is not None
    reserve = 64 if use_chat else 0  # headroom for chat-template special tokens
    budget = max(1, max_length - reserve)

    ids = tokenizer(user_msg, add_special_tokens=False)["input_ids"]
    if len(ids) > budget:
        half = budget // 2
        ids = ids[:half] + ids[-(budget - half):]  # keep head + tail, drop middle
        user_msg = tokenizer.decode(ids, skip_special_tokens=True)

    if use_chat:
        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    else:
        inputs = tokenizer(user_msg, return_tensors="pt")

    return {k: v.to(device) for k, v in inputs.items()}


def measure_cache(cache):
    """Measure actual retained cache state after generation.

    Returns per-layer lengths (so we see PyramidKV's uneven layers and can
    confirm the synchronized caches are uniform), the resident cache-tensor
    bytes (real memory the KV occupies now, post-eviction), and the layer-0
    sequence length for backward compatibility.
    """
    if cache is None or not hasattr(cache, "key_cache"):
        return {"final_cache_len": None, "layer_lengths": None, "cache_bytes": None}
    try:
        layer_lengths = [int(cache.key_cache[i].shape[2]) for i in range(len(cache.key_cache))]
        cache_bytes = 0
        for i in range(len(cache.key_cache)):
            cache_bytes += cache.key_cache[i].element_size() * cache.key_cache[i].nelement()
            cache_bytes += cache.value_cache[i].element_size() * cache.value_cache[i].nelement()
        return {
            "final_cache_len": layer_lengths[0] if layer_lengths else None,
            "layer_lengths": layer_lengths,
            "cache_bytes": int(cache_bytes),
        }
    except Exception:
        return {"final_cache_len": None, "layer_lengths": None, "cache_bytes": None}


def generate_with_eviction(model, input_ids, cache, max_new_tokens, eos_id):
    """
    Greedy decode loop that supports physical KV eviction.

    HF's `model.generate` tracks a monotonic absolute position counter that
    assumes the cache only grows. When an eviction cache physically drops
    tokens, the KV length shrinks but the counter keeps climbing, so RoPE is
    indexed out of range (index 701 into a 257-long table). The published fix
    (StreamingLLM) is to RE-BASE positions to the physical cache length: after
    eviction, the retained keys occupy positions [0, L), so the next query sits
    at position L = current physical cache length. We drive position_ids and
    cache_position from the cache length each step rather than a running count.

    Returns the generated token ids (excluding the prompt).
    """
    device = input_ids.device
    generated = []

    # Prefill: positions are the natural 0..S-1 (cache is empty going in).
    seq_len = input_ids.shape[1]
    cache_position = torch.arange(seq_len, device=device)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=cache_position)
    next_tok = out.logits[:, -1, :].argmax(dim=-1)
    generated.append(next_tok.item())

    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        # Physical cache length AFTER any eviction that fired last step. The new
        # token attends over [0, L) retained keys and itself sits at position L.
        # This position-rebasing assumes ALL layers share one length. Policies
        # with per-layer uneven lengths (PyramidKV) have no single position_ids
        # valid for every layer in one forward — each layer sizes its RoPE table
        # to its own kv length, so a shared position OOBs the shorter layers.
        # Such methods require attention-level patching (as the official
        # PyramidKV repo does) and aren't supported by this shared-position loop.
        layer_lens = {kc.shape[2] for kc in cache.key_cache}
        if len(layer_lens) > 1:
            raise NotImplementedError(
                "generate_with_eviction requires uniform per-layer cache lengths; "
                f"got {sorted(layer_lens)}. Per-layer-uneven policies (PyramidKV) "
                "need attention-level position patching for real generation."
            )
        L = cache.get_seq_length()
        pos = torch.tensor([[L]], device=device)
        cache_position = torch.tensor([L], device=device)
        out = model(input_ids=next_tok.unsqueeze(0), past_key_values=cache,
                    use_cache=True, position_ids=pos, cache_position=cache_position)
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_tok.item())

    return generated


def evaluate_sample(model, tokenizer, sample, cache, max_new_tokens=128,
                    max_input_length=7900, capture_attention=False):
    """Run generation on a single sample with the given cache policy."""
    from applications.kv_cache.attention_capture import AttentionCapture

    inputs = build_inputs(tokenizer, sample, max_input_length, model.device)
    input_len = inputs["input_ids"].shape[1]

    # Track peak GPU memory for this sample
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)

    eos_id = tokenizer.eos_token_id

    start_time = time.perf_counter()
    with torch.no_grad():
        if cache is None:
            # No eviction (full-cache baseline): plain HF generate is correct.
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
            gen_ids = outputs[0][input_len:].tolist()
        elif capture_attention:
            # Eviction + attention signal: custom decode loop (position re-basing
            # for physical eviction) with attention hooks streaming into cache.
            with AttentionCapture(model, cache):
                gen_ids = generate_with_eviction(
                    model, inputs["input_ids"], cache, max_new_tokens, eos_id)
        else:
            # Eviction without attention (window/random/snapkv-keynorm/qore-vqc).
            gen_ids = generate_with_eviction(
                model, inputs["input_ids"], cache, max_new_tokens, eos_id)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    peak_mem_MB = None
    resident_mem_MB = None
    if torch.cuda.is_available():
        peak_mem_MB = torch.cuda.max_memory_allocated(model.device) / (1024 * 1024)
        # Currently-allocated (post-eviction) memory — reflects compression.
        resident_mem_MB = torch.cuda.memory_allocated(model.device) / (1024 * 1024)

    output_len = len(gen_ids)
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    cache_stats = measure_cache(cache)

    return {
        "prediction": answer,
        "input_length": input_len,
        "output_length": output_len,
        "time_ms": elapsed_ms,
        "tokens_per_sec": output_len / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
        "final_cache_len": cache_stats["final_cache_len"],
        "layer_lengths": cache_stats["layer_lengths"],
        "cache_bytes": cache_stats["cache_bytes"],
        "peak_mem_MB": peak_mem_MB,
        "resident_mem_MB": resident_mem_MB,
    }


def compute_metrics(predictions, references, tasks=None):
    """Compute F1 for LongBench (micro + official macro-by-task if tasks given)."""
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

    result = {
        "f1": float(np.mean(f1_scores)),          # micro-average (all samples pooled)
        "f1_micro": float(np.mean(f1_scores)),
    }

    # Macro-average: mean per-task F1, then mean across tasks. This is how
    # official LongBench reports — it weights each task equally regardless of
    # how many samples it contributed, so a task-skewed sample mix doesn't bias
    # the headline number.
    if tasks is not None:
        per_task = {}
        for score, task in zip(f1_scores, tasks):
            per_task.setdefault(task, []).append(score)
        task_f1 = {t: float(np.mean(s)) for t, s in per_task.items()}
        result["f1_per_task"] = task_f1
        result["f1_macro"] = float(np.mean(list(task_f1.values())))
        result["f1"] = result["f1_macro"]  # headline = macro (official convention)

    return result


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

    # Attention-based policies need eager attention (only impl that exposes the
    # weight matrices the hooks read). Non-attention policies use the faster
    # default so their latency isn't penalized by eager. Record what we actually
    # loaded so the results file is truthful.
    capture = policy_needs_attention(args.policy, args.quality)
    attn_impl = "eager" if capture else "sdpa"

    if not args.skip_generation:
        model, tokenizer = load_llm(args.model_path, attn_implementation=attn_impl)
        num_layers = get_model_num_layers(model)
        kv_bytes_per_token = get_kv_bytes_per_token(model)
        print(f"  Model: {num_layers} layers, {kv_bytes_per_token} bytes/token in KV cache")

    # --- Evaluation loop ---
    print(f"\nRunning evaluation: policy={args.policy}, capacity={args.max_capacity}")
    predictions = []
    references = []
    tasks_list = []
    latencies = []
    throughputs = []
    input_lengths = []
    final_cache_lens = []
    peak_mems = []
    resident_mems = []
    cache_bytes_list = []
    per_sample = []  # full per-sample records for later inspection

    # capture computed above (drives attn_implementation). Announce it here.
    if capture:
        print(f"  Attention capture: ON (policy={args.policy} needs cumulative attention)")

    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}]")

        # Create a FRESH cache for each sample (avoid state leaking between samples)
        cache = create_cache(args.policy, args, num_layers)

        if model is not None:
            result = evaluate_sample(model, tokenizer, sample, cache,
                                     args.max_new_tokens, args.max_input_length,
                                     capture_attention=capture)
            predictions.append(result["prediction"])
            latencies.append(result["time_ms"])
            throughputs.append(result["tokens_per_sec"])
            input_lengths.append(result["input_length"])
            if result["final_cache_len"] is not None:
                final_cache_lens.append(result["final_cache_len"])
            if result["peak_mem_MB"] is not None:
                peak_mems.append(result["peak_mem_MB"])
            if result.get("resident_mem_MB") is not None:
                resident_mems.append(result["resident_mem_MB"])
            if result.get("cache_bytes") is not None:
                cache_bytes_list.append(result["cache_bytes"])
            per_sample.append({
                "task": sample.get("task"),
                "question": sample["input"],
                "prediction": result["prediction"],
                "references": sample["answers"],
                "input_length": result["input_length"],
                "final_cache_len": result["final_cache_len"],
                "layer_lengths": result.get("layer_lengths"),
                "cache_bytes": result.get("cache_bytes"),
                "peak_mem_MB": round(result["peak_mem_MB"], 1) if result["peak_mem_MB"] else None,
                "resident_mem_MB": round(result["resident_mem_MB"], 1) if result.get("resident_mem_MB") else None,
            })
        else:
            predictions.append("")
            input_lengths.append(0)

        references.append(sample["answers"])
        tasks_list.append(sample.get("task"))

    # --- Compute metrics ---
    metrics = {}
    if not args.skip_generation:
        metrics = compute_metrics(predictions, references, tasks=tasks_list)
        metrics["avg_latency_ms"] = float(np.mean(latencies))
        metrics["avg_throughput_tok_per_sec"] = float(np.mean(throughputs))
        print(f"\n  Results: F1={metrics['f1']:.4f}, "
              f"Latency={metrics['avg_latency_ms']:.0f}ms, "
              f"Throughput={metrics['avg_throughput_tok_per_sec']:.1f} tok/s")

    # --- Compression metrics (theoretical, from capacity) ---
    avg_input_len = float(np.mean(input_lengths)) if input_lengths and input_lengths[0] > 0 else 4096
    effective_capacity = args.max_capacity if args.policy != "full" else avg_input_len
    compression_ratio = effective_capacity / avg_input_len if avg_input_len > 0 else 1.0

    memory_full_MB = avg_input_len * kv_bytes_per_token / (1024 * 1024)
    memory_compressed_MB = effective_capacity * kv_bytes_per_token / (1024 * 1024)
    memory_saved_MB = memory_full_MB - memory_compressed_MB

    # --- Measured stats (actual retained cache + GPU memory) ---
    # avg_cache_bytes_MB is the REAL resident KV footprint after eviction —
    # unlike avg_peak_mem_MB (historical prefill peak) it reflects the actual
    # compression achieved, and unlike the theoretical figure it counts what the
    # tensors truly occupy (capturing PyramidKV's uneven per-layer lengths).
    avg_cache_bytes_MB = (round(float(np.mean(cache_bytes_list)) / (1024 * 1024), 2)
                          if cache_bytes_list else None)
    measured = {
        "avg_final_cache_len": round(float(np.mean(final_cache_lens)), 1) if final_cache_lens else None,
        "avg_peak_mem_MB": round(float(np.mean(peak_mems)), 1) if peak_mems else None,
        "max_peak_mem_MB": round(float(np.max(peak_mems)), 1) if peak_mems else None,
        "avg_resident_mem_MB": round(float(np.mean(resident_mems)), 1) if resident_mems else None,
        "avg_cache_MB": avg_cache_bytes_MB,
    }

    # --- Task distribution (sanity check for stratified sampling) ---
    task_counts = {}
    for s in samples:
        t = s.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1

    # --- Save results ---
    result = {
        "experiment": f"KV-Cache-{args.dataset}",
        "policy": args.policy,
        "model": args.model_path,
        "dataset": args.dataset,
        "num_samples": len(samples),
        "task_distribution": task_counts,
        "metrics": metrics,
        "compression": {
            "avg_input_length": round(avg_input_len),
            "tokens_kept": int(effective_capacity),
            "compression_ratio": round(compression_ratio, 4),
            "memory_full_MB": round(memory_full_MB, 1),
            "memory_compressed_MB": round(memory_compressed_MB, 1),
            "memory_saved_MB": round(memory_saved_MB, 1),
        },
        "measured": measured,
        "config": {
            "max_capacity": args.max_capacity,
            "trigger_every": args.trigger_every,
            "num_sink_tokens": args.num_sink_tokens,
            "num_reads": args.num_reads,
            "quality": args.quality,
            "window": args.window,
            "solver_method": args.solver_method,
            "redundancy_method": args.redundancy_method,
            "quantum_backend": args.quantum_backend,
            "attn_implementation": attn_impl,
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "seed": args.seed,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Write per-sample records (predictions, references, cache lengths) alongside.
    if per_sample:
        detail_path = output_path.with_suffix(".samples.json")
        with open(detail_path, "w") as f:
            json.dump(per_sample, f, indent=2, ensure_ascii=False)
        print(f"  Per-sample details saved to {detail_path}")

    print(f"\nResults saved to {output_path}")
    print(f"  Task distribution: {task_counts}")
    if measured["avg_peak_mem_MB"]:
        print(f"  Measured peak GPU mem: avg {measured['avg_peak_mem_MB']:.0f} MB, "
              f"max {measured['max_peak_mem_MB']:.0f} MB")
    print(f"Compression (theoretical): {avg_input_len:.0f} → {effective_capacity} tokens "
          f"({compression_ratio:.1%} kept, saves ~{memory_saved_MB:.0f} MB)")


if __name__ == "__main__":
    main()
