"""Question-conditioned semantic residual head; frozen Reader body, trainable head."""

from __future__ import annotations

from typing import Any

from applications.rag.qarcg_reader import QARCGConfig, make_torch_classical_control, make_torch_qarcg


def semantic_pool(hidden: Any, question_mask: Any, passage_mask: Any) -> tuple[Any, Any]:
    """Pool joint Reader representations without accepting target/label fields."""
    import torch
    if hidden.ndim != 3 or question_mask.shape != hidden.shape[:2] or passage_mask.shape != hidden.shape[:2]:
        raise ValueError("expected N x L x H hidden states and N x L masks")
    if question_mask.dtype != torch.bool or passage_mask.dtype != torch.bool:
        raise ValueError("token masks must be boolean")
    if bool((question_mask & passage_mask).any()) or not bool(question_mask.any(1).all() & passage_mask.any(1).all()):
        raise ValueError("question/passage masks must be disjoint and nonempty")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("non-finite Reader hidden states")
    q = (hidden * question_mask.unsqueeze(-1)).sum(1) / question_mask.sum(1, keepdim=True)
    query = torch.nn.functional.normalize(q, dim=-1)
    tokens = torch.nn.functional.normalize(hidden, dim=-1)
    # Question-conditioned pooling, not span confidence or target-conditioned attention.
    weights = torch.softmax((tokens * query[:, None, :]).sum(-1).masked_fill(~passage_mask, -torch.inf), dim=1)
    p = (hidden * weights.unsqueeze(-1)).sum(1)
    pooled = torch.cat((hidden[:, 0], q, p, q * p, torch.abs(q - p)), dim=1)
    return pooled, weights


def reader_token_masks(encoded: Any, tokenizer: Any) -> tuple[Any, Any]:
    """Recover disjoint question/passage token masks from a DPR Reader batch.

    Transformers' DPR Reader tokenizers do not reliably emit ``token_type_ids``
    (the slow tokenizer never does), so fall back to the first ``[SEP]``
    boundary. For a question/text pair the first segment holds the question and
    everything after the first ``[SEP]`` belongs to the passage.
    """
    import torch
    if "input_ids" not in encoded or "attention_mask" not in encoded:
        raise ValueError("Reader encoding is missing input_ids/attention_mask")
    input_ids = encoded["input_ids"]
    valid = encoded["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", ()) or ():
        valid = valid & (input_ids != int(token_id))
    token_types = encoded.get("token_type_ids")
    if token_types is not None:
        return valid & (token_types == 0), valid & (token_types == 1)
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    fallback = max(1, length // 2)
    if sep_id is not None:
        is_sep = input_ids == int(sep_id)
        first_sep = is_sep.float().argmax(1)
        split = torch.where(is_sep.any(1), first_sep, torch.full_like(first_sep, fallback))
    else:
        split = torch.full((input_ids.shape[0],), fallback, device=input_ids.device, dtype=torch.long)
    question = valid & (positions[None, :] < split[:, None])
    passage = valid & (positions[None, :] > split[:, None])
    return question, passage


def make_semantic_head(hidden_size: int, *, classical: bool = False, seed: int = 20261008,
                       config: QARCGConfig | None = None) -> Any:
    """Both arms share semantic input/projection initialization and parameter budget."""
    import torch
    cfg = config or QARCGConfig()
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")

    class SemanticHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = hidden_size
            self.project = torch.nn.Linear(5 * hidden_size, 4)
            self.interaction = make_torch_classical_control(cfg) if classical else make_torch_qarcg(cfg)

        def forward(self, base_scores, pooled):
            if pooled.shape != (base_scores.numel(), 5 * hidden_size) or not bool(torch.isfinite(pooled).all()):
                raise ValueError("invalid semantic pooled representation")
            normalized = torch.nn.functional.layer_norm(pooled, (5 * hidden_size,))
            # acos encoding has infinite derivatives at exactly +/-1.
            encoded = (1.0 - 1e-6) * torch.tanh(self.project(normalized))
            scores, gate, residual, observables = self.interaction(base_scores, encoded)
            return scores, gate, residual, observables, encoded

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SemanticHead()


def reader_semantic_forward(torch, tokenizer, reader, device, question, texts, *,
                            max_length=350, batch_size=8, capture_first=False):
    """Capture the final encoder state through a scoped hook, avoiding all-layer retention."""
    captured = []
    def hook(_module, _inputs, output):
        captured.append(output[0])
    encoder = reader.span_predictor.encoder
    handle = encoder.register_forward_hook(hook)
    bases, pools, span_rows = [], [], []
    sample = None
    try:
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset:offset + batch_size]
            encoded = tokenizer(questions=[question] * len(batch), texts=batch, return_tensors="pt",
                                padding=True, truncation=True, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            qm, pm = reader_token_masks(encoded, tokenizer)
            captured.clear()
            with torch.no_grad():
                output = reader(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], return_dict=True)
                if len(captured) != 1:
                    raise ValueError("Reader encoder hook coverage failure")
                hidden = captured.pop()
                pooled, weights = semantic_pool(hidden, qm, pm)
                from applications.rag.qarcg_reader import torch_reader_features
                _, span = torch_reader_features(output.relevance_logits, output.start_logits,
                                                output.end_logits, pm, max_answer_tokens=10)
            bases.append(output.relevance_logits.detach().cpu())
            pools.append(pooled.detach().cpu())
            for i in range(len(batch)):
                span_rows.append({"relevance_logit": float(output.relevance_logits[i]),
                                  "span_logit": float(span["best_span"][i]),
                                  "span_margin": float(span["span_margin"][i]),
                                  "start_entropy": float(span["start_entropy"][i]),
                                  "end_entropy": float(span["end_entropy"][i])})
            if capture_first and offset == 0:
                sample = {"hidden": hidden[0].detach().cpu().numpy(),
                          "input_ids": encoded["input_ids"][0].cpu().numpy(),
                          "question_mask": qm[0].cpu().numpy(), "passage_mask": pm[0].cpu().numpy(),
                          "pooling_weights": weights[0].cpu().numpy()}
    finally:
        captured.clear()
        handle.remove()
    from applications.rag.qarcg_reader import build_reader_features
    scalar = torch.as_tensor(build_reader_features(span_rows), dtype=torch.float32)
    return torch.cat(bases), torch.cat(pools), scalar, sample


def circuit_resource_contract(depth=2):
    return {"qubits": 4, "depth_blocks": depth, "trainable_circuit_parameters": depth * 8,
            "cnot_gates": depth * 4, "input_ry_gates": 4, "observables": 8,
            "simulation": "exact Torch statevector", "finite_shot": "not_run",
            "quantum_advantage": "not_established", "projection_is_classical": True}
