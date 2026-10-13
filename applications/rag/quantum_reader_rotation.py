"""Opt-in real-unitary Reader feature adapter, never a post-logit residual.

Little-endian amplitude coordinates; RY/CRY use half-angle convention. The
classical arm uses seeded coordinate matchings with the same angle budget and
number of active pairs. Both are exactly classically simulated orthogonal maps.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RotationConfig:
    layers: int = 2
    conditioning: int = 4
    seed: int = 20261013


def make_rotation_head(weight, bias, *, family="quantum", config=None):
    import torch
    cfg = config or RotationConfig()
    if family not in ("quantum", "classical"):
        raise ValueError("unknown rotation family")
    if cfg.layers not in (1, 2) or cfg.conditioning != 4:
        raise ValueError("only frozen one/two-layer, four-condition contracts supported")
    w = torch.as_tensor(weight).detach().reshape(-1).clone()
    b = torch.as_tensor(bias, dtype=w.dtype, device=w.device).detach().reshape(()).clone()
    if w.numel() < 2 or not w.is_floating_point() or not bool(w.isfinite().all() & b.isfinite()):
        raise ValueError("finite floating classifier required")
    hidden = w.numel()
    qubits = math.ceil(math.log2(hidden))
    size = 1 << qubits
    count = 2 * qubits * cfg.layers
    # Fixed row-normalized projection of RMS-normalized pooled input; bounded
    # algebraic compression avoids tiny 1/sqrt(3840) signals and tanh clipping.
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    projection = torch.randn(cfg.conditioning, 5 * hidden, generator=generator, dtype=w.dtype)
    projection = torch.nn.functional.normalize(projection, dim=1)
    matching_generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    schedules = []
    descriptors = []
    indices = torch.arange(size)
    for layer in range(cfg.layers):
        for controlled in (False, True):
            for target in range(qubits):
                control = (target + 1) % qubits if controlled else None
                if qubits == 1 and controlled:
                    raise ValueError("controlled circuit needs at least two qubits")
                low = indices[(indices & (1 << target)) == 0]
                if controlled:
                    low = low[(low & (1 << control)) != 0]
                high = low | (1 << target)
                descriptors.append((layer, control, target))
                if family == "classical":
                    permutation = torch.randperm(size, generator=matching_generator)
                    low, high = permutation[:len(low)], permutation[len(low):2 * len(low)]
                schedules.append((low, high))

    class RotationHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.family, self.config = family, cfg
            self.hidden_size, self.qubits, self.state_size = hidden, qubits, size
            self.descriptors = tuple(descriptors)
            self.theta = torch.nn.Parameter(torch.zeros(count, cfg.conditioning + 1, dtype=w.dtype, device=w.device))
            self.register_buffer("weight", w)
            self.register_buffer("bias", b)
            self.register_buffer("projection", projection.to(w.device))
            for gate, (low, high) in enumerate(schedules):
                self.register_buffer(f"low_{gate}", low.to(w.device))
                self.register_buffer(f"high_{gate}", high.to(w.device))

        def forward(self, cls, pooled):
            if cls.ndim != 2 or cls.shape[1] != hidden or pooled.shape != (len(cls), 5 * hidden):
                raise ValueError("CLS/pooled shape mismatch")
            if cls.dtype != self.theta.dtype or pooled.dtype != cls.dtype or cls.device != self.theta.device or pooled.device != cls.device:
                raise ValueError("explicit dtype/device agreement required")
            if not bool(cls.isfinite().all() & pooled.isfinite().all() & self.theta.isfinite().all()):
                raise ValueError("nonfinite rotation input/parameter")
            norm = cls.norm(dim=-1, keepdim=True)
            if bool((norm == 0).any()):
                raise ValueError("zero CLS has no amplitude encoding")
            projected = math.sqrt(5 * hidden) * (torch.nn.functional.normalize(pooled, dim=-1) @ self.projection.T)
            conditioning = projected / torch.sqrt(1 + projected.square())
            angles = torch.cat((torch.ones_like(conditioning[:, :1]), conditioning), dim=1) @ self.theta.T
            padded = torch.nn.functional.pad(cls, (0, size - hidden))
            # Rotate unnormalized coordinates to retain an exact identity feature
            # path. Dividing by norm gives the identical normalized quantum state.
            rotated = padded
            states = [padded / norm]
            for gate in range(count):
                low, high = getattr(self, f"low_{gate}"), getattr(self, f"high_{gate}")
                a, z = rotated[:, low], rotated[:, high]
                cosine, sine = torch.cos(angles[:, gate:gate+1] / 2), torch.sin(angles[:, gate:gate+1] / 2)
                nxt = rotated.clone()
                nxt[:, low], nxt[:, high] = cosine * a - sine * z, sine * a + cosine * z
                rotated = nxt
                states.append(rotated / norm)
            scores = torch.nn.functional.linear(rotated[:, :hidden], self.weight[None], self.bias[None]).squeeze(-1)
            return {"scores": scores, "rotated": rotated, "conditioning": conditioning,
                    "angles": angles, "states": torch.stack(states, dim=1), "norm": norm,
                    "padded": padded}

        def resources(self):
            return {"data_qubits": qubits, "layers": cfg.layers, "conditional_angles": count,
                    "trainable_parameters": self.theta.numel(), "family": family,
                    "readout": "signed real overlap, never squared fidelity",
                    "simulation": "exact real Torch statevector", "quantum_advantage": "unestablished",
                    "hardware_obligations": ["state preparation", "conditioning preparation", "controlled readout and ancilla", "finite shots"]}

    return RotationHead()


@contextmanager
def reader_rotation_hook(reader, head, question_mask, passage_mask, capture):
    """Scope the intervention to qa_classifier input; span heads stay untouched.

    Call Reader normally inside this context (without no_grad during adapter
    training). Reader parameters must already be frozen. Hooks always removed.
    No target/reference/generation fields enter this API.
    """
    from applications.rag.quantum_semantic_reader import semantic_pool
    if any(p.requires_grad for p in reader.parameters()) or reader.training:
        raise ValueError("Reader must be frozen and in evaluation mode")
    classifier = reader.span_predictor.qa_classifier
    import torch
    if not torch.equal(classifier.weight.detach().reshape(-1), head.weight) or not torch.equal(classifier.bias.detach().reshape(()), head.bias):
        raise ValueError("adapter/classifier identity mismatch")
    context = {}
    def encoded(_module, _inputs, output):
        hidden = output[0]
        pooled, weights = semantic_pool(hidden, question_mask, passage_mask)
        context.update(hidden=hidden, pooled=pooled, pooling_weights=weights)
    def before_classifier(_module, inputs):
        if "pooled" not in context or len(inputs) != 1:
            raise ValueError("encoder/classifier hook order mismatch")
        result = head(inputs[0], context["pooled"])
        capture.clear()
        capture.update(context, cls=inputs[0], **result)
        return (result["rotated"][:, :head.hidden_size],)
    handles = []
    try:
        handles.append(reader.span_predictor.encoder.register_forward_hook(encoded))
        handles.append(classifier.register_forward_pre_hook(before_classifier))
        yield
    finally:
        for handle in handles:
            handle.remove()


def top5_retention_objective(scores, base, row, *, margin=0.0, anchor_weight=0.01, protection_weight=1.0):
    """Protect membership, permitting a rescued positive to outrank kept items.

    Continuous tie surrogate only; actual stable Top5 must also be audited.
    Labels are accepted only here, never by the forward/hook/selection paths.
    """
    import torch
    from applications.rag.protected_boundary_learning import validate_row
    validate_row(row)
    if scores.shape != (50,) or base.shape != (50,) or not bool(scores.isfinite().all() & base.isfinite().all()):
        raise ValueError("finite Top50 scores required")
    if any(not math.isfinite(x) or x < 0 for x in (margin, anchor_weight, protection_weight)):
        raise ValueError("nonnegative finite loss settings required")
    zero = scores.sum() * 0
    if not row["draft_pairs"]:
        return dict(total=zero, pair=zero, anchor=zero, protection=zero)
    pairs = torch.as_tensor(row["draft_pairs"], device=scores.device)
    weights = torch.as_tensor(row["pair_weights"], device=scores.device, dtype=scores.dtype)
    pair = (torch.nn.functional.softplus(scores[pairs[:, 1]] - scores[pairs[:, 0]] + margin) * weights).sum()
    anchor = (scores - base.detach().to(scores)).square().mean()
    penalties = []
    for kept in row["immutable_selected_indices"]:
        others = torch.cat((scores[:kept], scores[kept+1:]))
        threshold = torch.topk(others, 5).values[-1]
        penalties.append(torch.relu(threshold - scores[kept]).square())
    protection = torch.stack(penalties).mean() if penalties else zero
    return {"total": pair + anchor_weight * anchor + protection_weight * protection,
            "pair": pair, "anchor": anchor, "protection": protection}


def stable_top5(scores):
    """Candidate order is frozen retrieval rank; labels cannot restrict the pool."""
    values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
    if len(values) != 50 or not all(math.isfinite(v) for v in values):
        raise ValueError("finite Top50 required")
    return sorted(range(50), key=lambda i: (-values[i], i))[:5]
