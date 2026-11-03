"""Q-CLS-ATTN v004 source preparation; numerical execution is not certified.

This module has no model loader, data reader, CLI, global patch or backend
override. Import/use is reserved for a reviewed, registered shared-queue recipe.
The caller supplies audited native inputs. Labels never enter online features.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONTRACT_SHA256 = "8edae2d632258e5fd7c649cdbf5b7ccf0e3b6e522f414c2cb2488503e5420035"
MODEL_REVISION = "ab1974d6023bc4dd27cd713c9ac2b6247509244d"
ARMS = ("MLP24", "PRODUCT24_nominal", "TRIG24", "QCLS24", "HEAD768_active")
UPSTREAM_ARMS = ARMS[:4]
AMPLITUDE = 0.25
SEED = 20261029


class QCLSContractError(RuntimeError):
    """Fatal input, lifecycle or numerical contract violation; never recover."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QCLSContractError(message)


def finite(value: Tensor, name: str) -> None:
    require(bool(torch.isfinite(value).all()), f"nonfinite {name}")


def require_eval(module: nn.Module, name: str) -> None:
    require(all(not child.training for child in module.modules()), f"non-eval child in {name}")


def tensor_digest(value: Tensor) -> str:
    """Detached integrity boundary only; never construct a bias through this."""
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}".encode())
    raw = value.reshape(-1).view(torch.uint8)
    # Identical CPU uint8 bytes, without boxing one Python integer per byte.
    digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def module_digest(module: nn.Module) -> dict[str, str]:
    return {name: tensor_digest(value) for name, value in module.state_dict().items()}


@dataclass(frozen=True)
class TokenBoundary:
    """Native padded tokens plus independently bound audited membership.

    audit_ids must be supplied from the frozen tokenizer audit, not recomputed
    from answers. membership: -1=special/pad, 0=question, 1=passage.
    Provider/source/role identities are a separate mandatory runner gate.
    """

    input_ids: Tensor
    attention_mask: Tensor
    audit_ids: Tensor
    membership: Tensor
    special_tokens_mask: Tensor
    cls_token_id: int

    def masks(self) -> tuple[Tensor, Tensor]:
        ids, attention = self.input_ids, self.attention_mask
        require(ids.ndim == 2 and 0 < ids.shape[0] <= 8 and 0 < ids.shape[1] <= 350,
                "native chunk must have 1..8 rows and 1..350 padded positions")
        require(ids.dtype == self.audit_ids.dtype == torch.int64 and not ids.requires_grad,
                "token IDs and their audit must be int64")
        require(torch.equal(ids, self.audit_ids), "native/audited token mismatch")
        for value in (attention, self.audit_ids, self.membership, self.special_tokens_mask):
            require(value.shape == ids.shape and value.device == ids.device, "token metadata shape/device")
        require(self.membership.dtype in (torch.int8, torch.int16, torch.int32, torch.int64),
                "membership must be an audited integer sequence index")
        require(self.special_tokens_mask.dtype == torch.bool, "special mask must be explicit bool")
        require(bool(((attention == 0) | (attention == 1)).all()), "nonbinary attention mask")
        valid = attention.bool()
        require(bool(valid[:, 0].all()) and bool((ids[:, 0] == self.cls_token_id).all()), "CLS identity")
        require(bool(self.special_tokens_mask[:, 0].all()), "CLS must be special")
        require(not bool(((~valid[:, :-1]) & valid[:, 1:]).any()), "only original right padding supported")
        require(bool(((self.membership == -1) | (self.membership == 0) |
                      (self.membership == 1)).all()), "unexpected membership role")
        excluded = (~valid) | self.special_tokens_mask
        require(bool((self.membership[excluded] == -1).all()), "special/pad membership leak")
        require(bool((self.membership[~excluded] >= 0).all()), "unassigned original visible token")
        q = (self.membership == 0) & ~excluded
        p = (self.membership == 1) & ~excluded
        require(bool(q.any(-1).all()) and bool(p.any(-1).all()), "empty question/passage")
        require(not bool((q & p).any()), "question/passage overlap")
        return q, p


def normalized_inputs(hidden: Tensor, boundary: TokenBoundary) -> tuple[Tensor, Tensor, Tensor]:
    qmask, pmask = boundary.masks()
    require(hidden.shape == (*qmask.shape, 768) and hidden.dtype == torch.float32,
            "final input must be native FP32 [B,L,768]")
    require(hidden.device == qmask.device, "hidden/token device mismatch")
    hidden = hidden.detach()
    finite(hidden, "native hidden")
    qpool = (hidden * qmask.unsqueeze(-1)).sum(1) / qmask.sum(1, keepdim=True)
    qpool = qpool / qpool.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    passage = hidden[pmask]
    passage = passage / passage.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    row = pmask.nonzero(as_tuple=True)[0]
    return qpool, qpool.index_select(0, row), passage


class Affine(nn.Module):
    """No implicit random Linear initialization: draw order belongs to v004."""

    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.bias = nn.Parameter(bias.clone())

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(value, self.weight, self.bias)


def _rotation(angle: Tensor, axis: str) -> Tensor:
    c, s = torch.cos(angle / 2).to(torch.complex64), torch.sin(angle / 2).to(torch.complex64)
    if axis == "Y":
        return torch.stack((c, -s, s, c), -1).reshape(-1, 2, 2)
    if axis == "Z":
        z = torch.zeros_like(c)
        return torch.stack((c - 1j*s, z, z, c + 1j*s), -1).reshape(-1, 2, 2)
    require(axis == "X", "unsupported circuit gate")
    return torch.stack((c, -1j*s, -1j*s, c), -1).reshape(-1, 2, 2)


def _single(state: Tensor, angle: Tensor, axis: str, qubit: int) -> Tensor:
    # Index i=8*b0+4*b1+2*b2+b3, q0 most significant. No state renormalization.
    moved = state.reshape(-1, 2, 2, 2, 2).movedim(qubit + 1, -1)
    output = torch.einsum("bij,bkj->bki", _rotation(angle, axis), moved.reshape(-1, 8, 2))
    return output.reshape(moved.shape).movedim(-1, qubit + 1).reshape(-1, 16)


class Circuit24(nn.Module):
    def __init__(self, angles: Tensor, *, product: bool):
        super().__init__()
        self.angles = nn.Parameter(angles.clone())
        self.product = product

    def forward(self, x: Tensor) -> Tensor:
        require(x.ndim == 2 and x.shape[1] == 4 and x.dtype == torch.float32, "circuit input")
        state = torch.zeros((len(x), 16), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1
        basis = torch.arange(16, device=x.device)
        for q in range(4):
            state = _single(state, x[:, q], "Y", q)
        for block in range(2):
            if block == 1:
                for q in range(4):
                    state = _single(state, x[:, q], "Z", q)
            theta = self.angles[block]
            for q in range(4):
                state = _single(state, theta[2*q].expand(len(x)), "Y", q)
                state = _single(state, theta[2*q+1].expand(len(x)), "Z", q)
            for gate, (control, target) in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
                angle = theta[8+gate].expand(len(x))
                if self.product:
                    state = _single(state, angle, "X", control)
                else:
                    rotated = _single(state, angle, "X", target)
                    on = (((basis >> (3-control)) & 1) == 1).unsqueeze(0)
                    state = torch.where(on, rotated, state)
        probabilities = state.abs().square()
        finite(probabilities, "state probabilities")
        require(bool((probabilities.sum(-1) - 1).abs().le(1e-5).all()), "state norm failure")
        bit = lambda q: (basis >> (3-q)) & 1
        observable = ((1-2*(bit(0) ^ bit(2))) + (1-2*(bit(1) ^ bit(3)))) / 2
        value = (probabilities * observable).sum(-1)
        finite(value, "circuit observable")
        require(bool(value.abs().le(1+1e-5).all()), "observable range")
        return value


class MLP24(nn.Module):
    def __init__(self, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor):
        super().__init__()
        self.first, self.second = Affine(w1, b1), Affine(w2, b2)
        self.mix = nn.Parameter(torch.zeros((), dtype=torch.float32, device="cpu"))

    def forward(self, x: Tensor) -> Tensor:
        pair = torch.tanh(self.second(torch.tanh(self.first(x))))
        mix = torch.sigmoid(self.mix)
        return mix*pair[:, 0] + (1-mix)*pair[:, 1]


class TRIG24(nn.Module):
    def __init__(self, w: Tensor, b: Tensor, v: Tensor):
        super().__init__()
        self.weight, self.bias, self.output = nn.Parameter(w.clone()), nn.Parameter(b.clone()), nn.Parameter(v.clone())

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh((torch.sin(F.linear(x, self.weight, self.bias))*self.output).sum(-1))


class ProjectedCritic(nn.Module):
    def __init__(self, query: Affine, passage: Affine, critic: nn.Module):
        super().__init__()
        self.query, self.passage, self.critic = copy.deepcopy(query), copy.deepcopy(passage), critic

    def forward(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        angles = math.pi * torch.tanh(torch.cat((self.query(q), self.passage(p)), -1))
        return self.critic(angles), angles


@dataclass
class DeltaValues:
    delta: Tensor
    contextual_question_pool: Tensor
    angles: Tensor
    reference_angles: Tensor
    observation: Tensor
    reference_observation: Tensor


class AnchoredBias(nn.Module):
    def __init__(self, learned: ProjectedCritic):
        super().__init__()
        self.learned = learned
        self.reference = copy.deepcopy(learned).eval().requires_grad_(False)
        self.reference_digest = module_digest(self.reference)
        live_ptrs = {p.data_ptr() for p in self.learned.parameters()}
        require(all(p.data_ptr() not in live_ptrs for p in self.reference.parameters()), "reference storage alias")
        self.eval()

    def verify_reference(self) -> None:
        require_eval(self, "learned and frozen-reference trees")
        require(not any(p.requires_grad for p in self.reference.parameters()), "reference unfrozen")
        require(module_digest(self.reference) == self.reference_digest, "reference changed")

    def forward(self, hidden: Tensor, boundary: TokenBoundary) -> DeltaValues:
        qpool, q, p = normalized_inputs(hidden, boundary)
        obs, angles = self.learned(q, p)
        with torch.no_grad():
            ref, reference_angles = self.reference(q, p)
        values = AMPLITUDE * (obs - ref)
        finite(values, "attention correction")
        require(bool(values.abs().le(0.5+1e-5).all()), "fixed correction range exceeded")
        _, pmask = boundary.masks()
        delta = torch.zeros_like(boundary.attention_mask, dtype=torch.float32).masked_scatter(pmask, values)
        return DeltaValues(delta, qpool, angles, reference_angles, obs, ref)


class Head768(nn.Module):
    def __init__(self, native_head: nn.Module):
        super().__init__()
        require(native_head.weight.shape == (1, 768) and native_head.bias.shape == (1,), "native head identity")
        self.weight = nn.Parameter(native_head.weight.detach().clone())
        self.register_buffer("bias", native_head.bias.detach().clone())

    def forward(self, cls: Tensor) -> Tensor:
        return F.linear(cls.detach(), self.weight, self.bias).squeeze(-1)


def initialize_arms(native_head: nn.Module, *, seed: int = SEED) -> tuple[nn.ModuleDict, dict[str, str]]:
    """One explicit CPU generator; caller moves completed arms to its device."""
    require(type(seed) is int and 0 <= seed < 2**63, "explicit initialization seed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def uniform(shape: tuple[int, ...], low: float, high: float) -> Tensor:
        return torch.empty(shape, dtype=torch.float32, device="cpu").uniform_(low, high, generator=generator)
    def xavier(shape: tuple[int, int]) -> Tensor:
        bound = math.sqrt(6/(shape[0]+shape[1]))
        return uniform(shape, -bound, bound)
    query = Affine(xavier((2, 768)), uniform((2,), -0.01, 0.01))
    passage = Affine(xavier((2, 768)), uniform((2,), -0.01, 0.01))
    theta = uniform((2, 12), -math.pi/4, math.pi/4)
    mlp = MLP24(xavier((3, 4)), uniform((3,), -0.01, 0.01),
                xavier((2, 3)), uniform((2,), -0.01, 0.01))
    trig = TRIG24(uniform((4, 4), -1, 1), uniform((4,), -math.pi, math.pi), uniform((4,), -0.5, 0.5))
    critics = {"MLP24": mlp, "PRODUCT24_nominal": Circuit24(theta, product=True),
               "TRIG24": trig, "QCLS24": Circuit24(theta, product=False)}
    arms = nn.ModuleDict({name: AnchoredBias(ProjectedCritic(query, passage, critics[name])) for name in UPSTREAM_ARMS})
    arms["HEAD768_active"] = Head768(native_head)
    require(tuple(arms) == ARMS, "arm order")
    expected = {**{name: 3100 for name in UPSTREAM_ARMS}, "HEAD768_active": 768}
    for name, arm in arms.items():
        require(sum(p.numel() for p in arm.parameters() if p.requires_grad) == expected[name], "parameter inventory")
        arm.eval()
    owned_storage: set[int] = set()
    for arm in arms.values():
        storage = {p.untyped_storage().data_ptr() for p in arm.parameters()}
        require(not owned_storage.intersection(storage), "parameter storage shared across comparator arms")
        owned_storage.update(storage)
    return arms, {"generator_final_state": tensor_digest(generator.get_state()),
                  **{name: hashlib.sha256(repr(module_digest(arm)).encode()).hexdigest() for name, arm in arms.items()}}


def ordinal_question_loss(scores: Tensor, tiers: Tensor) -> tuple[Tensor, int]:
    """One question; caller keeps all512 questions, including zero-pair ones."""
    require(scores.ndim == 1 and tiers.shape == scores.shape, "ordinal row shape")
    require(tiers.dtype == torch.int64 and tiers.device == scores.device, "ordinal target type/device")
    require(bool(((tiers >= 0) & (tiers <= 2)).all()), "unknown/unreviewed target passed as supervision")
    finite(scores, "ordinal scores")
    higher = tiers[:, None] > tiers[None, :]
    count = int(higher.sum())
    if count == 0:
        return scores.sum()*0, 0
    gaps = scores[:, None] - scores[None, :]
    return F.softplus(-gaps[higher]).sum()/count, count


def _validate_attention(attention: nn.Module) -> inspect.Signature:
    require(type(attention).__module__ == "transformers.models.bert.modeling_bert" and
            type(attention).__name__ == "BertSdpaSelfAttention", "original SDPA class required")
    require(not attention.training and not attention.is_decoder and
            attention.position_embedding_type == "absolute", "unsupported attention configuration")
    require_eval(attention, "native attention")
    require(attention.num_attention_heads == 12 and attention.attention_head_size == 64,
            "pinned native head layout required")
    require(all(not p.requires_grad and p.dtype == torch.float32 for p in attention.parameters()),
            "native attention must be frozen FP32")
    signature = inspect.signature(attention.forward)
    require(tuple(signature.parameters) == (
        "hidden_states", "attention_mask", "head_mask", "encoder_hidden_states",
        "encoder_attention_mask", "past_key_value", "output_attentions"), "pinned signature mismatch")
    return signature


def _hook_ids(module: nn.Module) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(module._forward_pre_hooks), tuple(module._forward_hooks)


class ScopedCLSKeyBias:
    """One native final-attention call; no backend override or global patch.

    `reference_null` requires an exactly initialized learned/reference pair.
    `probe` is restricted to the three frozen mask diagnostics. Only a declared
    preflight trace may set capture=True; release it after backward/export.
    Native/HEAD comparisons call the original model without this context.
    The recipe must separately certify source/role identity and profiler family.
    """

    MODES = ("disabled", "zero", "reference_null", "active", "probe")

    def __init__(self, attention: nn.Module, boundary: TokenBoundary, *, mode: str,
                 arm: AnchoredBias | None = None, probe: str | None = None,
                 capture: bool = False, observation: dict | None = None):
        require(mode in self.MODES, "unregistered intervention mode")
        require((arm is not None) == (mode in ("reference_null", "active")), "arm/mode mismatch")
        require((probe is not None) == (mode == "probe"), "probe/mode mismatch")
        require(probe in (None, "zero", "constant_eighth", "alternating_eighth"), "unregistered probe")
        require(observation is None or type(observation) is dict and not observation,
                "fresh owned layout observation required")
        self.attention, self.boundary, self.mode = attention, boundary, mode
        self.arm, self.probe, self.capture = arm, probe, capture
        # Scalar metadata only. Normal training shape guards must not retain
        # QKV/context/delta graphs merely to describe their actual mask layout.
        self.observation = observation
        self.signature = _validate_attention(attention)
        self.calls = self.outputs = 0
        self.entered = False
        self.handles: list[object] = []
        self.snapshots: dict[nn.Module, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        self.owned: dict[nn.Module, set[int]] = {}
        self.values: DeltaValues | None = None
        self.delta: Tensor | None = None
        self.actual_mask: Tensor | None = None
        self.context: Tensor | None = None
        self.projections: dict[str, Tensor] = {}
        self.input_mask_was_none: bool | None = None

    def _add(self, module: nn.Module, handle: object) -> None:
        self.handles.append(handle)
        self.owned.setdefault(module, set()).add(handle.id)

    def __enter__(self) -> ScopedCLSKeyBias:
        require(not self.entered, "context cannot be reused")
        self.entered = True
        # Global observers could retain values or change outputs: never remove
        # somebody else's hooks to make a candidate appear compatible.
        global_hooks = torch.nn.modules.module
        require(not global_hooks._global_forward_pre_hooks and not global_hooks._global_forward_hooks,
                "foreign global forward hooks")
        modules = [self.attention, self.attention.query, self.attention.key, self.attention.value]
        self.snapshots = {module: _hook_ids(module) for module in modules}
        require(all(not pre and not post for pre, post in self.snapshots.values()), "foreign module hooks")
        if self.arm is not None:
            self.arm.verify_reference()
            if self.mode == "reference_null":
                require(module_digest(self.arm.learned) == module_digest(self.arm.reference),
                        "reference-null requested after learned branch changed")
        try:
            self._add(self.attention, self.attention.register_forward_pre_hook(self._before, with_kwargs=True))
            self._add(self.attention, self.attention.register_forward_hook(self._after))
            if self.capture:
                for name in ("query", "key", "value"):
                    def save(module: nn.Module, args: tuple, output: Tensor, key: str = name) -> None:
                        require(key not in self.projections, "repeated QKV projection")
                        self.projections[key] = output.detach()
                    module = getattr(self.attention, name)
                    self._add(module, module.register_forward_hook(save))
        except BaseException:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            raise
        return self

    def _before(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple | None:
        self.calls += 1
        require(module is self.attention and self.calls == 1, "duplicate or wrong target invocation")
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        require(all(call[key] is None for key in ("head_mask", "encoder_hidden_states",
                    "encoder_attention_mask", "past_key_value")), "unsupported cross/head/cache path")
        require(call["output_attentions"] is False, "attention materialization would change native dispatch")
        hidden, incoming = call["hidden_states"], call["attention_mask"]
        qmask, pmask = self.boundary.masks()
        require(hidden.shape == (*pmask.shape, 768) and hidden.dtype == torch.float32 and
                hidden.device == pmask.device and not hidden.requires_grad,
                "native frozen final-input mismatch")
        batch, length = pmask.shape
        self.input_mask_was_none = incoming is None
        if incoming is None:
            require(bool(self.boundary.attention_mask.bool().all()), "None native mask with padding")
            base = hidden.new_zeros(batch, 1, length, length)
        else:
            require(incoming.shape == (batch, 1, length, length) and
                    incoming.dtype == torch.float32 and incoming.device == hidden.device and
                    not incoming.requires_grad, "native additive mask mismatch")
            require(not bool(torch.isnan(incoming).any()) and not bool(torch.isposinf(incoming).any()),
                    "invalid native mask values")
            valid = self.boundary.attention_mask.bool()[:, None, None, :].expand_as(incoming)
            require(bool((incoming[valid] == 0).all()) and bool((incoming[~valid] < -1e4).all()),
                    "mask does not encode original padding")
            base = incoming
        if self.mode == "disabled":
            # Returning None is the hook API's exact no-replacement path.
            self._observe(incoming, None)
            return None
        if self.arm is not None:
            require(torch.is_grad_enabled(), "learned bias chain cannot run under no_grad")
            values = self.arm(hidden, self.boundary)
            delta = values.delta
            if self.capture:
                self.values = values
        else:
            delta = hidden.new_zeros(batch, length)
            if self.probe == "constant_eighth":
                delta = pmask.to(hidden.dtype) * 0.125
            elif self.probe == "alternating_eighth":
                sign = 1 - 2*(torch.arange(length, device=hidden.device) % 2)
                delta = pmask.to(hidden.dtype) * sign[None, :] * 0.125
        finite(delta, "injected delta")
        require(delta.shape == pmask.shape and bool((delta[~pmask] == 0).all()), "delta outside passage")
        cls = torch.arange(length, device=hidden.device) == 0
        allowed = cls[None, None, :, None] & pmask[:, None, None, :]
        newmask = torch.where(allowed, base + delta[:, None, None, :], base)
        require(torch.equal(newmask[~allowed], base[~allowed]), "protected native mask changed")
        if self.capture:
            self.delta, self.actual_mask = delta, newmask.detach()
            if delta.requires_grad:
                delta.retain_grad()
        self._observe(newmask, delta)
        call["attention_mask"] = newmask
        return bound.args, bound.kwargs

    def _observe(self, mask: Tensor | None, delta: Tensor | None) -> None:
        if self.observation is None:
            return
        require(not self.observation, "duplicate layout observer invocation")
        self.observation.update(actual_mask_none=mask is None,
            actual_mask_shape=None if mask is None else list(mask.shape),
            actual_mask_strides=None if mask is None else list(mask.stride()),
            actual_mask_dtype=None if mask is None else str(mask.dtype),
            actual_mask_requires_grad=None if mask is None else mask.requires_grad,
            injected_delta_requires_grad=None if delta is None else delta.requires_grad,
            actual_mask_capture="metadata_only_no_tensor_retention")

    def _after(self, module: nn.Module, args: tuple, output: tuple) -> None:
        self.outputs += 1
        require(self.outputs == 1 and isinstance(output, tuple) and len(output) == 1,
                "unexpected native attention output")
        if self.capture:
            self.context = output[0]
            require(self.context.shape == (*self.boundary.input_ids.shape, 768), "native context layout")
            if self.context.requires_grad:
                self.context.retain_grad()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        problems = []
        try:
            for module, (before_pre, before_post) in self.snapshots.items():
                now_pre, now_post = _hook_ids(module)
                owned = self.owned.get(module, set())
                if (set(now_pre) | set(now_post)) != (set(before_pre) | set(before_post) | owned):
                    problems.append("foreign mutation of scoped hooks")
        finally:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
        for module, before in self.snapshots.items():
            if _hook_ids(module) != before:
                problems.append("hook cleanup mismatch")
        if self.arm is not None:
            self.arm.verify_reference()
        if exc_type is None and (self.calls != 1 or self.outputs != 1):
            problems.append("scoped call did not complete exactly once")
        if problems:
            raise QCLSContractError("; ".join(problems)) from exc
        return False

    def release_capture(self) -> None:
        require(not self.handles, "release only after scope cleanup")
        self.values = self.delta = self.actual_mask = self.context = None
        self.projections.clear()


@dataclass(frozen=True)
class BiasOracleResult:
    probabilities: Tensor
    oracle_context: Tensor
    actual_context: Tensor
    expected_delta_gradient: Tensor | None
    actual_delta_gradient: Tensor | None
    status: str
    maximum_oracle_derivative: float | None
    allowed_keys: Tensor


def inspect_bias_oracle(scope: ScopedCLSKeyBias, *, backward: bool) -> BiasOracleResult:
    """Detached analytic row oracle, explicitly not fused native weights.

    Collect one chunk of the fixed diagnostic probe after forward/backward.
    This function never qualifies a chunk: validate_complete_probe_oracles must
    see every declared chunk of the SAME arm/checkpoint/two-question probe.
    Actual retained context gradient is before output projection.
    """
    require(scope.capture and scope.context is not None and scope.actual_mask is not None,
            "oracle requires a complete declared intervention capture")
    require(set(scope.projections) == {"query", "key", "value"}, "missing native QKV capture")
    batch, length = scope.boundary.input_ids.shape
    heads, dim = 12, 64
    def layout(tensor: Tensor) -> Tensor:
        require(tensor.shape == (batch, length, heads*dim), "projection/context layout")
        return tensor.detach().reshape(batch, length, heads, dim).transpose(1, 2)
    query, key, value = (layout(scope.projections[name]) for name in ("query", "key", "value"))
    logits = (query[:, :, 0:1, :] @ key.transpose(-2, -1)).squeeze(-2) / math.sqrt(dim)
    logits = logits + scope.actual_mask[:, 0, 0, :][:, None, :]
    probabilities = torch.softmax(logits, dim=-1)
    oracle_context = (probabilities.unsqueeze(-1) * value).sum(-2)
    actual_context = layout(scope.context)[:, :, 0, :]
    finite(probabilities, "oracle probability")
    require(torch.allclose(oracle_context, actual_context, atol=2e-5, rtol=1e-5), "oracle/native context mismatch")
    _, pmask = scope.boundary.masks()
    if not backward:
        return BiasOracleResult(probabilities, oracle_context, actual_context, None, None,
                                "FORWARD_ONLY_NOT_BACKWARD_CERTIFIED", None, pmask.detach())
    require(scope.context.grad is not None and scope.delta is not None and scope.delta.grad is not None,
            "native mask/context backward absent")
    gradient = layout(scope.context.grad)[:, :, 0, :]
    expected = (probabilities * (gradient.unsqueeze(-2) *
                                (value - oracle_context.unsqueeze(-2))).sum(-1)).sum(1)
    actual = scope.delta.grad.detach()
    finite(expected, "oracle derivative")
    finite(actual, "actual delta derivative")
    require(bool((actual[~pmask] == 0).all()), "intervention Jacobian outside allowed keys")
    maximum = float(expected[pmask].abs().max())
    return BiasOracleResult(probabilities, oracle_context, actual_context, expected, actual,
                            "BACKWARD_COLLECTED_PENDING_COMPLETE_PROBE", maximum, pmask.detach())


def validate_complete_probe_oracles(records: dict[str, BiasOracleResult], *,
                                   expected_chunks: tuple[str, ...], arm_name: str,
                                   checkpoint: str) -> dict[str, str | float | int]:
    """One fixed two-question probe, per arm AND per checkpoint, all chunks.

    expected_chunks is frozen by the reviewed producer before outputs. It is
    not inferred from records with favorable derivatives. Initial and disposable
    updated copies are separate gates; never pool their maxima or pool arms.
    This integrity check cannot authenticate the producer's frozen population.
    No chunk-local sensitivity stop is permitted, including a zero-pair question.
    """
    require(arm_name in UPSTREAM_ARMS and checkpoint in ("initial", "diagnostic_updated"),
            "backward probe arm/checkpoint applicability")
    require(expected_chunks and len(set(expected_chunks)) == len(expected_chunks), "invalid frozen probe chunks")
    require(tuple(records) == expected_chunks, "incomplete/reordered/extra fixed-probe chunks")
    maximum, allowed_count = 0.0, 0
    for record in records.values():
        require(record.status == "BACKWARD_COLLECTED_PENDING_COMPLETE_PROBE" and
                record.expected_delta_gradient is not None and record.actual_delta_gradient is not None,
                "chunk is not complete backward evidence")
        expected, actual, allowed = record.expected_delta_gradient, record.actual_delta_gradient, record.allowed_keys
        require(allowed.dtype == torch.bool and allowed.shape == expected.shape == actual.shape and
                allowed.device == expected.device == actual.device and bool(allowed.any()), "probe gradient layout")
        finite(expected, "complete oracle derivative")
        finite(actual, "complete actual derivative")
        require(bool((actual[~allowed] == 0).all()), "complete probe Jacobian outside passage")
        maximum = max(maximum, float(expected[allowed].abs().max()))
        allowed_count += int(allowed.sum())
    base = {"arm": arm_name, "checkpoint": checkpoint, "chunks": len(records),
            "allowed_keys": allowed_count, "maximum_oracle_derivative": maximum}
    if maximum < 1e-4:
        return {**base, "status": "INCONCLUSIVE_BELOW_SENSITIVITY"}
    actual_activity = False
    for record in records.values():
        expected, actual, allowed = record.expected_delta_gradient, record.actual_delta_gradient, record.allowed_keys
        strong = allowed & (expected.abs() >= max(1e-4, 0.1*maximum))
        weak = allowed & ~strong
        error = (actual - expected).abs()
        require(bool((error[strong]/expected[strong].abs() <= 0.01).all()), "complete strong bias derivative mismatch")
        require(bool((error[weak] <= 1e-5).all()), "complete weak bias derivative mismatch")
        actual_activity |= bool(actual[allowed].abs().gt(0).any())
    require(actual_activity, "missing complete allowed bias backward")
    return {**base, "status": "BACKWARD_ORACLE_PASSED_ACTIVITY_SEPARATE"}


@dataclass
class ChunkScores:
    scores: Tensor
    final_cls: Tensor
    diagnostic_scope: ScopedCLSKeyBias | None
    backend_policy: str = "unforced"


def validate_registered_arm(name: str, arm: nn.Module) -> None:
    require(name in ARMS and not arm.training, "unknown arm or stochastic train mode")
    require_eval(arm, "registered arm including frozen reference")
    if name == "HEAD768_active":
        require(type(arm) is Head768 and arm.weight.requires_grad and not arm.bias.requires_grad,
                "HEAD registration mismatch")
        return
    require(type(arm) is AnchoredBias, "upstream registry mismatch")
    critic = arm.learned.critic
    expected = {"MLP24": MLP24, "PRODUCT24_nominal": Circuit24,
                "TRIG24": TRIG24, "QCLS24": Circuit24}
    require(type(critic) is expected[name], "critic/name mismatch")
    if type(critic) is Circuit24:
        require(critic.product == (name == "PRODUCT24_nominal"), "entangling/product registry mismatch")
    require(sum(p.numel() for p in arm.parameters() if p.requires_grad) == 3100,
            "upstream active parameter inventory")
    arm.verify_reference()


def score_cached_final(final_layer: nn.Module, native_head: nn.Module, hidden: Tensor,
                       native_mask: Tensor | None, boundary: TokenBoundary, *,
                       path: str, arm_name: str | None = None,
                       arm: nn.Module | None = None, capture: bool = False,
                       observation: dict | None = None,
                       probe: str | None = None) -> ChunkScores:
    """Run the unchanged FULL final BertLayer and relevance head on one chunk.

    This is not a loader, prefix extractor, profiler or training runner. Its
    caller must certify H/mask provenance and full-native/subset/cache parity.
    It must retain all question chunks until the single all-pair backward.
    """
    require(type(final_layer).__module__ == "transformers.models.bert.modeling_bert" and
            type(final_layer).__name__ == "BertLayer", "original whole final layer required")
    require(not final_layer.training and not native_head.training, "native modules must be eval")
    require_eval(final_layer, "complete native final layer")
    require_eval(native_head, "native relevance head")
    require(all(not p.requires_grad and p.dtype == torch.float32
                for module in (final_layer, native_head) for p in module.parameters()),
            "native layer/head must be frozen FP32")
    require(native_head.weight.shape == (1, 768) and native_head.bias.shape == (1,), "native relevance head")
    _validate_attention(final_layer.attention.self)
    boundary.masks()
    require(hidden.shape == (*boundary.input_ids.shape, 768) and hidden.dtype == torch.float32,
            "cached H layout")
    require(path in ("native", "disabled", "zero", "reference_null", "active", "probe"),
            "unknown path")
    require((probe is not None) == (path == "probe") and
            probe in (None, "zero", "constant_eighth", "alternating_eighth"),
            "mask probe/path mismatch")
    require((arm_name is None and arm is None) or arm_name in ARMS and arm is not None,
            "unknown/missing registered arm")
    if arm_name is not None:
        validate_registered_arm(arm_name, arm)
    require(not capture or path in ("zero", "reference_null", "active"), "capture applicability")
    def whole_layer() -> Tensor:
        result = final_layer(hidden_states=hidden.detach(), attention_mask=native_mask,
                             head_mask=None, encoder_hidden_states=None,
                             encoder_attention_mask=None, past_key_value=None,
                             output_attentions=False)
        require(isinstance(result, tuple) and len(result) == 1, "native whole-layer output")
        return result[0][:, 0, :]
    if arm_name == "HEAD768_active":
        require(type(arm) is Head768 and path == "native" and not capture,
                "HEAD cannot install an upstream intervention")
        with torch.no_grad():
            cls = whole_layer()
        scores = arm(cls)
        finite(scores, "head-only scores")
        return ChunkScores(scores, cls, None)
    if path == "native":
        require(arm is None, "native anchor must not receive a learned arm")
        with torch.no_grad():
            cls = whole_layer()
            scores = native_head(cls).squeeze(-1)
        finite(scores, "native scores")
        return ChunkScores(scores, cls, None)
    if path in ("active", "reference_null"):
        require(arm_name in UPSTREAM_ARMS and type(arm) is AnchoredBias, "projected arm identity")
    else:
        require(arm is None, "plumbing controls cannot receive a learned arm")
    scope = ScopedCLSKeyBias(final_layer.attention.self, boundary, mode=path,
                            arm=arm, probe=probe, capture=capture, observation=observation)
    try:
        with scope:
            cls = whole_layer()
            scores = native_head(cls).squeeze(-1)
        finite(scores, "intervention scores")
        return ChunkScores(scores, cls, scope if capture else None)
    except BaseException:
        scope.release_capture()
        raise


def build_optimizers(arms: nn.ModuleDict) -> dict[str, torch.optim.Optimizer]:
    """No optimizer state is advanced until the recipe's scheduled update."""
    require(tuple(arms) == ARMS, "optimizer arm order")
    for name in ARMS:
        validate_registered_arm(name, arms[name])
    return {name: torch.optim.AdamW((p for p in arms[name].parameters() if p.requires_grad),
            lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) for name in ARMS}


def validate_optimizer_ownership(arm: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Reject a wrong-arm/stale/frozen/duplicate optimizer before any mutation."""
    require(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
            "fixed single-group AdamW required")
    active = [p for p in arm.parameters() if p.requires_grad]
    group = optimizer.param_groups[0]
    owned = list(group["params"])
    require(active and len({id(p) for p in owned}) == len(owned) and
            {id(p) for p in owned} == {id(p) for p in active}, "optimizer active parameter ownership")
    require(group["lr"] == 0.001 and group["betas"] == (0.9, 0.999) and
            group["eps"] == 1e-8 and group["weight_decay"] == 0 and
            group.get("amsgrad", False) is False and group.get("maximize", False) is False,
            "fixed AdamW configuration drift")
    require(all(id(p) in {id(q) for q in active} for p in optimizer.state), "foreign optimizer state")


def update_two_questions(arm: nn.Module, optimizer: torch.optim.Optimizer,
                         question_scores: tuple[Tensor, Tensor],
                         question_tiers: tuple[Tensor, Tensor]) -> dict[str, float | int | str]:
    """Exactly two original questions; the parent enforces256 round-robin steps.

    No-pair batches advance the parent's scheduled index without optimizer.step,
    including no update to Adam moments. There is one backward over ALL chunks.
    This helper does not select training items, pairs, steps or checkpoints.
    """
    require(len(question_scores) == len(question_tiers) == 2, "two original questions required")
    require_eval(arm, "updating arm")
    validate_optimizer_ownership(arm, optimizer)
    optimizer.zero_grad(set_to_none=True)
    terms = [ordinal_question_loss(scores, tiers) for scores, tiers in zip(question_scores, question_tiers)]
    pairs = sum(count for _, count in terms)
    loss = (terms[0][0] + terms[1][0]) / 2
    finite(loss, "two-question loss")
    if pairs == 0:
        return {"status": "NO_SUPERVISION_SCHEDULE_ADVANCES_NO_OPTIMIZER_STEP", "pairs": 0,
                "loss": 0.0, "gradient_norm_before_clip": 0.0}
    require(loss.requires_grad, "supervised loss detached from active arm")
    loss.backward()
    params = [p for p in arm.parameters() if p.requires_grad]
    require(params and any(p.grad is not None for p in params), "all active parameter gradients absent")
    for parameter in params:
        if parameter.grad is not None:
            finite(parameter.grad, "active parameter gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1, error_if_nonfinite=True)
    optimizer.step()
    for parameter in params:
        finite(parameter, "updated active parameter")
    if type(arm) is AnchoredBias:
        arm.verify_reference()
    return {"status": "ONE_FIXED_UPDATE", "pairs": pairs, "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm)}
