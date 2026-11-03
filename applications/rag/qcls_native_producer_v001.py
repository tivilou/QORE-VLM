"""Source-only native DPR producer; no CLI, admission, label reader or training.

Scientific imports in this file are reserved for a reviewed, registered P4
queue attempt. Merely preparing/AST-parsing this file never certifies runtime.
The parent must authenticate the complete B1 source/target manifest beforehand.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from applications.rag.qcls_attention_v004 import (
    MODEL_REVISION, TokenBoundary, finite, require, require_eval,
    score_cached_final, tensor_digest,
)

MODEL_ID = "facebook/dpr-reader-multiset-base"
AUDIT_MODEL_ID = "facebook__dpr-reader-multiset-base"
SOURCE_HASHES = {
    "transformers.models.dpr.modeling_dpr": "32cf97a50065213e3e31ae6f3ad81e767b0bf6428227c175f2dfa2888f319c00",
    "transformers.models.bert.modeling_bert": "39409b93fc1a0c355822b4742daac7781c41e6f63b408017646b017b508ca699",
    "transformers.modeling_utils": "d178a8810e5bd846ab3c36bf7391740f2208bb29a2fbceeffc217a55974252c1",
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def require_native_flags() -> None:
    """Assert the original numerical policy; never set global flags here."""
    require(not torch.is_autocast_enabled() and not torch.is_inference_mode_enabled(),
            "ambient mixed precision/inference mode")
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32 and
            torch.are_deterministic_algorithms_enabled(), "original precision/determinism flags drift")


@dataclass(frozen=True)
class SourceBoundCase:
    """One resolved evidence-free view, not an authority to read new data.

    from_records validates supplied original cohort/raw/visibility/trace rows.
    The parent still has to hash-bind their complete source files and verify
    full 512/128 coverage before invoking it. Targets intentionally stay out.
    """
    question: str
    question_sha256: str
    role: str
    component: str
    candidate_ids: tuple[str, ...]
    texts: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    membership: tuple[tuple[int, ...], ...]
    fallback: bool
    source_manifest_sha256: str

    @classmethod
    def from_records(cls, cohort: Mapping, raw: Mapping, visible: Mapping,
                     traces: Sequence[Mapping], *, source_manifest_sha256: str):
        require(len(source_manifest_sha256) == 64 and
                all(c in "0123456789abcdef" for c in source_manifest_sha256), "source manifest digest")
        role, question, qhash = cohort["role"], cohort["question"], cohort["question_sha256"]
        require(role in ("training", "development") and cohort["split"] == "train", "B1 role boundary")
        require(isinstance(question, str) and question.strip(), "empty source question")
        require(hashlib.sha256(question.encode()).hexdigest() == qhash == raw["question_sha256"] ==
                visible["question_sha256"] and raw["question"] == question, "question source drift")
        records, views = raw["candidates"], visible["candidates"]
        require(len(records) == len(views) == 50, "complete original50 required")
        ids = tuple(str(row["id"]) for row in records)
        require(len(set(ids)) == 50, "duplicate source candidate")
        index = {}
        for row in traces:
            if row["model_id"] != AUDIT_MODEL_ID or row["lane"] != "primary_common_context":
                continue
            require(row["question_sha256"] == qhash, "foreign trace question")
            key = str(row["candidate_id"])
            require(key not in index, "duplicate primary trace before dictionary insertion")
            index[key] = row
        require(set(index) == set(ids), "missing/extra original primary trace")
        texts, tokens, memberships = [], [], []
        for rank, (record, view) in enumerate(zip(records, views, strict=True), 1):
            candidate = ids[rank-1]
            trace = index[candidate]
            text, original = view["common_visible_passage"], record["text"]
            require(isinstance(text, str) and text.strip() and isinstance(original, str), "empty passage")
            require(str(view["candidate_id"]) == candidate and view["retrieved_rank"] == rank,
                    "visibility identity/order drift")
            require(trace["role"] == role and trace["component_id"] == cohort["component_id"] and
                    trace["retrieved_rank"] == rank, "token role/component/order drift")
            require(hashlib.sha256(original.encode()).hexdigest() == view["source_text_sha256"] ==
                    trace["source_text_sha256"], "original passage drift")
            require(text == original[:view["visible_passage_char_end"]] and
                    hashlib.sha256(text.encode()).hexdigest() == trace["model_passage_sha256"] and
                    len(text) == trace["model_passage_char_length"], "common model view drift")
            require(trace["question_complete"] is True and trace["passage_visible"] is True and
                    trace["truncated"] is False, "incomplete original input audit")
            token, seq, offsets = trace["input_ids"], trace["sequence_ids"], trace["offset_mapping"]
            require(0 < len(token) <= 350 and len(token) == len(seq) == len(offsets) and
                    trace["attention_mask"] == [1]*len(token), "audit coordinate coverage")
            require(all(type(x) is int and x >= 0 for x in token), "invalid audited token")
            require(all(s is None or type(s) is int and s in (0, 1) for s in seq), "invalid sequence role")
            for s, offset in zip(seq, offsets, strict=True):
                require(len(offset) == 2 and all(type(x) is int for x in offset), "bad offset")
                lo, hi = offset
                require(0 <= lo <= hi <= len(question if s == 0 else text) if s is not None
                        else lo == hi == 0, "offset outside exact source")
            members = tuple(-1 if s is None else s for s in seq)
            require(0 in members and 1 in members and members[0] == -1, "missing membership/CLS")
            texts.append(text); tokens.append(tuple(token)); memberships.append(members)
        fallback = visible["case_original_retrieval_fallback_required"]
        require(type(fallback) is bool and all(index[i]["B0_fallback"] == fallback for i in ids),
                "historical fallback identity")
        return cls(question, qhash, role, str(cohort["component_id"]), ids, tuple(texts),
                   tuple(tokens), tuple(memberships), fallback, source_manifest_sha256)


@dataclass(frozen=True)
class NativeChunk:
    """Original padded chunk, optionally row-filtered without any repadding."""
    encoded: dict[str, Tensor]
    boundary: TokenBoundary
    original_indices: tuple[int, ...]
    original_batch: int
    original_length: int
    question_sha256: str
    role: str
    source_manifest_sha256: str


def native_chunks(tokenizer: Any, case: SourceBoundCase, device: torch.device,
                  *, retained_indices: frozenset[int] | None = None):
    """Filtering is computational only; parent must separately bind train targets.

    Always collate all original rows first. No labels, answer strings, target
    generation, offsets retokenization, or new negatives are introduced here.
    """
    require(type(case) is SourceBoundCase and tokenizer.padding_side == "right", "native tokenizer boundary")
    require(device.type == "cuda" and (retained_indices is None or case.role == "training"),
            "P4/row-filter role boundary")
    if retained_indices is not None:
        require(all(type(i) is int and 0 <= i < 50 for i in retained_indices), "invalid row filter")
    for start in range(0, 50, 8):
        texts = list(case.texts[start:start+8])
        encoded = tokenizer(questions=[case.question]*len(texts), texts=texts,
                            return_tensors="pt", padding=True, truncation=True, max_length=350)
        require(set(encoded) <= {"input_ids", "attention_mask", "token_type_ids"} and
                {"input_ids", "attention_mask"} <= set(encoded), "native encoded keys")
        length = encoded["input_ids"].shape[1]
        require(0 < length <= 350, "original dynamic padding length")
        audit, member = [], []
        for row in range(len(texts)):
            i = start+row; ids = case.token_ids[i]; pad = length-len(ids)
            require(pad >= 0, "native collation shortened original audit")
            expected = list(ids)+[tokenizer.pad_token_id]*pad
            require(encoded["input_ids"][row].tolist() == expected and
                    encoded["attention_mask"][row].tolist() == [1]*len(ids)+[0]*pad,
                    "native token/mask mismatch")
            audit.append(expected); member.append(list(case.membership[i])+[-1]*pad)
        indices = tuple(i for i in range(start, start+len(texts))
                        if retained_indices is None or i in retained_indices)
        if not indices:
            continue
        rows = torch.tensor([i-start for i in indices], dtype=torch.int64)
        selected = {k: v.index_select(0, rows).to(device) for k, v in encoded.items()}
        audited = torch.tensor(audit, dtype=torch.int64).index_select(0, rows).to(device)
        membership = torch.tensor(member, dtype=torch.int64).index_select(0, rows).to(device)
        boundary = TokenBoundary(selected["input_ids"], selected["attention_mask"], audited,
                                 membership, membership == -1, tokenizer.cls_token_id)
        boundary.masks()
        yield NativeChunk(selected, boundary, indices, len(texts), length, case.question_sha256,
                          case.role, case.source_manifest_sha256)


def verify_loaded_reader(model: nn.Module, tokenizer: Any, *, config_sha256: str,
                         checkpoint_path: Path, checkpoint_sha256: str) -> dict:
    """Compare every actual model state tensor with the pinned local checkpoint.

    Parent must bind the complete tokenizer/config file inventory and snapshot
    revision, including their loaded identities. This is no offline loader.
    No download, attention override, dtype conversion or model construction.
    """
    import sys
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink() and
            file_sha(checkpoint_path) == checkpoint_sha256, "checkpoint bytes drift")
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        require(checkpoint_path.suffix == ".bin", "unsupported original weight container")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    active = model.state_dict()
    for name, value in active.items():
        require(name in checkpoint and checkpoint[name].dtype == value.dtype and
                torch.equal(value.detach().cpu(), checkpoint[name]), "active checkpoint tensor drift")
    extras = set(checkpoint)-set(active)
    pooler = {"span_predictor.encoder.bert_model.pooler.dense.weight",
              "span_predictor.encoder.bert_model.pooler.dense.bias"}
    for name in extras-pooler:
        require(name.endswith((".position_ids", ".token_type_ids")), "unknown inactive checkpoint tensor")
        require(torch.equal(model.get_buffer(name).detach().cpu(), checkpoint[name]), "inactive buffer drift")
    require(file_sha(checkpoint_path) == checkpoint_sha256, "checkpoint changed during comparison")
    del checkpoint, active
    require(type(model).__module__ == "transformers.models.dpr.modeling_dpr" and
            type(model).__name__ == "DPRReader", "unexpected Reader implementation")
    require(type(tokenizer).__name__ == "DPRReaderTokenizer" and tokenizer.padding_side == "right",
            "original slow DPR tokenizer required")
    for name, expected in SOURCE_HASHES.items():
        module = sys.modules.get(name)
        require(module is not None and file_sha(Path(inspect.getfile(module))) == expected,
                "runtime source does not match exact reviewed source")
    require_eval(model, "whole original Reader")
    params = tuple(model.parameters())
    require(params and all(not p.requires_grad and p.dtype == torch.float32 and
                           p.device.type == "cuda" for p in params), "frozen CUDA FP32 Reader")
    config = model.config
    require(hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False,
                                      default=str).encode()).hexdigest() == config_sha256,
            "actual model config drift")
    bert = model.span_predictor.encoder.bert_model
    require(len(bert.encoder.layer) == 12 and bert.pooler is None and
            config.hidden_size == 768 and config.num_attention_heads == 12 and
            not config.is_decoder and config.position_embedding_type == "absolute" and
            getattr(config, "_attn_implementation", None) == "sdpa", "native architecture/dispatch")
    require(not config.output_attentions and not config.output_hidden_states and
            not bert.encoder.gradient_checkpointing, "native output/checkpoint flags")
    require(all(type(layer.attention.self).__name__ == "BertSdpaSelfAttention"
                for layer in bert.encoder.layer), "actual non-SDPA layer")
    require_native_flags()
    return dict(model_id=MODEL_ID, expected_revision=MODEL_REVISION, config_sha256=config_sha256,
                source_hashes=dict(SOURCE_HASHES), dtype="torch.float32", backend="sdpa",
                active_state_equal_pinned_checkpoint=True, checkpoint_sha256=checkpoint_sha256,
                tokenizer_file_identity_certified_by_this_function=False,
                tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


class _PrefixReady(BaseException):
    pass


@dataclass(frozen=True)
class NativePrefix:
    hidden: Tensor
    mask: Tensor | None
    source_manifest_sha256: str
    input_ids_sha256: str
    mask_sha256: str | None
    hidden_sha256: str
    full_logits: Tensor | None


def extract_native_prefix(model: nn.Module, chunk: NativeChunk, *, complete_native: bool) -> NativePrefix:
    """Observe original full forward, or stop at its last-layer entry.

    No hand-written first11-layer loop/mask construction. Owned exception and
    hook are scoped; output_hidden_states/output_attentions stay disabled.
    no_grad creates ordinary detached tensors, not inference tensors.
    """
    final = model.span_predictor.encoder.bert_model.encoder.layer[-1]
    require_eval(model, "prefix Reader")
    require(all(not p.requires_grad and p.dtype == torch.float32 and p.device.type == "cuda"
                for p in model.parameters()), "prefix must use frozen CUDA FP32 backbone")
    globals_ = torch.nn.modules.module
    require(not globals_._global_forward_pre_hooks and not globals_._global_forward_hooks,
            "foreign global observers")
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),
            "foreign model hooks")
    require_native_flags()
    captured: dict[str, Any] = {}
    signature = inspect.signature(final.forward)
    def observe(module, args, kwargs):
        require(not captured, "repeated last-layer entry")
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        for key in ("head_mask", "encoder_hidden_states", "encoder_attention_mask", "past_key_value"):
            require(bound.arguments.get(key) is None, "unexpected last-layer input")
        require(bound.arguments.get("output_attentions") is False, "attention fallback requested")
        hidden, mask = bound.arguments["hidden_states"], bound.arguments.get("attention_mask")
        require(hidden.shape == (*chunk.boundary.input_ids.shape, 768) and
                hidden.dtype == torch.float32 and not hidden.requires_grad, "prefix hidden layout")
        finite(hidden, "native prefix")
        captured.update(hidden=hidden.detach(), mask=None if mask is None else mask.detach())
        if not complete_native:
            raise _PrefixReady()
    handle = final.register_forward_pre_hook(observe, with_kwargs=True)
    output = None
    try:
        with torch.no_grad():
            try:
                output = model(**chunk.encoded, output_attentions=False, output_hidden_states=False,
                               return_dict=True)
            except _PrefixReady:
                require(not complete_native and bool(captured), "unexpected prefix-stop")
    finally:
        handle.remove()
    require(bool(captured) and (output is not None) == complete_native, "incomplete prefix lifecycle")
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()),
            "prefix hook cleanup mismatch")
    logits = None if output is None else output.relevance_logits.detach()
    if logits is not None:
        require(logits.shape == (len(chunk.original_indices),), "native logit shape")
        finite(logits, "native full logits")
    return NativePrefix(captured["hidden"], captured["mask"], chunk.source_manifest_sha256,
                        tensor_digest(chunk.boundary.input_ids),
                        None if captured["mask"] is None else tensor_digest(captured["mask"]),
                        tensor_digest(captured["hidden"]), logits)


SDPA_FAMILIES = {
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_scaled_dot_product_efficient_attention": "efficient",
    "aten::_scaled_dot_product_attention_math": "math",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
}


class NativeShapeProfiler:
    """Bounded actual cached-final forward records, no kernel selection.

    CUDA kernels must correlate with recognized SDPA CPU events via the real
    profiler's kernel correlation. Unknown/missing event metadata fails closed.
    This forward component never claims actual bias-backward certification.
    A separate complete-probe backward recorder is still mandatory.
    """
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.used = len(b'{"records":{},"samples":{}}')
        self.record_bytes = {}
        self.poisoned = False
        self.cleanup_failed = False
        self._pending = None
        self.sampled_requests = {}
        self.forward_calls = 0
        self.profiled_calls = 0
        self.reused_calls = 0

    def run(self, final: nn.Module, head: nn.Module, prefix: NativePrefix,
            chunk: NativeChunk, *, path: str, arm_name=None, arm=None, capture=False,
            probe=None, backend_policy="unforced"):
        require(not self.poisoned, "profiler is terminal after failure")
        try:
            return self._run(final, head, prefix, chunk, path=path, arm_name=arm_name,
                             arm=arm, capture=capture, probe=probe,
                             backend_policy=backend_policy)
        except BaseException:
            self.poisoned = True
            result, self._pending = self._pending, None
            if result is not None:
                try:
                    if result.diagnostic_scope is not None:
                        result.diagnostic_scope.release_capture()
                except BaseException:
                    self.cleanup_failed = True
                finally:
                    # A traceback may retain this object; remove its owned graph
                    # references before propagating. Never return a partial result.
                    result.scores = result.final_cls = result.diagnostic_scope = None
            result = None
            raise

    def _run(self, final, head, prefix, chunk, *, path, arm_name, arm, capture, probe,
             backend_policy):
        require_native_flags()
        require(backend_policy in ("unforced", "math"), "unknown SDPA backend policy")
        hidden = prefix.hidden
        require(hidden.device.type == "cuda" and hidden.dtype == torch.float32 and
                prefix.source_manifest_sha256 == chunk.source_manifest_sha256 and
                prefix.input_ids_sha256 == tensor_digest(chunk.boundary.input_ids), "prefix provenance drift")
        require(prefix.mask_sha256 == (None if prefix.mask is None else tensor_digest(prefix.mask)),
                "cached native mask drift")
        require(prefix.hidden_sha256 == tensor_digest(hidden), "cached prefix hidden drift")
        signature = dict(shape=list(hidden.shape), strides=list(hidden.stride()), dtype=str(hidden.dtype),
                         heads=12, head_dim=64, device_index=hidden.device.index,
                         backend_policy=backend_policy)
        key = json_sha(signature)
        require(key in self.records or path == "native" and arm_name is None and arm is None,
                "first shape use must be untouched native, not HEAD")
        require(key in self.records or len(self.records) < 4096, "shape cap")
        def layout(mask):
            return None if mask is None else dict(shape=list(mask.shape), strides=list(mask.stride()),
                                                   dtype=str(mask.dtype), requires_grad=mask.requires_grad)
        request = json_sha(dict(path=path, arm=arm_name, probe=probe,
                                grad=torch.is_grad_enabled(), capture=capture,
                                backend_policy=backend_policy, dtype=str(hidden.dtype),
                                device_index=hidden.device.index,
                                original_mask_none=prefix.mask is None))
        sampled = self.sampled_requests.get(request)
        self.forward_calls += 1
        # Sample each path/backend/grad class once, on its first actual shape.
        # This is representative dispatch evidence, NOT per-shape certification.
        # Repeated forwards still execute
        # all provenance/mask/autograd checks, but do not start Kineto again.
        # A sampled kernel is evidence for that sample, never for every repeat.
        prof = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                       record_shapes=True) if sampled is None else None)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            with prof if prof is not None else nullcontext():
                selected = (sdpa_kernel(SDPBackend.MATH)
                            if backend_policy == "math" else nullcontext())
                with selected:
                    observation = {}
                    result = score_cached_final(final, head, hidden, prefix.mask, chunk.boundary,
                                                path=path, arm_name=arm_name, arm=arm, capture=capture,
                                                observation=observation, probe=probe)
                    result.backend_policy = backend_policy
                    self._pending = result
                    if prof is not None:
                        torch.cuda.synchronize(hidden.device)
        require(not observed, "warning during selected SDPA profile")
        families, roots, cuda = [], [], []
        if prof is not None:
            self.profiled_calls += 1
            events = list(prof.events())
            for event in events:
                name = event.name
                if name in SDPA_FAMILIES:
                    families.append(SDPA_FAMILIES[name])
                    roots.append(event)
                elif name.startswith("aten::_scaled_dot_product") and "backward" not in name:
                    require(False, "unrecognized SDPA family")
            root_ids = {id(root) for root in roots}
            for event in events:
                parent = event
                while parent is not None and id(parent) not in root_ids:
                    parent = parent.cpu_parent
                if parent is not None:
                    cuda.extend(dict(name=k.name, device=k.device) for k in event.kernels)
        else:
            self.reused_calls += 1
            families = [sampled["aten_family"]]
            cuda = sampled["actual_cuda_kernels"]
        require(len(families) == 1 and cuda, "missing unique forward family/correlated CUDA kernel")
        require(backend_policy != "math" or families[0] == "math",
                "math policy did not select math SDPA")
        require(key not in self.records or self.records[key]["family"] == families[0],
                "native/zero/null/active family changed")
        if path == "native":
            # Native and HEAD never install an attention intervention. For
            # modified paths _before reports metadata from the actual newmask,
            # not an inferred/detached mask or fabricated autograd flag.
            mask = prefix.mask
            require(not observation, "native unexpectedly received a mask hook")
            observation = dict(actual_mask_none=mask is None,
                actual_mask_shape=None if mask is None else list(mask.shape),
                actual_mask_strides=None if mask is None else list(mask.stride()),
                actual_mask_dtype=None if mask is None else str(mask.dtype),
                actual_mask_requires_grad=None if mask is None else mask.requires_grad,
                injected_delta_requires_grad=None,
                actual_mask_capture="native_argument_metadata_no_tensor_retention")
        else:
            require(bool(observation), "actual mask metadata observer did not run")
        if sampled is None:
            require(len(self.sampled_requests) < 128, "profile request class cap")
            sampled = dict(aten_family=families[0], actual_cuda_kernels=cuda,
                           signature=signature)
            sample_size = len(json.dumps(sampled, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True).encode())
            self.used += len(json.dumps(request).encode())+1+sample_size+bool(self.sampled_requests)
            self.sampled_requests[request] = sampled
        row = dict(path=path, arm=arm_name, probe=probe, grad_enabled=torch.is_grad_enabled(), capture=capture,
                   original_mask_none=prefix.mask is None, **observation,
                   representative_aten_family=families[0], kernel_sample_id=request,
                   kernel_evidence="representative_sample_only_not_per_shape_or_call",
                   output_requires_grad=result.scores.requires_grad,
                   native_attention_class=type(final.attention.self).__name__,
                   backend_policy=backend_policy, backend_forced=backend_policy != "unforced",
                   backward_certified=False, byte_identical_instructions_claimed=False)
        # Mask layout/grad is a nested variant, not another population/shape.
        # Actual values stay out of the signature; same-mask variants may have
        # different pad positions. Family still has the original per-H anchor.
        variant = json_sha(dict(path=path, arm=arm_name, probe=probe,
                                grad=torch.is_grad_enabled(), capture=capture,
                                backend_policy=backend_policy,
                                original_mask=layout(prefix.mask),
                                actual_mask=dict(shape=row["actual_mask_shape"],
                                    strides=row["actual_mask_strides"], dtype=row["actual_mask_dtype"]),
                                output_requires_grad=result.scores.requires_grad,
                                delta_requires_grad=row["injected_delta_requires_grad"]))
        record = self.records.setdefault(key, dict(signature=signature, family=families[0], variants={}))
        # Check actual forward metadata on every call. Kernel metadata on reused
        # calls is explicitly inherited from the one recorded structural sample.
        prior = record["variants"].get(variant)
        if prior is None:
            record["variants"][variant] = row
        else:
            require(prior == row, "same signature/variant changed kernel record")
        if prior is None:
            # Count the changed shape record, not the entire growing registry.
            size = len(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True).encode())
            previous = self.record_bytes.get(key)
            if previous is None:
                self.used += len(json.dumps(key).encode())+1+size+(len(self.record_bytes) > 0)
            else:
                self.used += size-previous
            self.record_bytes[key] = size
        # Include the shared sample registry, and reserve space for summary
        # counters/schema. JSON expands shared lists; only sample IDs belong in
        # per-shape variants, never copies of the representative kernel list.
        require(self.used <= 16*(1 << 20)-4096, "complete profiler metadata cap")
        self._pending = None
        return result

    def summary(self):
        result = dict(schema="qcls.sdpa-profiler-summary.v2", records=self.records,
                      samples=self.sampled_requests, registry_serialized_bytes=self.used,
                      sampling_policy="first_path_backend_grad_class_representative_only",
                      per_shape_kernel_certification=False, forward_calls=self.forward_calls,
                      profiled_calls=self.profiled_calls, unprofiled_calls=self.reused_calls,
                      backend_policies=["unforced_native_anchor", "scoped_math_candidate"])
        require(len(json.dumps(result, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode()) <= 16*(1 << 20),
                "complete profiler summary cap")
        return result
