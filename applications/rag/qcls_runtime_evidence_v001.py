"""Actual queued QCLS observations. Source preparation; never a local test.

This layer neither chooses data nor changes the model, objective or optimizer.
Its caller supplies the complete fixed probe and original native chunks. A
successful byte writer alone cannot set the sample-trace qualification flag.
"""
from __future__ import annotations

import hashlib
import functools
import json
import math
import re
import sys
import warnings
from dataclasses import asdict

import torch

from applications.rag.qcls_attention_v004 import (
    ARMS, UPSTREAM_ARMS, finite, inspect_bias_oracle, module_digest,
    ordinal_question_loss, require, tensor_digest, validate_complete_probe_oracles,
)
from applications.rag.qcls_artifacts_v001 import read_full_coordinates
from applications.rag.qcls_native_producer_v001 import require_native_flags


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def terminal(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.failed = True
            raise
    return call


def same_tensor(actual, expected, *, exact=False, atol=2e-5, rtol=1e-5):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            "replay tensor shape/dtype")
    finite(actual, "replay actual"); finite(expected, "replay expected")
    okay = torch.equal(actual, expected) if exact else torch.allclose(actual, expected, atol=atol, rtol=rtol)
    require(bool(okay), "replay numerical mismatch")
    difference = actual.detach().to(torch.complex128 if actual.is_complex() else torch.float64)
    difference = difference-expected.detach().to(difference.dtype)
    return float(difference.abs().max()) if actual.numel() else 0.0


def rank50(scores, candidate_ids):
    require(scores.ndim == 1 and scores.numel() == len(candidate_ids) == 50,
            "complete original50 required")
    finite(scores, "ranking scores")
    values = scores.detach().cpu().tolist()
    # No rounding, tolerance-based ties, omitted rows, or label input.
    return tuple(sorted(range(50), key=lambda i: (-values[i], i, str(candidate_ids[i]))))


def pair_loss(scores, tiers):
    require(len(scores) == len(tiers) == 2, "exact complete two-question probe")
    parts = [ordinal_question_loss(s, t) for s, t in zip(scores, tiers, strict=True)]
    return (parts[0][0]+parts[1][0])*0.5, (parts[0][1], parts[1][1])


def backward_without_update(arm, scores, tiers):
    """Diagnostic-updated checkpoint: backward only; never a second update."""
    arm.zero_grad(set_to_none=True)
    loss, counts = pair_loss(scores, tiers)
    require(sum(counts) > 0, "fixed supervised probe lost all pairs")
    finite(loss, "fixed diagnostic loss")
    loss.backward()
    return dict(status="diagnostic_backward_no_update", loss=float(loss.detach()), pairs=list(counts))


class BackwardRecorder:
    """Observe the actual complete two-question loss/backward, without overrides.

    For math SDPA the backward is decomposed: retain actual softmax/bmm event
    names and disclose that the complete step also contains critic/optimizer
    work. Oracle/context/delta provenance, rather than a name substring alone,
    binds the bias gradient to the observed original attention output.
    """
    def __init__(self):
        self.rows = []
        self.bytes = 0
        self.failed = False

    @terminal
    def run(self, *, arm_name, arm, checkpoint, chunks, forward_rows, action):
        require(not self.failed and arm_name in ARMS and
                checkpoint in ("initial", "diagnostic_updated"), "backward recorder state")
        require(len(chunks) == len(forward_rows) and chunks, "complete backward chunk inventory")
        require(len(set(chunks)) == len(chunks), "duplicate probe chunk before recording")
        require_native_flags()
        backend_policies = {getattr(row, "backend_policy", None) for row in forward_rows}
        require(backend_policies == {"math"}, "fixed probe requires one math SDPA backend policy")
        before = {n: p.detach().clone() for n, p in arm.named_parameters() if p.requires_grad}
        state = module_digest(arm)
        fired = {}
        handles = []
        def watch(tensor, key):
            require(tensor.requires_grad and key not in fired, "missing/duplicate current gradient target")
            fired[key] = 0
            def arrived(gradient):
                finite(gradient, "current backward hook gradient")
                fired[key] += 1
            handles.append(tensor.register_hook(arrived))
        try:
            for key, row in zip(chunks, forward_rows, strict=True):
                binding = getattr(row, "forward_binding", None)
                require(type(binding) is dict and binding["checkpoint"] == checkpoint and
                        binding["arm"] == arm_name and binding["chunk_key"] == key and
                        binding["arm_state"] == state, "old/wrong forward checkpoint or arm")
                watch(row.scores, key+":scores")
                if arm_name in UPSTREAM_ARMS:
                    scope = row.diagnostic_scope
                    require(scope is not None and scope.arm is arm and scope.mode == "active" and
                            scope.context is not None and scope.delta is not None and
                            scope.context.grad is None and scope.delta.grad is None,
                            "stale or missing probe gradients")
                    require(tensor_digest(scope.boundary.input_ids) == binding["input_ids_sha256"] and
                            tensor_digest(scope.boundary.attention_mask) == binding["attention_mask_sha256"],
                            "probe chunk token/mask binding")
                    watch(scope.context, key+":context")
                    watch(scope.delta, key+":delta")
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True) as profiler:
                    result = action()
                    torch.cuda.synchronize()
            require(not observed, "warning during actual backward")
            require(fired and all(count == 1 for count in fired.values()),
                    "recorded forward tensors did not participate exactly once in this backward")
            parameters = {n: p for n, p in arm.named_parameters() if p.requires_grad}
            require(parameters.keys() == before.keys(), "trainable inventory changed")
            groups = {"weight": []} if arm_name == "HEAD768_active" else {
                "query": [], "passage": [], "critic": []}
            for name, parameter in parameters.items():
                require(parameter.grad is not None, "missing declared trainable gradient")
                finite(parameter.grad, "actual parameter gradient")
                finite(parameter, "actual updated parameter")
                key = "weight" if arm_name == "HEAD768_active" else name.split(".")[1]
                require(key in groups, "unknown parameter group")
                groups[key].append((parameter.grad.detach(), parameter.detach()-before[name]))
            stats = {}
            for name, values in groups.items():
                require(bool(values), "empty declared group")
                g = math.sqrt(sum(float((a.double()**2).sum()) for a, _ in values))
                u = math.sqrt(sum(float((b.double()**2).sum()) for _, b in values))
                if checkpoint == "diagnostic_updated":
                    require(u == 0, "diagnostic-updated backward changed parameters")
                stats[name] = dict(gradient_l2=g, update_l2=u)
            roots = []
            names = []
            events = list(profiler.events())
            for event in events:
                lower = event.name.lower()
                if "backward" in lower:
                    names.append(event.name)
                    if ("scaled_dot_product" in lower or "softmax" in lower or "bmm" in lower):
                        roots.append(event)
            kernels = []
            for event in events:
                parent = event
                while parent is not None and not any(parent is item for item in roots):
                    parent = parent.cpu_parent
                if parent is not None:
                    kernels.extend(dict(name=k.name, device=k.device) for k in event.kernels)
            oracle_status = {"status": "NOT_APPLICABLE_HEAD_NO_ATTENTION_BACKWARD"}
            if arm_name in UPSTREAM_ARMS:
                require(roots and kernels, "missing correlated actual attention-backward events")
                records = {}
                for key, row in zip(chunks, forward_rows, strict=True):
                    require(key not in records, "duplicate oracle record before insertion")
                    require(row.diagnostic_scope is not None, "missing real forward scope")
                    records[key] = inspect_bias_oracle(row.diagnostic_scope, backward=True)
                oracle_status = validate_complete_probe_oracles(
                    records, expected_chunks=tuple(chunks), arm_name=arm_name, checkpoint=checkpoint)
                if oracle_status["status"] == "INCONCLUSIVE_BELOW_SENSITIVITY":
                    raise ProbeBelowSensitivity(oracle_status)
                require(oracle_status["status"] == "BACKWARD_ORACLE_PASSED_ACTIVITY_SEPARATE",
                        "fixed oracle failed")
            else:
                require(all(row.diagnostic_scope is None for row in forward_rows), "HEAD altered attention")
            # A fixed low-sensitivity oracle stops cleanly before interpreting
            # downstream small gradients as a fatal mechanism defect.
            for item in stats.values():
                require(item["gradient_l2"] > 1e-10 and
                        (checkpoint != "initial" or item["update_l2"] > 1e-10),
                        "inactive declared gradient/update group")
            row = dict(arm=arm_name, checkpoint=checkpoint, groups=stats, action=result,
                       oracle=oracle_status, actual_backward_cpu_names=sorted(set(names)),
                       forward_arm_state_sha256=canonical_sha(state), current_gradient_hooks=fired,
                       correlated_backward_cuda_kernels=kernels,
                       event_scope="complete fixed loss/backward/update; math names may include critic work",
                       isolated_math_kernel_attribution=False, backend_policy="math",
                       backend_forced=True)
            require(names, "actual backward events unavailable")
            self.rows.append(row)
            self.bytes = len(json.dumps(self.rows, sort_keys=True).encode())
            require(self.bytes <= 16*1024**2, "backward summary cap")
            return row
        except BaseException:
            self.failed = True
            raise
        finally:
            for handle in reversed(handles):
                handle.remove()
            before.clear()


class TraceWriter:
    """Full private coordinates, exhausted independent reader and real replay.

    Each record fixes its source/role/chunk/arm/checkpoint binding before
    values are written. Descriptor files remain honest byte descriptors; a
    separate receipt records finite/coordinate/slice/same-sample checks.
    """
    def __init__(self, artifacts, *, field_specs, namespace: str):
        self.artifacts = artifacts
        require(type(field_specs) is dict and field_specs, "predeclared complete field specification")
        self.specs = json.loads(json.dumps(field_specs, allow_nan=False, sort_keys=True))
        self.specs_sha256 = canonical_sha(self.specs)
        require(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", namespace)), "trace namespace")
        self.namespace = namespace
        for name, spec in self.specs.items():
            require(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", name)) and
                    set(spec) == {"dtype", "shape", "binding", "replay"} and
                    spec["dtype"] in ("float32", "complex64", "int64", "bool") and
                    type(spec["replay"]) is bool and len(namespace+"-"+name) <= 92,
                    "field specification")
        self.records = {}
        self.failed = False

    @terminal
    def tensor(self, name, tensor, binding):
        require(not self.failed and name not in self.records, "trace duplicate/terminal")
        try:
            require(sys.byteorder == "little", "full-value byte order")
            value = tensor.detach().to("cpu").contiguous()
            kinds = {torch.float32: "float32", torch.int64: "int64",
                     torch.bool: "bool", torch.complex64: "complex64"}
            require(value.dtype in kinds, "unregistered trace dtype")
            require(canonical_sha(self.specs) == self.specs_sha256 and name in self.specs,
                    "undeclared/mutated trace field")
            frozen_binding = json.loads(json.dumps(binding, allow_nan=False, sort_keys=True))
            spec = self.specs[name]
            require(kinds[value.dtype] == spec["dtype"] and list(value.shape) == spec["shape"] and
                    canonical_sha(frozen_binding) == canonical_sha(spec["binding"]),
                    "field dtype/shape/source binding differs from prospective specification")
            finite(value, "producer full-value")
            raw = value.numpy().tobytes(order="C")
            blocks = (raw[i:i+1024**2] for i in range(0, len(raw), 1024**2))
            stem = self.namespace+"-"+name
            receipt = self.artifacts.full_value(stem+".bin", blocks, dtype=kinds[value.dtype],
                shape=tuple(value.shape), producer_binding_sha256=canonical_sha(frozen_binding))
            descriptor = json.loads(self.artifacts.read(self.artifacts.committed[stem+".json"]))
            require(descriptor["producer_binding_sha256"] == canonical_sha(frozen_binding),
                    "descriptor source binding mismatch")
            flat = value.reshape(-1)
            selected = {0, flat.numel()//2, flat.numel()-1} if flat.numel() else set()
            slices = {}
            count = 0
            # Exhaust every coordinate: no short-circuit generator success.
            for index, actual in read_full_coordinates(self.artifacts.root, descriptor):
                expected = flat[index].item()
                require(actual == expected, "full-value independent reader mismatch")
                if index in selected:
                    slices[str(index)] = dict(producer=repr(expected), reader=repr(actual))
                count += 1
            require(count == flat.numel(), "full-value reader not exhausted")
            record = dict(binding=frozen_binding, values=asdict(receipt), dtype=kinds[value.dtype],
                          shape=list(value.shape), coordinates=count, reader_eof=True,
                          finite=True, reader_slices=slices, caller_values_equal=False,
                          artifact_stem=stem, replay_required=spec["replay"],
                          same_sample_replay_authenticated=False,
                          exact_replay_binding=None)
            self.records[name] = record
            return receipt
        except BaseException:
            self.failed = True
            raise

    @terminal
    def compare_caller_values(self, names, fresh, expected, *, exact=False):
        """Numerical comparison only; passing the same value cannot certify a rerun."""
        require(not self.failed and names and len(names) == len(fresh) == len(expected), "trace replay inventory")
        try:
            for name, actual, original in zip(names, fresh, expected, strict=True):
                require(name in self.records, "replay of unwritten trace")
                record = self.records[name]
                require(list(original.shape) == record["shape"] and
                        hashlib.sha256(original.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                        == record["values"]["sha256"], "recorded replay source drift")
                error = same_tensor(actual, original, exact=exact)
                record.update(caller_values_equal=True, comparison_max_absolute_error=error)
        except BaseException:
            self.failed = True
            raise

    @terminal
    def authenticate_exact_replay(self, names, fresh, expected, *, replay_binding, exact=False):
        """Authenticate a reviewed fresh source/model forward, not a caller clone.

        The exact parent supplies tensors returned by a second tokenizer/native
        prefix/final-forward pass.  Storage aliasing and a complete fixed replay
        identity are rejected before numerical comparison.
        """
        require(not self.failed and names and len(names) == len(fresh) == len(expected),
                "exact replay inventory")
        required = {"source_manifest_sha256", "question_sha256", "candidate_indices",
                    "checkpoint", "arm", "model_state_sha256", "arm_state_sha256",
                    "fresh_native_prefix", "fresh_final_forward"}
        require(type(replay_binding) is dict and set(replay_binding) == required and
                replay_binding["fresh_native_prefix"] is True and
                replay_binding["fresh_final_forward"] is True,
                "exact replay source/model binding")
        frozen_binding = json.loads(json.dumps(replay_binding, allow_nan=False, sort_keys=True))
        for name, actual, original in zip(names, fresh, expected, strict=True):
            require(name in self.records and self.specs[name]["replay"] is True,
                    "replay of unregistered/non-replay field")
            require(actual is not original and actual.device == original.device and
                    actual.dtype == original.dtype and actual.shape == original.shape,
                    "fresh replay tensor identity/layout")
            if actual.numel():
                require(actual.untyped_storage().data_ptr() != original.untyped_storage().data_ptr(),
                        "fresh replay aliases original storage")
            record = self.records[name]
            require(hashlib.sha256(original.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    == record["values"]["sha256"], "recorded exact replay source drift")
            error = same_tensor(actual, original, exact=exact)
            record.update(caller_values_equal=True, comparison_max_absolute_error=error,
                          same_sample_replay_authenticated=True,
                          exact_replay_binding=frozen_binding)

    @terminal
    def finish(self, receipt_name: str):
        require(not self.failed and set(self.specs) == set(self.records) and self.records and
                canonical_sha(self.specs) == self.specs_sha256,
                "complete frozen trace inventory required")
        require(bool(re.fullmatch(r"[a-z][a-z0-9_-]{0,83}\.json", receipt_name)),
                "trace completion receipt name")
        require(all(r["caller_values_equal"] and
                    r["same_sample_replay_authenticated"] == r["replay_required"]
                    for r in self.records.values()), "uncompared or incorrectly certified field")
        for name, record in self.records.items():
            spec = self.specs[name]
            require(record["binding"] == spec["binding"] and record["shape"] == spec["shape"] and
                    record["dtype"] == spec["dtype"], "field metadata mutated after write")
            from applications.rag.qcls_artifacts_v001 import Receipt
            receipt = Receipt(**record["values"])
            self.artifacts._verify(receipt)
            descriptor_receipt = self.artifacts.committed[record["artifact_stem"]+".json"]
            descriptor = json.loads(self.artifacts.read(descriptor_receipt))
            require(descriptor["producer_binding_sha256"] == canonical_sha(spec["binding"]) and
                    descriptor["values"] == record["values"] and descriptor["shape"] == spec["shape"] and
                    descriptor["dtype"] == spec["dtype"], "descriptor differs from frozen field")
            exhausted = []
            count = sum(1 for _ in read_full_coordinates(
                self.artifacts.root, descriptor, exhausted=exhausted.append))
            require(count == record["coordinates"] and len(exhausted) == 1,
                    "full-value final reader not exhausted")
            record["descriptor_receipt"] = asdict(descriptor_receipt)
            record["final_reader_receipt"] = exhausted[0]
        replayed = sorted(name for name, record in self.records.items()
                          if record["same_sample_replay_authenticated"])
        return self.artifacts.json(receipt_name, dict(
            schema="qcls.sample-trace.field-replay.v1",
            status="complete_field_coordinates_with_declared_exact_replays",
            private=True, fields=self.records, scientific_screen_complete=False,
            prospective_field_specs_sha256=self.specs_sha256,
            exact_replayed_fields=replayed,
            full_sample_trace_v2_qualified=False,
            replay_origin_authenticated_by_this_component=bool(replayed),
            required_parent="fixed complete field/arm/checkpoint seal plus sample selection and reviewed exact rerun"),
            trace=True)


def release_rows(rows):
    """Every owned capture is released even if one release fails."""
    failures = []
    for row in rows:
        try:
            if row.diagnostic_scope is not None:
                row.diagnostic_scope.release_capture()
        except BaseException as error:
            failures.append(type(error).__name__)
        finally:
            row.scores = row.final_cls = row.diagnostic_scope = None
    require(not failures, "owned graph cleanup failed: "+",".join(failures))


class ProbeBelowSensitivity(RuntimeError):
    """Clean fixed-probe sensitivity stop; no alternate probe or rescue."""
    def __init__(self, evidence):
        super().__init__("INCONCLUSIVE_BELOW_SENSITIVITY")
        self.evidence = evidence
