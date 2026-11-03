"""Fixed queued QCLS producer core, not a standalone/admitted experiment.

The exact parent must still own source deployment, task/attempt admission,
aggregate resources, complete trace harness and atomic completion/failure.
All tensor/model use in this file belongs exclusively to that reviewed P4 job.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import DPRReader, DPRReaderTokenizer

from applications.rag.qcls_attention_v004 import (
    ARMS, UPSTREAM_ARMS, build_optimizers, initialize_arms, module_digest,
    ordinal_question_loss, require, tensor_digest, update_two_questions,
)
from applications.rag.qcls_b1_resolver_v001 import (
    BoundFiles, BoundPopulation, TrainingTargets, bind_model_inventory,
)
from applications.rag.qcls_native_producer_v001 import (
    NativeShapeProfiler, extract_native_prefix, native_chunks, require_native_flags,
    verify_loaded_reader,
)
from applications.rag.qcls_runtime_evidence_v001 import (
    BackwardRecorder, backward_without_update, rank50, release_rows, same_tensor, terminal,
)
from applications.rag.qcls_sealed_evaluation_v001 import PREDICTORS


def offline_reader(model_root, asset_manifest, *, device):
    """Pinned local files only; original slow tokenizer/default SDPA dispatch.

    Establish the original baseline numeric policy before any model work.
    No attention-backend override, download, network retry or config rewrite.
    Complete model-file inventory and every active state tensor are checked.
    The returned config digest describes the actually loaded config, rather
    than pretending a runtime-generated config hash was an external pin.
    """
    require(device.type == "cuda" and torch.cuda.is_available(), "original P4 CUDA required")
    files = BoundFiles()
    spec, inventory = bind_model_inventory(files, asset_manifest, model_root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require_native_flags()
    tokenizer = DPRReaderTokenizer.from_pretrained(str(model_root), local_files_only=True)
    model = DPRReader.from_pretrained(str(model_root), local_files_only=True, torch_dtype=torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    config_sha = hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, ensure_ascii=False,
                                          default=str).encode()).hexdigest()
    receipt = verify_loaded_reader(model, tokenizer, config_sha256=config_sha,
        checkpoint_path=Path(model_root)/spec["weight"]["name"],
        checkpoint_sha256=spec["weight"]["sha256"])
    # Reauthenticate tokenizer/config/weights after loader and active-state
    # validation. No cached statement substitutes for this actual file pass.
    _, after_inventory = bind_model_inventory(BoundFiles(), asset_manifest, model_root)
    require(inventory == after_inventory, "offline assets changed during load")
    receipt.update(file_inventory_sha256=inventory, config_hash_kind="actual_loaded_observation",
                   tokenizer_file_inventory_verified=True, local_files_only=True)
    return model, tokenizer, receipt


@dataclass
class CachedQuestion:
    case: object
    chunks: tuple
    tiers: torch.Tensor | None
    native_scores: torch.Tensor | None = None


class FixedFiveArmDriver:
    """Shared prefix per complete two-question batch, five independent arms.

    budget_checkpoint is supplied by the reviewed aggregate supervisor. It is
    invoked around every costly operation; this callback is not itself a
    resource guard or an authorization token. Full-entry review is mandatory.
    """
    def __init__(self, model, tokenizer, population, targets, *, budget_checkpoint,
                 initialization_seed=20261029):
        require(type(population) is BoundPopulation and type(targets) is TrainingTargets,
                "actual source and train-only target binding required")
        require(len(population.cases) == 640 and len(population.train_order) == 512,
                "unchanged source denominator")
        self.model, self.tokenizer = model, tokenizer
        self.population, self.targets = population, targets
        self.cases = {case.question_sha256: case for case in population.cases}
        require(len(self.cases) == 640 and set(targets.rows_by_question) == set(population.train_order),
                "training source/target identity")
        require(sum(len(v) for v in targets.rows_by_question.values()) == 7339,
                "original reviewed training union only")
        self.final = model.span_predictor.encoder.bert_model.encoder.layer[-1]
        self.head = model.span_predictor.qa_classifier
        self.device = next(model.parameters()).device
        require(self.device.type == "cuda", "CPU scientific fallback forbidden")
        require(callable(budget_checkpoint), "aggregate parent checkpoint required")
        self.check = budget_checkpoint
        self.profiler, self.backward = NativeShapeProfiler(), BackwardRecorder()
        self.candidate_backend = "math"
        self.backend_parity_max = 0.0
        self.initialization_seed = initialization_seed
        self.initial_arms, self.initial_digests = initialize_arms(self.head, seed=initialization_seed)
        self.initial_arms.to(self.device)
        self.frozen_backbone = module_digest(model)
        self.native_repeat_max = None
        self.scheduled_steps = {name: 0 for name in ARMS}
        self.failed = False

    @terminal
    def integrity(self):
        require(not self.failed, "driver terminal after failure")
        require_native_flags()
        require(all(not p.requires_grad and p.grad is None for p in self.model.parameters()),
                "backbone trainable/gradient leak")
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()),
                "owned hook survived operation")

    @terminal
    def cache(self, questions, *, reviewed_only, phase):
        require(len(questions) <= 2, "no whole-corpus hidden cache")
        if phase == "screen_training":
            self.check(phase, category="shared_feature_preparation")
        else:
            self.check(phase)
        output = []
        for case in questions:
            retained = None
            if reviewed_only:
                require(case.role == "training", "no development supervision")
                retained = frozenset(i for i, _ in self.targets.rows_by_question[case.question_sha256])
            chunks, native_parts = [], []
            for chunk in native_chunks(self.tokenizer, case, self.device, retained_indices=retained):
                self.check(phase)
                prefix = extract_native_prefix(self.model, chunk, complete_native=False)
                # Every new filtered B/L shape gets an untouched native anchor
                # before any arm/modified mask uses it. All cost is charged.
                native = self.profiler.run(self.final, self.head, prefix, chunk, path="native",
                                           backend_policy="unforced")
                compatible = self.profiler.run(self.final, self.head, prefix, chunk, path="native",
                                               backend_policy=self.candidate_backend)
                try:
                    self.backend_parity_max = max(
                        self.backend_parity_max, same_tensor(compatible.scores, native.scores))
                    if not reviewed_only:
                        native_parts.append(native.scores.detach())
                finally:
                    release_rows([native, compatible])
                chunks.append((chunk, prefix))
                self.check(phase)
            indices = tuple(i for chunk, _ in chunks for i in chunk.original_indices)
            if reviewed_only:
                target = self.targets.rows_by_question[case.question_sha256]
                require(indices == tuple(i for i, _ in target), "original reviewed row order")
                tiers = torch.tensor([y for _, y in target], dtype=torch.int64, device=self.device)
            else:
                require(indices == tuple(range(50)), "all original50 candidates retained")
                tiers = None
            output.append(CachedQuestion(case, tuple(chunks), tiers,
                None if reviewed_only else torch.cat(native_parts)))
        if phase == "screen_training":
            self.check(phase, category="all_learned_training")
        return tuple(output)

    @terminal
    def forward(self, name, arm, cached, *, capture, phase, checkpoint=None):
        require(name in ARMS and len(cached) <= 2, "registered bounded arm batch")
        scores, rows, keys = [], [], []
        try:
            for question in cached:
                parts = []
                for chunk, prefix in question.chunks:
                    self.check(phase)
                    key = question.case.question_sha256+":"+str(chunk.original_indices[0])
                    require(key not in keys, "duplicate chunk before append")
                    row = self.profiler.run(self.final, self.head, prefix, chunk,
                        path="native" if name == "HEAD768_active" else "active",
                        arm_name=name, arm=arm, capture=capture and name in UPSTREAM_ARMS,
                        backend_policy=self.candidate_backend)
                    row.forward_binding = dict(arm=name, checkpoint=checkpoint, chunk_key=key,
                        arm_state=module_digest(arm), question_sha256=question.case.question_sha256,
                        role=question.case.role, component=question.case.component,
                        original_indices=list(chunk.original_indices),
                        source_manifest_sha256=prefix.source_manifest_sha256,
                        input_ids_sha256=tensor_digest(chunk.boundary.input_ids),
                        attention_mask_sha256=tensor_digest(chunk.boundary.attention_mask),
                        hidden_sha256=prefix.hidden_sha256)
                    rows.append(row); keys.append(key); parts.append(row.scores)
                scores.append(torch.cat(parts) if parts else torch.empty(0, device=self.device))
            return tuple(scores), rows, tuple(keys)
        except BaseException:
            self.failed = True
            release_rows(rows)
            raise

    @terminal
    def native_null_parity(self, *, record):
        """Every question/candidate of the fixed16 complete source components."""
        selected = set(self.population.preflight_components)
        cases = tuple(self.cases[q] for q in self.population.train_order if self.cases[q].component in selected)
        require(cases and {c.component for c in cases} == selected, "fixed whole component inventory")
        repeat_max = 0.0
        for case in cases:
            collected = {"full": [], "repeat": [], "cached": [], "disabled": [], "zero": []}
            collected.update({name: [] for name in UPSTREAM_ARMS})
            for chunk in native_chunks(self.tokenizer, case, self.device):
                self.check("preflight")
                first = extract_native_prefix(self.model, chunk, complete_native=True)
                repeated = extract_native_prefix(self.model, chunk, complete_native=True)
                require(first.full_logits is not None and repeated.full_logits is not None,
                        "full original forwards required")
                repeat_max = max(repeat_max,
                    same_tensor(first.full_logits, repeated.full_logits, atol=1e-7, rtol=0),
                    same_tensor(first.hidden, repeated.hidden, atol=1e-7, rtol=0))
                collected["full"].append(first.full_logits); collected["repeat"].append(repeated.full_logits)
                for path in ("native", "disabled", "zero"):
                    row = self.profiler.run(self.final, self.head, first, chunk, path=path,
                                            backend_policy=self.candidate_backend)
                    try:
                        same_tensor(row.scores, first.full_logits)
                        collected[{"native": "cached", "disabled": "disabled", "zero": "zero"}[path]].append(
                            row.scores.detach())
                    finally:
                        release_rows([row])
                for name in UPSTREAM_ARMS:
                    row = self.profiler.run(self.final, self.head, first, chunk, path="reference_null",
                        arm_name=name, arm=self.initial_arms[name],
                        backend_policy=self.candidate_backend)
                    try:
                        same_tensor(row.scores, first.full_logits)
                        collected[name].append(row.scores.detach())
                    finally:
                        release_rows([row])
                self.check("preflight")
            expected = rank50(torch.cat(collected["full"]), case.candidate_ids)
            for name, parts in collected.items():
                require(rank50(torch.cat(parts), case.candidate_ids) == expected,
                        "native/null complete50 ordering changed")
            record(dict(question_sha256=case.question_sha256, component=case.component,
                        role=case.role, candidates=50, null_paths=list(collected), top50_equal=True))
        self.native_repeat_max = repeat_max
        return cases

    @terminal
    def mask_probes(self, cases, *, record):
        """Frozen disposable SDPA plumbing probes on the complete fixed16 source.

        These probes never enter an optimizer or checkpoint.  Zero must retain
        exact native ranking/numerical tolerance; the two nonzero probes only
        report their fixed effects and are never selected by their magnitude.
        """
        require(cases and len({c.question_sha256 for c in cases}) == len(cases),
                "fixed complete mask-probe population")
        expected = set(self.population.preflight_components)
        require({c.component for c in cases} == expected, "mask-probe component drift")
        summary = {name: {"chunks": 0, "maximum_absolute_logit_change": 0.0}
                   for name in ("zero", "constant_eighth", "alternating_eighth")}
        for case in cases:
            for chunk in native_chunks(self.tokenizer, case, self.device):
                self.check("preflight")
                prefix = extract_native_prefix(self.model, chunk, complete_native=False)
                native = self.profiler.run(self.final, self.head, prefix, chunk, path="native",
                                           backend_policy=self.candidate_backend)
                try:
                    anchor = native.scores.detach().clone()
                finally:
                    release_rows([native])
                for probe in summary:
                    row = self.profiler.run(self.final, self.head, prefix, chunk,
                                            path="probe", probe=probe,
                                            backend_policy=self.candidate_backend)
                    try:
                        difference = float((row.scores.detach()-anchor).abs().max())
                        require(math.isfinite(difference), "non-finite fixed mask-probe effect")
                        if probe == "zero":
                            same_tensor(row.scores, anchor)
                        item = summary[probe]
                        item["chunks"] += 1
                        item["maximum_absolute_logit_change"] = max(
                            item["maximum_absolute_logit_change"], difference)
                        record(dict(question_sha256=case.question_sha256,
                                    component=case.component,
                                    first_original_index=chunk.original_indices[0],
                                    probe=probe, candidates=len(chunk.original_indices),
                                    maximum_absolute_logit_change=difference,
                                    optimizer_or_checkpoint_effect=False))
                    finally:
                        release_rows([row])
                self.check("preflight")
        require(all(item["chunks"] > 0 for item in summary.values()), "empty mask probe")
        require(all(summary[name]["maximum_absolute_logit_change"] > 0.0
                    for name in ("constant_eighth", "alternating_eighth")),
                "fixed nonzero mask probes did not reach logits")
        return dict(status="FIXED_MASK_PROBES_COMPLETE_NOT_UTILITY",
                    selection="all questions/candidates in fixed16 components",
                    probes=summary, selected_by_effect=False)

    @terminal
    def fixed_probe(self, cases, *, observe, dependency_record, forward_trace,
                    backward_trace):
        """One disposable update plus updated backward; first eligible batch."""
        require(callable(observe) and callable(dependency_record) and
                callable(getattr(forward_trace, "capture", None)) and
                callable(getattr(backward_trace, "before", None)) and
                callable(getattr(backward_trace, "after", None)),
                "fixed probe evidence writers required")
        batch = None
        for start in range(0, len(cases)-1, 2):
            pair = cases[start:start+2]
            tiers = [[y for _, y in self.targets.rows_by_question[c.question_sha256]] for c in pair]
            if any(len(set(y)) > 1 for y in tiers):
                batch = pair
                break
        if batch is None:
            return dict(status="INCONCLUSIVE_NO_ELIGIBLE_FIXED_PROBE", replacement_used=False)
        cache_wall = time.monotonic()
        cache_cpu = time.process_time()
        cached = self.cache(batch, reviewed_only=True, phase="preflight")
        cache_timing = {
            "wall_seconds": time.monotonic()-cache_wall,
            "cpu_seconds": time.process_time()-cache_cpu,
        }
        arms = copy.deepcopy(self.initial_arms)
        optimizers = build_optimizers(arms)
        dependency = None
        counts = []
        initial_timings = []
        forward_trace_records = []
        backward_trace_records = []
        try:
            from applications.rag.qcls_preflight_diagnostics_v001 import ContextualPoolRotation
            dependency = ContextualPoolRotation(self, cases)
            pool_binding = dependency.collect()
            initial_dependency = dependency.evaluate(
                arms, checkpoint="initial", record=dependency_record)
            for name in ARMS:
                for checkpoint in ("initial", "diagnostic_updated"):
                    unit_wall = time.monotonic()
                    unit_cpu = time.process_time()
                    scores, rows, keys = self.forward(name, arms[name], cached, capture=True,
                                                     phase="preflight", checkpoint=checkpoint)
                    try:
                        forward_trace_records.append(
                            forward_trace.capture(name, arms[name], checkpoint))
                        backward_trace.before(checkpoint, name, cached, scores, arms[name])
                        observe("before_backward", checkpoint, name, cached, rows, scores, arms[name])
                        tiers = tuple(q.tiers for q in cached)
                        action = (lambda: update_two_questions(arms[name], optimizers[name], scores, tiers)) \
                            if checkpoint == "initial" else \
                            (lambda: backward_without_update(arms[name], scores, tiers))
                        evidence = self.backward.run(arm_name=name, arm=arms[name], checkpoint=checkpoint,
                            chunks=keys, forward_rows=rows, action=action)
                        backward_trace_records.append(
                            backward_trace.after(checkpoint, name, cached, scores, arms[name]))
                        observe("after_backward", checkpoint, name, cached, rows, scores, arms[name])
                        counts.append(evidence)
                        if checkpoint == "initial":
                            initial_timings.append({
                                "arm": name,
                                "wall_seconds": time.monotonic()-unit_wall,
                                "cpu_seconds": time.process_time()-unit_cpu,
                                "includes": "forward+observability+actual_backward+one_fixed_update",
                            })
                        self.integrity(); self.check("preflight")
                    finally:
                        release_rows(rows)
            require(len(counts) == 10, "all five arms at both probe checkpoints")
            backward_trace_summary = backward_trace.verify_complete()
            updated_dependency = dependency.evaluate(
                arms, checkpoint="diagnostic_updated", record=dependency_record)
            return dict(status="FIXED_PROBE_BACKWARD_OBSERVED_NOT_FULL_PREFLIGHT", records=counts,
                        contextual_pool_binding=pool_binding.__dict__,
                        dependency=(initial_dependency, updated_dependency),
                        exact_forward_trace_records=forward_trace_records,
                        actual_backward_trace_records=backward_trace_records,
                        actual_backward_trace_summary=backward_trace_summary,
                        conservative_evaluation_unit=
                            forward_trace.conservative_evaluation_unit(),
                        measured_training_like_unit={
                            "questions": 2, "arms": list(ARMS), "cache": cache_timing,
                            "per_arm": initial_timings,
                            "wall_seconds": cache_timing["wall_seconds"]+
                                sum(row["wall_seconds"] for row in initial_timings),
                            "cpu_seconds": cache_timing["cpu_seconds"]+
                                sum(row["cpu_seconds"] for row in initial_timings),
                            "screen_steps_projected_from_this_unit": 256,
                            "observer_overhead_included": True,
                        },
                        scientific_qualification_from_this_record=False)
        except BaseException:
            self.failed = True
            raise
        finally:
            if dependency is not None:
                dependency.close()
            optimizers.clear(); cached = (); arms = None

    @terminal
    def train256(self, *, preflight_authorization, record):
        """Only the full reviewed parent may issue this after complete preflight.

        The callback authenticates actual preflight/trace/parity/resource
        receipts; this component does not turn an arbitrary JSON flag into an
        admission. Source remains non-executable until that parent is reviewed.
        """
        require(callable(preflight_authorization), "full-entry preflight authenticator required")
        preflight_authorization()
        require(all(v == 0 for v in self.scheduled_steps.values()), "single screen, no resume/retry")
        self.check("screen_preparation")
        arms = copy.deepcopy(self.initial_arms)
        optimizers = build_optimizers(arms)
        try:
            for step in range(256):
                pair = tuple(self.cases[q] for q in self.population.train_order[2*step:2*step+2])
                require(len(pair) == 2 and all(c.role == "training" for c in pair), "fixed512 complete schedule")
                cached = self.cache(pair, reviewed_only=True, phase="screen_training")
                try:
                    for name in ARMS:
                        scores, rows, keys = self.forward(name, arms[name], cached, capture=False,
                                                         phase="screen_training")
                        try:
                            result = update_two_questions(arms[name], optimizers[name], scores,
                                                          tuple(q.tiers for q in cached))
                            self.scheduled_steps[name] += 1
                            record(dict(step=step+1, arm=name, source_questions=[c.question_sha256 for c in pair],
                                        chunks=list(keys), result=result))
                            self.integrity(); self.check("screen_training")
                        finally:
                            release_rows(rows)
                finally:
                    cached = ()
            require(all(v == 256 for v in self.scheduled_steps.values()), "incomplete final epoch")
            require(module_digest(self.model) == self.frozen_backbone, "frozen backbone changed")
            return arms, optimizers
        except BaseException:
            self.failed = True
            optimizers.clear()
            raise

    @terminal
    def final_predictions(self, arms, *, barrier, checkpoint_digests, record_question=None):
        require(tuple(arms) == ARMS and all(v == 256 for v in self.scheduled_steps.values()),
                "all scheduled final arms required")
        all_rows = {name: [] for name in PREDICTORS}
        for case in barrier.cases:
            cached = self.cache((case,), reviewed_only=False, phase="screen_evaluation")
            try:
                # cache() already compared these exact unforced native scores
                # with math on every chunk. Retain that anchor, avoiding a
                # redundant third native forward without relaxing any check.
                require(cached[0].native_scores is not None, "missing native evaluation anchor")
                predictions = {"frozen_native": cached[0].native_scores}
                for name in ARMS:
                    scores, rows, _ = self.forward(name, arms[name], cached, capture=False,
                                                   phase="screen_evaluation")
                    try:
                        predictions[name] = scores[0].detach()
                    finally:
                        release_rows(rows)
                for name in PREDICTORS:
                    score = predictions[name]
                    all_rows[name].append(dict(question_sha256=case.question_sha256, role=case.role,
                        component_id=case.component, candidate_ids=list(case.candidate_ids),
                        scores=score.cpu().tolist(), ranking=list(rank50(score, case.candidate_ids)),
                        fallback=case.fallback))
                self.integrity(); self.check("screen_evaluation")
                if record_question is not None:
                    record_question(len(all_rows["frozen_native"]),
                        {name: all_rows[name][-1] for name in PREDICTORS})
            finally:
                cached = ()
        for name in PREDICTORS:
            barrier.add(name, tuple(all_rows[name]), checkpoint_sha256=checkpoint_digests[name],
                        scheduled_steps=0 if name == "frozen_native" else self.scheduled_steps[name])
        return dict(status="ALL_PREDICTIONS_WRITTEN_NOT_YET_JOINT_SEALED", predictors=list(PREDICTORS))
