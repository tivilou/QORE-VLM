"""Fixed source-dependence diagnostics for the QCLS v004 preflight.

This is a queue-only scientific component, not a standalone experiment or an
admission token.  It sees the frozen complete preflight population but never
training/development labels.  Only normalized contextual question pools are
retained between passes; native hidden states and passage-token rows are
released after each original contiguous candidate chunk.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from applications.rag.qcls_attention_v004 import (
    ARMS, UPSTREAM_ARMS, AnchoredBias, finite, module_digest,
    normalized_inputs, require,
)
from applications.rag.qcls_native_producer_v001 import extract_native_prefix, native_chunks


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class PoolBinding:
    source_manifest_sha256: str
    case_order: tuple[str, ...]
    coordinate_sha256: str
    value_sha256: str
    candidate_rows: int
    retained_bytes: int


class _Effect:
    def __init__(self) -> None:
        self.rows = 0
        self.total = 0.0
        self.maximum = 0.0
        self.digest = hashlib.sha256()

    def add(self, *, coordinate: tuple[str, int], normal, rotated) -> None:
        require(normal.shape == rotated.shape and normal.ndim == 1,
                "dependency critic output layout")
        finite(normal, "normal dependency output")
        finite(rotated, "rotated dependency output")
        difference = (normal.detach()-rotated.detach()).abs().to(device="cpu", dtype=torch.float32)
        count = int(difference.numel())
        require(count > 0, "empty dependency effect")
        self.rows += count
        self.total += float(difference.sum())
        self.maximum = max(self.maximum, float(difference.max()))
        self.digest.update(canonical({"question_sha256": coordinate[0],
                                      "first_original_index": coordinate[1],
                                      "count": count}))
        self.digest.update(difference.contiguous().numpy().tobytes(order="C"))

    def summary(self) -> dict:
        require(self.rows > 0 and math.isfinite(self.total) and math.isfinite(self.maximum),
                "incomplete dependency accumulator")
        return {
            "passage_token_rows": self.rows,
            "mean_absolute_change": self.total/self.rows,
            "maximum_absolute_change": self.maximum,
            "ordered_effect_sha256": self.digest.hexdigest(),
        }


class ContextualPoolRotation:
    """Two-pass, label-free, fixed cyclic contextual-pool diagnostic."""

    def __init__(self, driver, cases):
        require(cases and len(cases) >= 2, "at least two fixed preflight questions required")
        expected_components = set(driver.population.preflight_components)
        expected_cases = tuple(driver.cases[q] for q in driver.population.train_order
                               if driver.cases[q].component in expected_components)
        require(tuple(cases) == expected_cases and
                {case.component for case in cases} == expected_components,
                "complete fixed preflight order required")
        require(len({case.question_sha256 for case in cases}) == len(cases),
                "duplicate fixed preflight question")
        self.driver = driver
        self.cases = tuple(cases)
        self.case_order = tuple(case.question_sha256 for case in cases)
        self.next_question = dict(zip(self.case_order, self.case_order[1:]+self.case_order[:1], strict=True))
        self.pools: dict[tuple[str, int], torch.Tensor] = {}
        self.binding: PoolBinding | None = None
        self.closed = False

    def collect(self) -> PoolBinding:
        require(not self.closed and self.binding is None and not self.pools,
                "contextual pool collection is single-use")
        coordinate = hashlib.sha256()
        values = hashlib.sha256()
        source = None
        retained = 0
        for case in self.cases:
            seen = []
            for chunk in native_chunks(self.driver.tokenizer, case, self.driver.device):
                self.driver.check("preflight")
                prefix = extract_native_prefix(self.driver.model, chunk, complete_native=False)
                qpool, _, _ = normalized_inputs(prefix.hidden, chunk.boundary)
                require(qpool.shape == (len(chunk.original_indices), 768),
                        "contextual pool candidate layout")
                qpool = qpool.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
                for row, original_index in zip(qpool, chunk.original_indices, strict=True):
                    key = (case.question_sha256, original_index)
                    require(key not in self.pools, "duplicate contextual pool coordinate")
                    self.pools[key] = row
                    seen.append(original_index)
                    coordinate.update(canonical({"question_sha256": key[0],
                                                 "original_index": key[1]}))
                    raw = row.numpy().tobytes(order="C")
                    values.update(raw); retained += len(raw)
                if source is None:
                    source = prefix.source_manifest_sha256
                require(prefix.source_manifest_sha256 == source,
                        "contextual pool source manifest drift")
                prefix = qpool = None
                self.driver.check("preflight")
            require(tuple(seen) == tuple(range(50)), "contextual pool complete original50 order")
        require(len(self.pools) == len(self.cases)*50 and source is not None,
                "incomplete contextual pool index")
        self.binding = PoolBinding(
            source_manifest_sha256=source,
            case_order=self.case_order,
            coordinate_sha256=coordinate.hexdigest(),
            value_sha256=values.hexdigest(),
            candidate_rows=len(self.pools),
            retained_bytes=retained,
        )
        return self.binding

    def evaluate(self, arms, *, checkpoint: str, record) -> dict:
        require(not self.closed and self.binding is not None and
                checkpoint in ("initial", "diagnostic_updated"),
                "dependency checkpoint/pool lifecycle")
        require(tuple(arms) == ARMS and callable(record), "fixed dependency arm registry")
        accumulators = {
            name: {"learned": _Effect(), "reference": _Effect()} for name in UPSTREAM_ARMS
        }
        constants = {}
        arm_state = {}
        with torch.no_grad():
            for name in UPSTREAM_ARMS:
                arm = arms[name]
                require(type(arm) is AnchoredBias, "dependency upstream arm type")
                arm.verify_reference()
                zero = torch.zeros((1, 4), dtype=torch.float32, device=self.driver.device)
                learned = arm.learned.critic(zero)
                reference = arm.reference.critic(zero)
                finite(learned, "learned critic at zero")
                finite(reference, "reference critic at zero")
                require(learned.shape == reference.shape == (1,), "zero critic output layout")
                constants[name] = {"learned": float(learned[0]), "reference": float(reference[0])}
                arm_state[name] = module_digest(arm)
            for case in self.cases:
                rotated_question = self.next_question[case.question_sha256]
                for chunk in native_chunks(self.driver.tokenizer, case, self.driver.device):
                    self.driver.check("preflight")
                    prefix = extract_native_prefix(self.driver.model, chunk, complete_native=False)
                    _, native_q, passage = normalized_inputs(prefix.hidden, chunk.boundary)
                    _, passage_mask = chunk.boundary.masks()
                    row = passage_mask.nonzero(as_tuple=True)[0]
                    rotated_pool = torch.stack([
                        self.pools[(rotated_question, index)] for index in chunk.original_indices
                    ]).to(device=self.driver.device, dtype=torch.float32)
                    rotated_q = rotated_pool.index_select(0, row)
                    require(rotated_q.shape == native_q.shape and
                            rotated_q.shape[0] == passage.shape[0],
                            "rotated contextual pool/passages layout")
                    for name in UPSTREAM_ARMS:
                        arm = arms[name]
                        learned_native, _ = arm.learned(native_q, passage)
                        learned_rotated, _ = arm.learned(rotated_q, passage)
                        reference_native, _ = arm.reference(native_q, passage)
                        reference_rotated, _ = arm.reference(rotated_q, passage)
                        coordinate = (case.question_sha256, chunk.original_indices[0])
                        accumulators[name]["learned"].add(
                            coordinate=coordinate, normal=learned_native, rotated=learned_rotated)
                        accumulators[name]["reference"].add(
                            coordinate=coordinate, normal=reference_native, rotated=reference_rotated)
                    prefix = native_q = passage = rotated_pool = rotated_q = None
                    self.driver.check("preflight")
        rows = []
        for name in UPSTREAM_ARMS:
            row = {
                "schema": "qcls.contextual-pool-dependency.v1",
                "checkpoint": checkpoint,
                "arm": name,
                "source_manifest_sha256": self.binding.source_manifest_sha256,
                "pool_coordinate_sha256": self.binding.coordinate_sha256,
                "pool_value_sha256": self.binding.value_sha256,
                "permutation": "next_question_in_complete_fixed_preflight_order_same_candidate_index",
                "constant_angles": constants[name],
                "learned_cyclic_effect": accumulators[name]["learned"].summary(),
                "reference_cyclic_effect": accumulators[name]["reference"].summary(),
                "arm_state_sha256": arm_state[name],
                "labels_read": 0,
                "utility_claim": False,
                "isolated_query_causality_claim": False,
            }
            record(row); rows.append(row)
        result = {
            "status": "FIXED_CONTEXTUAL_POOL_DEPENDENCY_COMPLETE_NOT_UTILITY",
            "checkpoint": checkpoint,
            "binding": self.binding.__dict__,
            "upstream_arms": list(UPSTREAM_ARMS),
            "not_applicable": ["frozen_native", "disabled", "HEAD768_active"],
            "records_sha256": hashlib.sha256(canonical(rows)).hexdigest(),
        }
        return result

    def close(self) -> None:
        self.pools.clear()
        self.closed = True
