"""Joint final-prediction barrier before any development label decoding.

New source only. No CLI, model loading, execution permission, arbitrary label
campaign, historical scoring replay or task admission is supplied here.
"""
from __future__ import annotations

import math
import functools
import json
from dataclasses import asdict

from applications.rag.qcls_artifacts_v001 import Receipt
from applications.rag.qcls_b1_resolver_v001 import (
    BoundPopulation, digest, require, resolve_development_targets,
)


PREDICTORS = ("frozen_native", "MLP24", "PRODUCT24_nominal", "TRIG24", "QCLS24", "HEAD768_active")


def terminal(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.failed = True
            raise
    return call


class JointPredictionBarrier:
    """Fixed complete 6x128x50 final predictions; no labels enter this class."""
    def __init__(self, artifacts, population):
        require(type(population) is BoundPopulation and len(population.cases) == 640,
                "authenticated original population required")
        self.artifacts, self.population = artifacts, population
        self.cases = tuple(case for case in population.cases if case.role == "development")
        require(len(self.cases) == 128 and len({c.question_sha256 for c in self.cases}) == 128,
                "original development denominator")
        self.rows = {}
        self.receipts = {}
        self.seal_receipt = None
        self.failed = False

    @terminal
    def add(self, arm, rows, *, checkpoint_sha256, scheduled_steps):
        require(not self.failed, "prediction barrier terminal after failure")
        try:
            require(self.seal_receipt is None and arm in PREDICTORS and arm not in self.rows,
                    "unregistered/duplicate/post-seal predictor")
            require(type(scheduled_steps) is int and scheduled_steps == (0 if arm == "frozen_native" else 256),
                    "incomplete epoch cannot become final")
            require(isinstance(checkpoint_sha256, str) and len(checkpoint_sha256) == 64 and
                    all(c in "0123456789abcdef" for c in checkpoint_sha256), "final checkpoint binding")
            require(isinstance(rows, tuple) and len(rows) == 128, "all original development questions")
            fixed = []
            for row, case in zip(rows, self.cases, strict=True):
                require(set(row) == {"question_sha256", "role", "component_id", "candidate_ids", "scores",
                                     "ranking", "fallback"}, "prediction field whitelist")
                require(row["question_sha256"] == case.question_sha256 and row["role"] == "development" and
                        row["component_id"] == case.component and
                        tuple(row["candidate_ids"]) == case.candidate_ids and
                        type(row["fallback"]) is bool and row["fallback"] == case.fallback,
                        "prediction source/role/order identity")
                scores = row["scores"]
                require(type(scores) is list and len(scores) == 50 and
                        all(type(v) is float and math.isfinite(v) for v in scores), "complete finite50 scores")
                ranked = tuple(sorted(range(50), key=lambda i: (-scores[i], i, str(case.candidate_ids[i]))))
                require(tuple(row["ranking"]) == ranked, "original exact ordering/ties")
                # Copy immutable scalar containers; later caller mutations cannot
                # alter the identity/rows retained behind the byte barrier.
                fixed.append(dict(question_sha256=case.question_sha256, role=case.role,
                    component_id=case.component, candidate_ids=list(case.candidate_ids),
                    scores=list(scores), ranking=list(ranked), fallback=case.fallback))
            name = "final-predictions-"+arm.lower()+".json"
            receipt = self.artifacts.json(name, dict(schema="qcls.final-predictions.v1", arm=arm,
                source_manifest_sha256=self.population.source_manifest_sha256,
                checkpoint_sha256=checkpoint_sha256, scheduled_steps=scheduled_steps, rows=fixed))
            self.rows[arm] = fixed
            self.receipts[arm] = receipt
            return receipt
        except BaseException:
            self.failed = True
            raise

    @terminal
    def seal(self, *, preflight_receipt, final_checkpoint_receipt):
        require(not self.failed, "prediction barrier terminal after failure")
        try:
            require(self.seal_receipt is None and tuple(self.rows) == PREDICTORS and
                    tuple(self.receipts) == PREDICTORS, "all required predictors before joint seal")
            self.seal_receipt = self.artifacts.seal("joint-final-prediction-seal.json",
                tuple(self.receipts.values())+(preflight_receipt, final_checkpoint_receipt),
                dict(kind="all_predictions_before_any_development_label_decode", predictors=list(PREDICTORS),
                     questions_per_predictor=128, candidates_per_question=50,
                     source_manifest_sha256=self.population.source_manifest_sha256,
                     preflight_receipt=asdict(preflight_receipt),
                     final_checkpoint_receipt=asdict(final_checkpoint_receipt),
                     development_labels_decoded_by_barrier=0))
            return self.seal_receipt
        except BaseException:
            self.failed = True
            raise

    @terminal
    def verify(self):
        require(not self.failed and type(self.seal_receipt) is Receipt and
                tuple(self.receipts) == PREDICTORS, "joint final prediction seal is not complete")
        try:
            self.artifacts._owner_check()
            sealed = json.loads(self.artifacts.read(self.seal_receipt))
            require(sealed["schema"] == "qcls.explicit-artifact-seal.v1", "joint seal schema")
            meta = sealed["metadata"]
            require(meta["kind"] == "all_predictions_before_any_development_label_decode" and
                    tuple(meta["predictors"]) == PREDICTORS and meta["questions_per_predictor"] == 128 and
                    meta["candidates_per_question"] == 50 and
                    meta["source_manifest_sha256"] == self.population.source_manifest_sha256 and
                    meta["development_labels_decoded_by_barrier"] == 0, "joint completion identity")
            frozen = tuple(Receipt(**row) for row in sealed["files"])
            require(len(frozen) == 8 and len({r.relative_path for r in frozen}) == 8 and
                    frozen[-2:] == (Receipt(**meta["preflight_receipt"]),
                                    Receipt(**meta["final_checkpoint_receipt"])), "joint required artifacts")
            require(tuple(r.relative_path for r in frozen[:6]) ==
                    tuple("final-predictions-"+name.lower()+".json" for name in PREDICTORS),
                    "joint predictor filenames/order")
            # Consume the exact bytes authenticated by the frozen joint seal,
            # never a mutable replacement receipt or a verify/reopen pair.
            authenticated = {}
            for arm, receipt in zip(PREDICTORS, frozen[:6], strict=True):
                require(self.receipts[arm] == receipt, "predictor receipt replaced after joint seal")
                payload = json.loads(self.artifacts.read(receipt))
                require(payload["arm"] == arm and payload["rows"] == self.rows[arm] and
                        payload["source_manifest_sha256"] == self.population.source_manifest_sha256,
                        "retained predictions changed after seal")
                authenticated[arm] = payload["rows"]
            for receipt in frozen[-2:]:
                self.artifacts.read(receipt)
            return authenticated
        except BaseException:
            self.failed = True
            raise


def join_sealed_development(barrier, *, bundle_path, bundle_sha256):
    """Only this post-barrier API may open the1839-row dev target bundle."""
    require(type(barrier) is JointPredictionBarrier, "exact reviewed prediction barrier")
    try:
        return _join(barrier, bundle_path=bundle_path, bundle_sha256=bundle_sha256)
    except BaseException:
        barrier.failed = True
        raise


def selected_positions(row):
    indices = range(5) if row["fallback"] else row["ranking"][:5]
    return tuple((row["question_sha256"], i) for i in indices)


def _join(barrier, *, bundle_path, bundle_sha256):
    predictions = barrier.verify()  # Must finish before opening the dev bundle.
    population = barrier.population
    development = resolve_development_targets(
        population, bundle_path=bundle_path, bundle_sha256=bundle_sha256)
    targets = development.positions
    rows = {}
    for arm in PREDICTORS:
        useful = direct = direct_questions = reviewed = unreviewed = fully_reviewed_questions = 0
        for row in predictions[arm]:
            selected = selected_positions(row)
            labels = [targets.get(position) for position in selected]
            reviewed += sum(label is not None for label in labels)
            unreviewed += sum(label is None for label in labels)
            fully_reviewed_questions += int(all(label is not None for label in labels))
            useful += sum(label is not None and label > 0 for label in labels)
            direct += sum(label == 2 for label in labels)
            direct_questions += int(any(label == 2 for label in labels))
        rows[arm] = dict(reviewed_selected=reviewed, unreviewed_selected=unreviewed,
                         fully_reviewed_questions=fully_reviewed_questions, denominator_questions=128,
                         denominator_selected=640, observed_useful=useful, observed_direct=direct,
                         observed_direct_questions=direct_questions, complete_metrics=unreviewed == 0)
    paired = {}
    for arm in PREDICTORS[1:]:
        shared = []
        for native, candidate in zip(predictions["frozen_native"], predictions[arm], strict=True):
            require(native["question_sha256"] == candidate["question_sha256"], "paired source order")
            original = [targets.get(key) for key in selected_positions(native)]
            selected = [targets.get(key) for key in selected_positions(candidate)]
            if any(y is None for y in original+selected):
                continue
            shared.append((sum(y > 0 for y in original), sum(y > 0 for y in selected),
                           sum(y == 2 for y in original), sum(y == 2 for y in selected),
                           int(2 in original), int(2 in selected)))
        paired[arm] = dict(complete_shared_questions=len(shared), original_question_denominator=128,
            native_useful=sum(r[0] for r in shared), candidate_useful=sum(r[1] for r in shared),
            native_direct=sum(r[2] for r in shared), candidate_direct=sum(r[3] for r in shared),
            native_direct_questions=sum(r[4] for r in shared), candidate_direct_questions=sum(r[5] for r in shared),
            useful_case_wins=sum(r[1] > r[0] for r in shared),
            useful_case_ties=sum(r[1] == r[0] for r in shared),
            useful_case_losses=sum(r[1] < r[0] for r in shared),
            complete_population_utility_claimed=False)
    complete = all(row["complete_metrics"] for row in rows.values())
    catastrophe = None
    if complete:
        native = rows["frozen_native"]
        catastrophe = {arm: native["observed_useful"]-rows[arm]["observed_useful"] > 13 or
            native["observed_direct_questions"]-rows[arm]["observed_direct_questions"] > 3
            for arm in PREDICTORS[1:]}
    # Re-verify the prediction barrier after the entire target join as well.
    barrier.verify()
    return dict(schema="qcls.sealed-development-evaluation.v1", prediction_seal=asdict(barrier.seal_receipt),
        source_manifest_sha256=population.source_manifest_sha256,
        development_target_manifest_sha256=development.target_manifest_sha256,
        target_files=list(development.provenance),
        target_mapping_sha256=digest(sorted((q, i, y) for (q, i), y in targets.items())),
        population="existing exposed development; exploratory only", results=rows,
        paired_complete_shared_questions=paired,
        all_required_selections_reviewed=complete, catastrophe=catastrophe,
        qualification_status="CATASTROPHE_STOP" if complete and any(catastrophe.values()) else
            "DESCRIPTIVE_COMPLETE_NOT_INDEPENDENT_CONFIRMATION" if complete else
            "INCONCLUSIVE_INCOMPLETE_SELECTED_LABEL_COVERAGE",
        real_precision_claimed=False, unseen_confirmation_claimed=False)
