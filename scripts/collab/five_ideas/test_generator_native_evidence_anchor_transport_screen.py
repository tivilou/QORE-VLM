from __future__ import annotations

import copy

import pytest

from scripts.collab.five_ideas import run_generator_native_evidence_anchor_transport_screen as screen


def _diagnostic(fallback: bool = False, located_count: int = 1) -> dict:
    return {
        "candidate_id": screen.CANDIDATE_ID,
        "plugin_version": screen.PLUGIN_VERSION,
        "boundary_status": "wrapped" if not fallback else "unwrapped_disabled_fallback",
        "decision": {"requested_arm": "reader", "effective_arm": "disabled" if fallback else "reader", "fallback": fallback, "reason": "reader_span_empty" if fallback else None},
        "provider": {"provider_id": "frozen_dpr_reader_hypotheses_v1", "provider_status": "empty" if fallback else "ok", "passage_count": 5, "hypothesis_count": located_count, "located_count": located_count, "rejected_count": 0, "span_count": located_count, "gold_used": False, "answer_labels_used": False, "evaluator_used": False, "generation_output_used": False},
        "mapping": None,
        "hook_call_count": 0 if fallback else 2,
        "hook_prefill_seen": not fallback,
        "raw_content_persisted": False,
    }


def _result(reader_f1: float = 0.4, control_f1: float = 0.2) -> dict:
    samples = []
    for index in range(50):
        samples.append({
            "question_id": f"nq_{index}",
            "retrieval_hit": True,
            "selected_hit": True,
            "selected_context_K": 5,
            "selected_context_parity": True,
            "selection_time_ms": 1.0,
            "arms": {
                "baseline": {"em": 0.0, "f1": 0.1, "generation_time_ms": 1.0, "pipeline_time_ms": 2.0, "selected_hit": True, "diagnostic": {"mode": "frozen_generator", "fallback": False}},
                "disabled": {"em": 0.0, "f1": 0.1, "generation_time_ms": 1.0, "pipeline_time_ms": 2.0, "selected_hit": True, "diagnostic": _diagnostic(True, 0)},
                "reader": {"em": 1.0, "f1": reader_f1, "generation_time_ms": 1.2, "pipeline_time_ms": 2.4, "selected_hit": True, "diagnostic": _diagnostic(False)},
                "control": {"em": 0.0, "f1": control_f1, "generation_time_ms": 1.2, "pipeline_time_ms": 2.4, "selected_hit": True, "diagnostic": _diagnostic(False)},
            },
        })
    return {"schema_version": 1, "phase": screen.CANDIDATE_ID + "_screen", "stage": "screen", "diagnostic_only": True, "selection_mutation": False, "report_only": True, "config": {}, "samples": samples}


def test_summary_is_compact_and_positive_signal_is_predeclared():
    result = _result()
    summary = screen._summarize(result, {"bootstrap_repetitions": 200, "bootstrap_seed": 10501, "minimum_reader_span_coverage": 0.5, "reader_baseline_harm_ci95_low_min": -0.05, "reader_control_f1_delta_min": 0.0, "positive_reader_control_f1_delta_min": 0.02, "max_pipeline_cost_ratio": 2.0})
    assert summary["decision"] == "screen_positive_signal"
    assert summary["reader_span_coverage"] == 1.0
    assert not screen._forbidden(result)


def test_normal_null_is_inconclusive_and_forbidden_content_is_rejected():
    result = _result(reader_f1=0.2, control_f1=0.2)
    summary = screen._summarize(result, {"bootstrap_repetitions": 200, "bootstrap_seed": 10501, "minimum_reader_span_coverage": 0.5, "reader_baseline_harm_ci95_low_min": -0.05, "reader_control_f1_delta_min": 0.0, "positive_reader_control_f1_delta_min": 0.02, "max_pipeline_cost_ratio": 2.0})
    assert summary["decision"] == "screen_inconclusive"
    bad = copy.deepcopy(result)
    bad["samples"][0]["question"] = "must not persist"
    assert screen._forbidden(bad)


def test_config_validation_does_not_access_task_data(tmp_path):
    config = screen.PROJECT_ROOT / "configs/experiments/generator_native_evidence_anchor_transport_screen.yaml"
    phase = screen._load_config(config)
    assert phase["dataset"]["sample_offset"] == 1900
    assert phase["dataset"]["max_samples"] == 50
