from __future__ import annotations

import copy

import numpy as np
import pytest

from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head
from applications.rag.reader_larger_untouched_transport import (
    build_trace,
    candidate_text,
    larger_transport_gate,
    replay_head_vectors,
    restore_rotation_head,
    score_heads,
    validate_trace,
    verification_spec_from_manifest,
)
from applications.rag.reader_unseen_transport import transport_metrics


def manifest(cases: int = 2) -> dict:
    return {
        "manifest_identity_sha256": "a" * 64,
        "cohort": {"cases": cases},
        "head": {"layers": 2, "conditioning": 4, "seed": 20261013},
        "transport_gate": {
            "min_quantum_changed_cases_for_review": 1,
            "min_quantum_classical_different_cases_for_review": 1,
        },
        "label_boundary": {
            "answers_read": False,
            "support_annotations_read": False,
            "silver_oracle_read": False,
            "generator_called": False,
            "semantic_review_started": False,
        },
    }


def checkpoint(tmp_path):
    import torch

    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(768, generator=generator)
    bias = torch.tensor([0.25], dtype=torch.float32)
    arrays = {}
    for family, offset in (("quantum", 0.02), ("classical", -0.015)):
        head = make_rotation_head(
            weight, bias, family=family, config=RotationConfig(layers=2, conditioning=4, seed=20261013)
        )
        for name, value in head.state_dict().items():
            arrays[f"{family}__head__{name}"] = value.detach().cpu().numpy()
        final = np.full((40, 5), offset, dtype=np.float64)
        final[:, 1:] *= np.linspace(0.1, 0.4, 4)
        arrays[f"{family}__final_theta"] = final
    path = tmp_path / "sequential_values.npz"
    np.savez(path, **arrays)
    return path, weight.numpy(), bias.numpy()


def test_candidate_text_and_prospective_gate():
    assert candidate_text({"title": "T", "text": " body "}) == "T. body"
    assert candidate_text({"title": "", "text": "body"}) == "body"
    with pytest.raises(ValueError, match="non-empty"):
        candidate_text({"text": " "})
    result = larger_transport_gate(
        {"candidate_changed_questions": 1, "candidate_control_different_questions": 1}, manifest()
    )
    assert result["advance_to_blind_review"] is True


def test_frozen_manifest_archive_name_is_explicitly_bound():
    contract = manifest()
    contract.update({
        "provenance": {"cohort_source_snapshot": {"file": "executed_sources.zip"}},
        "sequential_checkpoint": {"source_snapshot": {}},
    })
    contract["cohort"]["source_snapshot"] = {}
    rebound = verification_spec_from_manifest(contract)
    assert rebound["cohort"]["source_snapshot"]["file"] == "executed_sources.zip"
    assert rebound["sequential_checkpoint"]["source_snapshot"]["file"] == "executed_sources.zip"
    assert "file" not in contract["sequential_checkpoint"]["source_snapshot"]
    crossed = copy.deepcopy(contract)
    crossed["provenance"]["cohort_source_snapshot"]["file"] = "unexpected.zip"
    with pytest.raises(ValueError, match="filename"):
        verification_spec_from_manifest(crossed)


def test_complete_trace_and_independent_vector_replay(tmp_path):
    import torch

    path, weight, bias = checkpoint(tmp_path)
    contract = manifest()
    with np.load(path, allow_pickle=False) as values:
        quantum = restore_rotation_head(values, "quantum", contract, weight, bias, device=torch.device("cpu"))
        classical = restore_rotation_head(values, "classical", contract, weight, bias, device=torch.device("cpu"))
    rng = np.random.default_rng(11)
    cls = rng.normal(size=(2, 50, 768)).astype(np.float32)
    pooled = rng.normal(size=(2, 50, 3840)).astype(np.float32)
    identity = np.stack([
        torch.nn.functional.linear(
            torch.from_numpy(row), torch.from_numpy(weight).reshape(1, -1), torch.from_numpy(bias).reshape(1)
        ).squeeze(-1).numpy()
        for row in cls
    ]).astype(np.float32)
    quantum_scores, classical_scores = score_heads(
        cls, pooled, quantum, classical, device=torch.device("cpu")
    )
    numeric = {
        "identity_scores": identity,
        "cls": cls,
        "pooled": pooled,
        "quantum_scores": quantum_scores,
        "classical_scores": classical_scores,
    }
    metrics = transport_metrics(identity, quantum_scores, classical_scores)
    gate = larger_transport_gate(metrics, contract)
    records = [
        {
            "prefix": f"unseen_{index + 1:03d}",
            "question_sha256": str(index) * 64,
            "candidate_id_sha256": [f"{number:064x}" for number in range(50)],
            "candidate_retrieved_ranks": list(range(1, 51)),
            "input_sha256": f"{index + 2:064x}",
        }
        for index in range(2)
    ]
    trace = build_trace(
        run_id="synthetic",
        records=records,
        values=numeric,
        values_sha256="b" * 64,
        manifest=contract,
        metrics=metrics,
        gate=gate,
        provenance={"fixture": True},
        synthetic_only=True,
    )
    validated = validate_trace(trace, numeric, contract, expected_records=records)
    assert validated["cases_replayed"] == 2
    replay = replay_head_vectors(
        values=numeric,
        sequential_values=path,
        manifest=contract,
        device=torch.device("cpu"),
    )
    assert replay["identity"]["top5_exact"] is True
    assert replay["quantum"]["top5_exact"] is True
    assert replay["classical"]["top5_exact"] is True

    crossed = copy.deepcopy(trace)
    crossed["label_boundary"]["answers_read"] = True
    with pytest.raises(ValueError, match="label boundary"):
        validate_trace(crossed, numeric, contract, expected_records=records)
