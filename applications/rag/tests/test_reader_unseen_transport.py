import argparse
import json
from pathlib import Path

import numpy as np
import torch

from applications.rag.quantum_reader_rotation import RotationConfig, make_rotation_head
from applications.rag.reader_unseen_transport import evaluate_transport_gate, transport_metrics
from scripts.collab.five_ideas.audit_reader_unseen_transport_gate import audit
from scripts.collab.five_ideas.run_reader_unseen_transport_gate import run


def test_transport_metrics_and_gate_are_label_free():
    identity = np.tile(np.arange(50, 0, -1, dtype=float), (7, 1))
    quantum = identity.copy()
    classical = identity.copy()
    for i in range(5):
        quantum[i, 9] = 100.0 + i
    for i in range(3):
        classical[i, 10] = 100.0 + i
    quantum[5, 11] = 99.0
    metrics = transport_metrics(identity, quantum, classical)
    assert metrics["candidate_changed_questions"] == 6
    assert metrics["candidate_control_different_questions"] >= 3
    gate = evaluate_transport_gate(metrics, {"transport_gate": {"min_candidate_changed_questions": 5, "min_candidate_control_different_questions": 3}})
    assert gate["advance_to_blind_review"]


def test_no_transport_does_not_advance():
    scores = np.tile(np.arange(50, 0, -1, dtype=float), (7, 1))
    metrics = transport_metrics(scores, scores.copy(), scores.copy())
    gate = evaluate_transport_gate(metrics, {"transport_gate": {"min_candidate_changed_questions": 1, "min_candidate_control_different_questions": 1}})
    assert not gate["advance_to_blind_review"]


def test_synthetic_full_trace_replays(tmp_path: Path):
    torch.manual_seed(29)
    features = tmp_path / "features"
    features.mkdir()
    hidden = 8
    weight = torch.randn(hidden)
    bias = torch.tensor(0.2)
    arrays = {}
    records = []
    for i in range(87):
        cls = torch.randn(50, hidden)
        pooled = torch.randn(50, 5 * hidden)
        pooled[:, :hidden] = cls
        base = cls @ weight + bias
        prefix = f"validation_{i+1:03d}"
        arrays[prefix + "__base"] = base.numpy()
        arrays[prefix + "__cls"] = cls.numpy()
        arrays[prefix + "__pooled"] = pooled.numpy()
        order = sorted(range(50), key=lambda j: (-float(base[j]), j))[:5]
        records.append({"prefix": prefix, "number": i + 1, "question_sha256": f"{i:064x}",
                        "candidate_id_sha256": [f"{i * 50 + j:064x}" for j in range(50)],
                        "candidate_retrieved_ranks": list(range(1, 51)), "baseline_top5_indices": order,
                        "input_sha256": f"{i + 1:064x}", "lane": "validation"})
    np.savez(features / "reader_features.npz", **arrays)
    np.save(features / "relevance_weight.npy", weight.numpy())
    np.save(features / "relevance_bias.npy", bias.numpy())
    (features / "cache_index.json").write_text(json.dumps({"records": records, "label_boundary": {"answers_read": False, "support_annotations_read": False, "silver_oracle_read": False}}))
    sequential = tmp_path / "sequential_values.npz"
    saved = {}
    for family, value in (("quantum", 0.03), ("classical", -0.02)):
        head = make_rotation_head(weight, bias, family=family, config=RotationConfig(layers=2, conditioning=4, seed=20261013))
        for name, tensor in head.state_dict().items():
            saved[f"{family}__head__{name}"] = tensor.numpy()
        saved[f"{family}__final_theta"] = np.full(head.theta.shape, value, dtype=np.float32)
    np.savez(sequential, **saved)
    output = tmp_path / "runs"
    result = run(argparse.Namespace(features=features, sequential_values=sequential, output=output, synthetic=True))
    replay = audit(result)
    assert replay["status"] == "passed"
    assert replay["score_vectors_replayed"] == 261
