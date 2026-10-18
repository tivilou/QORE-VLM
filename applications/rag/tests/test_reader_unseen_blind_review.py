import hashlib
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/collab/five_ideas"))
import prepare_reader_unseen_blind_review as prepare
import score_reader_unseen_blind_review as scorer


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    candidates = [{"id": f"p{i}", "title": f"T{i}", "text": f"passage {i}", "reader_score": 50-i,
                   "retrieved_rank": i+1} for i in range(50)]
    cases = []
    samples = []
    for number in range(2):
        question = f"question {number}?"
        qsha = hashlib.sha256(question.encode()).hexdigest()
        cases.append({"question": question, "question_sha256": qsha, "answers": ["secret"], "future_role": "validation",
                      "candidates": candidates, "reader_top5_indices": [0, 1, 2, 3, 4]})
        qtop = [0, 1, 2, 3, 5] if number == 0 else [0, 1, 2, 3, 4]
        samples.append({"question_sha256": qsha,
                        "candidate_id_sha256": [hashlib.sha256(str(c["id"]).encode()).hexdigest() for c in candidates],
                        "identity_top5": [0, 1, 2, 3, 4], "baseline_top5_indices": [0, 1, 2, 3, 4],
                        "quantum_top5": qtop, "classical_top5": [0, 1, 2, 3, 4]})
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("source.json", json.dumps({"cases": cases}))
    trace = tmp_path / "trace.json"
    write(trace, {"run_id": "fixture", "population": {"validation": 2}, "samples": samples,
                  "gate": {"advance_to_blind_review": True}, "label_boundary": {"answers_read": False}})
    config = tmp_path / "config.json"
    write(config, {"claim_ceiling": "fixture", "source_bundle_sha256": sha(bundle), "blind_seed": 7,
                   "transport_run": {"run_id": "fixture", "trace_sha256": sha(trace), "expected_validation_cases": 2,
                                     "expected_changed_cases": 1},
                   "semantic_codes": {"answer_support": ["direct", "partial", "none", "uncertain"],
                                      "scope_match": ["yes", "partial", "no", "uncertain"]},
                   "prospective_gate": {"min_quantum_useful_gain_over_identity": 1,
                                        "min_quantum_direct_gain_over_identity": 0,
                                        "min_quantum_useful_gain_over_classical": 1,
                                        "min_quantum_case_win_margin_over_identity": 1,
                                        "max_uncertain_selected_items": 0},
                   "blinding": {"unblind_only_after": "consensus"}})
    return bundle, trace, config


def test_packet_has_no_answer_score_rank_or_method_leak(tmp_path):
    bundle, trace, config = fixture(tmp_path)
    out = tmp_path / "packet"
    manifest = prepare.build_packet(bundle, trace, out, config)
    packet = json.loads((out / "blind_packet.json").read_text())
    assert manifest["changed_cases"] == 1
    assert manifest["review_items"] == 6
    assert not (prepare.BANNED_REVIEW_KEYS & prepare.key_names(packet))
    assert "secret" not in json.dumps(packet)
    assert json.loads((out / "sealed_method_key.json").read_text())["packet_sha256"] == sha(out / "blind_packet.json")


def completed_review(template: dict, direct_item: str) -> dict:
    result = json.loads(json.dumps(template))
    result["reviewer_id"] = "reviewer"
    result["completed"] = True
    for row in result["judgments"]:
        is_direct = row["review_item_id"] == direct_item
        row.update(answer_support="direct" if is_direct else "none", scope_match="yes" if is_direct else "no",
                   candidate_answer="x" if is_direct else "", evidence_text="e" if is_direct else "", rationale="checked")
    return result


def test_merge_stays_blind_and_score_opens_key_only_after_consensus(tmp_path):
    bundle, trace, config = fixture(tmp_path)
    packet_dir = tmp_path / "packet"
    prepare.build_packet(bundle, trace, packet_dir, config)
    key = json.loads((packet_dir / "sealed_method_key.json").read_text())
    case = key["cases"][0]
    quantum_only_index = next(i for i in case["selections"]["quantum"] if i not in case["selections"]["identity_reader"])
    item_id = next(row["review_item_id"] for row in case["items"] if row["source_index"] == quantum_only_index)
    template = json.loads((packet_dir / "review_template.json").read_text())
    a = completed_review(template, item_id); b = completed_review(template, item_id)
    review_a, review_b = tmp_path / "a.json", tmp_path / "b.json"
    write(review_a, a); write(review_b, b)
    consensus_path, disputes_path = tmp_path / "consensus.json", tmp_path / "disputes.json"
    consensus = scorer.merge(packet_dir / "blind_packet.json", review_a, review_b, consensus_path, disputes_path, config)
    assert consensus["complete"] and consensus["unresolved_count"] == 0
    result = scorer.score(packet_dir / "blind_packet.json", consensus_path, packet_dir / "sealed_method_key.json",
                          tmp_path / "score.json", config)
    assert result["advance_to_larger_untouched_utility_study"]
    assert result["totals_on_changed_cases"]["quantum"]["direct"] == 1
    assert result["totals_on_changed_cases"]["identity_reader"]["direct"] == 0


def test_disagreement_blocks_unblinding(tmp_path):
    bundle, trace, config = fixture(tmp_path)
    packet_dir = tmp_path / "packet"
    prepare.build_packet(bundle, trace, packet_dir, config)
    template = json.loads((packet_dir / "review_template.json").read_text())
    first = template["judgments"][0]["review_item_id"]
    a = completed_review(template, first); b = completed_review(template, first)
    b["judgments"][0]["answer_support"] = "partial"
    review_a, review_b = tmp_path / "a.json", tmp_path / "b.json"
    write(review_a, a); write(review_b, b)
    consensus_path = tmp_path / "consensus.json"
    consensus = scorer.merge(packet_dir / "blind_packet.json", review_a, review_b, consensus_path, tmp_path / "disputes.json", config)
    assert not consensus["complete"] and consensus["unresolved_count"] == 1
    try:
        scorer.score(packet_dir / "blind_packet.json", consensus_path, packet_dir / "sealed_method_key.json",
                     tmp_path / "score.json", config)
    except ValueError as error:
        assert "complete blind consensus" in str(error)
    else:
        raise AssertionError("unblinding should be blocked")
