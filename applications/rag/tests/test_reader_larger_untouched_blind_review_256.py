import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/collab/five_ideas"))
import prepare_reader_larger_untouched_blind_review_256 as prepare
import score_reader_larger_untouched_blind_review_256 as scorer
import audit_reader_larger_untouched_blind_review_256 as auditor


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    cases, samples = [], []
    for number in range(4):
        question = f"question {number}?"
        question_sha = hashlib.sha256(question.encode()).hexdigest()
        candidates = [
            {
                "id": f"p{number}-{index}",
                "candidate_id_sha256": hashlib.sha256(f"p{number}-{index}".encode()).hexdigest(),
                "title": f"T{index}",
                "text": f"passage {index}",
                "retrieved_rank": index + 1,
                "retrieval_score": 50 - index,
            }
            for index in range(50)
        ]
        cases.append({"dataset_index": number, "question": question, "question_sha256": question_sha,
                      "source_id_sha256": "hidden", "candidates": candidates})
        quantum = [0, 1, 2, 3, 5] if number < 2 else [0, 1, 2, 3, 4]
        classical = [0, 1, 2, 3, 6] if number == 1 else [0, 1, 2, 3, 4]
        samples.append({
            "prefix": f"unseen_{number}", "question_sha256": question_sha,
            "candidate_id_sha256": [item["candidate_id_sha256"] for item in candidates],
            "candidate_retrieved_ranks": list(range(1, 51)), "input_sha256": "hidden",
            "identity_top5": [0, 1, 2, 3, 4], "quantum_top5": quantum, "classical_top5": classical,
        })
    cohort = tmp_path / "cohort.json"
    write(cohort, {"cases": cases})
    trace = tmp_path / "trace.json"
    write(trace, {
        "run_id": "fixture", "population": {"unseen_transport": 4}, "samples": samples,
        "gate": {"advance_to_blind_review": True},
        "label_boundary": {"answers_read": False, "support_annotations_read": False,
                           "silver_oracle_read": False, "generator_called": False,
                           "semantic_review_started": False},
    })
    config = tmp_path / "config.json"
    write(config, {
        "claim_ceiling": "fixture", "blind_seed": 7,
        "cohort": {"sha256": sha(cohort), "expected_questions": 4},
        "transport_run": {"run_id": "fixture", "trace_sha256": sha(trace),
                          "expected_changed_cases": 2, "expected_review_items": 13},
        "semantic_codes": {"answer_support": ["direct", "partial", "none", "uncertain"],
                           "scope_match": ["yes", "partial", "no", "uncertain"]},
        "prospective_gate": {
            "min_quantum_useful_gain_over_identity": 1,
            "min_quantum_direct_gain_over_identity": 0,
            "min_quantum_useful_gain_over_classical": 1,
            "min_quantum_useful_case_win_margin_over_identity": 1,
            "min_quantum_useful_case_win_margin_over_classical": 1,
            "max_uncertain_quantum_selected_items": 0,
            "bootstrap": {"resamples": 400, "seed": 11, "confidence": 0.95,
                          "interval": "two_sided_percentile"},
        },
        "blinding": {"unblind_only_after": "consensus"},
    })
    return cohort, trace, config


def completed_review(template: dict, direct_items: set[str], reviewer: str) -> dict:
    result = json.loads(json.dumps(template))
    result["reviewer_id"] = reviewer
    result["completed"] = True
    for row in result["judgments"]:
        direct = row["review_item_id"] in direct_items
        row.update(answer_support="direct" if direct else "none",
                   scope_match="yes" if direct else "no",
                   candidate_answer="x" if direct else "",
                   evidence_text="e" if direct else "",
                   rationale="checked independently")
    return result


def test_packet_is_hash_locked_blind_and_separates_key(tmp_path):
    cohort, trace, config = fixture(tmp_path)
    output = tmp_path / "packet"
    manifest = prepare.build_packet(cohort, trace, output, config)
    packet = json.loads((output / "public/blind_packet.json").read_text())
    assert manifest["changed_cases"] == 2
    assert manifest["review_items"] == 13
    assert not (prepare.BANNED_REVIEW_KEYS & prepare.key_names(packet))
    assert (output / "public/review_template.json").exists()
    assert (output / "private/sealed_method_key.json").exists()


def test_consensus_then_unblind_scores_full_population_bootstrap(tmp_path):
    cohort, trace, config = fixture(tmp_path)
    output = tmp_path / "packet"
    prepare.build_packet(cohort, trace, output, config)
    key = json.loads((output / "private/sealed_method_key.json").read_text())
    direct_items = set()
    for case in key["cases"]:
        identity = set(case["selections"]["identity_reader"])
        for index in case["selections"]["quantum"]:
            if index not in identity:
                direct_items.add(next(row["review_item_id"] for row in case["items"]
                                      if row["source_index"] == index))
    template = json.loads((output / "public/review_template.json").read_text())
    review_a, review_b = tmp_path / "a.json", tmp_path / "b.json"
    write(review_a, completed_review(template, direct_items, "a"))
    write(review_b, completed_review(template, direct_items, "b"))
    consensus_path, disputes_path = tmp_path / "consensus.json", tmp_path / "disputes.json"
    consensus = scorer.merge(output / "public/blind_packet.json", review_a, review_b,
                             consensus_path, disputes_path, config)
    assert consensus["complete"] and consensus["unresolved_count"] == 0
    score_path = tmp_path / "score.json"
    result = scorer.score(output / "public/blind_packet.json", consensus_path,
                          output / "private/sealed_method_key.json", score_path, config)
    assert result["population"] == {"questions": 4, "changed_cases": 2, "unchanged_cases": 2}
    assert result["pairwise_full_population"]["quantum_vs_identity"]["useful_total_delta"] == 2
    assert result["pairwise_full_population"]["quantum_vs_identity"]["point_delta_bootstrap"]["clusters"] == 4
    adjudications_path = tmp_path / "adjudications.json"
    write(adjudications_path, {
        "schema_version": "rag.reader_larger_untouched_blind_adjudications.v1",
        "packet_sha256": sha(output / "public/blind_packet.json"),
        "judgments": [],
    })
    audit = auditor.audit(output / "public/blind_packet.json", review_a, review_b,
                          disputes_path, adjudications_path, consensus_path,
                          output / "private/sealed_method_key.json", score_path, config,
                          tmp_path / "audit.json")
    assert audit["status"] == "passed"
    assert audit["review"]["unresolved_items"] == 0


def test_disagreement_blocks_key_until_blind_adjudication(tmp_path):
    cohort, trace, config = fixture(tmp_path)
    output = tmp_path / "packet"
    prepare.build_packet(cohort, trace, output, config)
    template = json.loads((output / "public/review_template.json").read_text())
    review_a = completed_review(template, set(), "a")
    review_b = completed_review(template, set(), "b")
    review_b["judgments"][0]["answer_support"] = "partial"
    review_a_path, review_b_path = tmp_path / "a.json", tmp_path / "b.json"
    write(review_a_path, review_a)
    write(review_b_path, review_b)
    partial_path, disputes_path = tmp_path / "partial.json", tmp_path / "disputes.json"
    partial = scorer.merge(output / "public/blind_packet.json", review_a_path, review_b_path,
                           partial_path, disputes_path, config)
    assert not partial["complete"] and partial["unresolved_count"] == 1
    try:
        scorer.score(output / "public/blind_packet.json", partial_path,
                     output / "private/sealed_method_key.json", tmp_path / "score.json", config)
    except ValueError as error:
        assert "complete blind consensus" in str(error)
    else:
        raise AssertionError("sealed key should remain blocked")
    dispute = json.loads(disputes_path.read_text())["unresolved"][0]
    adjudications_path = tmp_path / "adjudications.json"
    write(adjudications_path, {
        "schema_version": "rag.reader_larger_untouched_blind_adjudications.v1",
        "packet_sha256": sha(output / "public/blind_packet.json"),
        "judgments": [{"review_case_id": dispute["review_case_id"],
                       "review_item_id": dispute["review_item_id"],
                       "answer_support": "none", "scope_match": "no",
                       "rationale": "third blind review"}],
    })
    final = scorer.adjudicate(output / "public/blind_packet.json", partial_path, disputes_path,
                              adjudications_path, tmp_path / "final.json", config)
    assert final["complete"] and final["unresolved_count"] == 0
