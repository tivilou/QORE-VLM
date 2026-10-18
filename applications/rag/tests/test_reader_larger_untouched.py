import argparse
import json
from pathlib import Path
import zipfile

from applications.rag.reader_larger_untouched import (
    exclusion_question_hashes,
    file_sha256,
    select_fresh_questions,
    sha256_text,
    validate_cohort,
)
from scripts.collab.five_ideas.prepare_reader_larger_untouched_cohort_256 import run


def test_frozen_hash_selection_is_deterministic_and_excludes_history():
    rows = [{"id": str(i), "question": f"question {i}?"} for i in range(40)]
    excluded = {sha256_text("question 17?"), sha256_text("question 19?")}
    left = select_fresh_questions(rows, excluded, pool_start=10, pool_end=40, count=12, seed=7)
    right = select_fresh_questions(rows, excluded, pool_start=10, pool_end=40, count=12, seed=7)
    assert left == right
    assert len(left) == 12
    assert not ({row["question_sha256"] for row in left} & excluded)
    assert all(10 <= row["dataset_index"] < 40 for row in left)


def test_exclusion_sources_are_hash_locked_and_question_only(tmp_path: Path):
    support = tmp_path / "support.zip"
    with zipfile.ZipFile(support, "w") as archive:
        archive.writestr("source.json", json.dumps({"cases": [
            {"question": "support one?", "question_sha256": sha256_text("support one?"), "answers": ["secret"]},
            {"question": "support two?", "question_sha256": sha256_text("support two?"), "answers": ["secret"]},
        ]}))
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({"cases": [{"question": "historical?", "gold_answers": ["secret"]}]}), encoding="utf-8")
    identities, counts = exclusion_question_hashes(
        support, panel, support_sha256=file_sha256(support), panel_sha256=file_sha256(panel),
    )
    assert identities == {sha256_text("support one?"), sha256_text("support two?"), sha256_text("historical?")}
    assert counts["union_unique"] == 3


def test_synthetic_export_emits_complete_target_free_256x50_trace(tmp_path: Path):
    result = run(argparse.Namespace(
        config=Path("configs/experiments/reader_larger_untouched_utility_256.json"),
        plan=Path("configs/experiments/reader_larger_untouched_utility_256_plan.json"),
        output=tmp_path, support_bundle=None, historical_panel=None,
        exchange_url="http://127.0.0.1:1", token_env="UNUSED",
        upload=False, synthetic=True, validate_only=False,
    ))
    cohort = json.loads((result / "cohort.json").read_text(encoding="utf-8"))
    lock = json.loads((result / "cohort_lock.json").read_text(encoding="utf-8"))
    checks = validate_cohort(cohort, expected_cases=256)
    assert checks["top50_rows"] == 12800
    assert lock["schema_version"] == "sample-trace.v2"
    assert len(lock["samples"]) == 256
    assert not any(key in json.dumps(cohort).lower() for key in ('"answers"', '"gold_answers"', '"support_label"'))
    manifest = json.loads((result / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_directory"].startswith("five_ideas/reader_larger_untouched_cohort_256/")
    assert {row["name"] for row in manifest["exchange_files"]} == {
        "cohort.json", "cohort_lock.json", "executed_sources.zip", "summary.json",
    }
    assert manifest["github_files"] == ["summary.json", "upload_manifest.json", "upload_receipts.json"]
    assert set(manifest["exchange_only"]) == {"cohort.json", "cohort_lock.json", "executed_sources.zip"}
