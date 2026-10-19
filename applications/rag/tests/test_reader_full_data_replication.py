import argparse
import copy
import json
from pathlib import Path
import zipfile

import pytest

from applications.rag.reader_full_data_replication import (
    COHORT_SCHEMA,
    cohort_identity,
    freeze_complete_population,
    load_exposure_registry,
    split_cohort_identities,
    validate_full_cohort,
    validate_population_artifacts,
)
from applications.rag.reader_larger_untouched import (
    cohort_identity as spent_cohort_identity,
    file_sha256,
    sha256_text,
)
from scripts.collab.five_ideas.prepare_reader_full_data_replication import (
    run,
    synthetic_cases,
    synthetic_inputs,
)


CONFIG = Path("configs/experiments/reader_full_data_replication.json")
PLAN = Path("configs/experiments/reader_full_data_replication_plan.json")


def _args(tmp_path: Path, **overrides):
    values = dict(
        config=CONFIG,
        plan=PLAN,
        output=tmp_path,
        support_bundle=None,
        historical_panel=None,
        spent_cohort=None,
        exchange_url="http://127.0.0.1:1",
        token_env="UNUSED",
        upload=False,
        synthetic=False,
        validate_only=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _candidate(case_index: int, rank: int) -> dict:
    candidate_id = f"candidate-{case_index}-{rank}"
    return {
        "id": candidate_id,
        "candidate_id_sha256": sha256_text(candidate_id),
        "title": f"title {rank}",
        "text": f"passage {rank}",
        "retrieved_rank": rank,
        "retrieval_score": float(51 - rank),
    }


def test_complete_population_scans_all_3098_positions_and_balances_splits():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, sources, _registry = synthetic_inputs(cfg)
    left, left_audit = freeze_complete_population(
        rows, sources, pool_start=512, pool_end=3610,
        split_seed=2026101907, count_bounds=[2308, 2742],
    )
    right, right_audit = freeze_complete_population(
        rows, sources, pool_start=512, pool_end=3610,
        split_seed=2026101907, count_bounds=[2308, 2742],
    )
    assert left == right
    assert left_audit == right_audit
    assert len(left) == 2308
    assert left_audit["scanned_positions"] == 3098
    assert left_audit["excluded_union_in_pool"] == 790
    assert left_audit["accounted_positions"] == 3098
    assert left_audit["split_counts"] == {"split_A": 1154, "split_B": 1154}
    assert not ({row["question_sha256"] for row in left} & set().union(*sources.values()))


def test_exposure_registry_replays_all_three_hash_locked_sources(tmp_path: Path):
    support = tmp_path / "support.zip"
    with zipfile.ZipFile(support, "w") as archive:
        archive.writestr("source.json", json.dumps({"cases": [
            {"question": "protected one?", "answers": ["secret"]},
            {"question": "protected two?", "answers": ["secret"]},
        ]}))
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"cases": [
        {"question": "historical?", "gold_answers": ["secret"]},
    ]}), encoding="utf-8")
    question = "spent?"
    spent_case = {
        "dataset_index": 7,
        "question": question,
        "question_sha256": sha256_text(question),
        "candidates": [_candidate(7, rank) for rank in range(1, 51)],
    }
    spent = tmp_path / "spent.json"
    spent.write_text(json.dumps({
        "schema_version": "rag.reader_larger_untouched_cohort.v1",
        "cohort_identity_sha256": spent_cohort_identity([spent_case]),
        "label_boundary": {"answers_read": False},
        "cases": [spent_case],
    }), encoding="utf-8")
    registry = {
        "union_rule": "exact union",
        "protected434": {
            "bytes": support.stat().st_size, "sha256": file_sha256(support),
            "expected_question_identities": 2,
        },
        "historical100": {
            "bytes": historical.stat().st_size, "sha256": file_sha256(historical),
            "expected_question_identities": 1,
        },
        "spent256": {
            "bytes": spent.stat().st_size, "sha256": file_sha256(spent),
            "cohort_identity_sha256": spent_cohort_identity([spent_case]),
            "expected_question_identities": 1,
        },
        "expected_union_identities": 4,
    }
    sources, audit = load_exposure_registry(support, historical, spent, registry)
    assert {name: len(values) for name, values in sources.items()} == {
        "protected434": 2, "historical100": 1, "spent256": 1,
    }
    assert audit["union_identities"] == 4
    assert all("secret" not in value for values in sources.values() for value in values)


def test_duplicate_source_question_and_forbidden_target_are_rejected():
    rows = [{"id": str(index), "question": f"question {index}?"} for index in range(10)]
    rows[8]["question"] = rows[7]["question"]
    empty_sources = {"protected434": set(), "historical100": set(), "spent256": set()}
    with pytest.raises(ValueError, match="duplicate question"):
        freeze_complete_population(
            rows, empty_sources, pool_start=2, pool_end=10,
            split_seed=3, count_bounds=[8, 8],
        )

    rows[8]["question"] = "question 8?"
    population, audit = freeze_complete_population(
        rows, empty_sources, pool_start=2, pool_end=10,
        split_seed=3, count_bounds=[8, 8],
    )
    cases = []
    for row in population:
        cases.append({
            **row,
            "candidates": [_candidate(row["dataset_index"], rank) for rank in range(1, 51)],
        })
    cohort = {
        "schema_version": COHORT_SCHEMA,
        "cohort_identity_sha256": cohort_identity(cases),
        "split_cohort_identity_sha256": split_cohort_identities(cases),
        "population_audit": audit,
        "label_boundary": {"answers_read": False},
        "cases": cases,
    }
    assert validate_full_cohort(cohort, count_bounds=[8, 8])["cases"] == 8
    tampered = copy.deepcopy(cohort)
    tampered["answers"] = ["forbidden"]
    with pytest.raises(ValueError, match="forbidden"):
        validate_full_cohort(tampered, count_bounds=[8, 8])


def test_validate_only_loads_no_dataset_or_model(tmp_path: Path, capsys):
    assert run(_args(tmp_path, validate_only=True)) is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["pool_cases"] == 3098
    assert payload["realized_count_bounds"] == [2308, 2742]
    assert payload["model_loaded"] is False
    assert payload["dataset_loaded"] is False
    assert payload["wiki_dpr_started"] is False
    assert payload["formal_run_authorized"] is False


def test_synthetic_export_emits_complete_2308x50_target_free_trace(tmp_path: Path):
    result = run(_args(tmp_path, synthetic=True))
    assert result is not None
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((result / "full_population_lock.json").read_text(encoding="utf-8"))
    cohort = json.loads((result / "cohort.json").read_text(encoding="utf-8"))
    assert summary["cases"] == 2308
    assert summary["top50_rows"] == 115400
    assert summary["split_counts"] == {"split_A": 1154, "split_B": 1154}
    assert summary["formal_run_authorized"] is False
    assert lock["schema_version"] == "sample-trace.v2"
    assert lock["stages"] == ["data", "preprocess", "retrieval", "diagnosis"]
    assert len(lock["samples"]) == 2308
    serialized = json.dumps(cohort).lower()
    assert '"answers"' not in serialized
    assert '"gold_answers"' not in serialized
    assert '"support_label"' not in serialized
    replay = validate_population_artifacts(result, count_bounds=[2308, 2742])
    assert replay["status"] == "passed"
    assert replay["lock_samples_replayed"] == 2308
    assert replay["manifest_files_replayed"] == 4
    manifest = json.loads((result / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_directory"].startswith("five_ideas/reader_full_data_replication/")
    assert {row["name"] for row in manifest["exchange_files"]} == {
        "cohort.json", "full_population_lock.json", "executed_sources.zip", "summary.json",
    }
    assert set(manifest["exchange_only"]) == {
        "cohort.json", "full_population_lock.json", "executed_sources.zip",
    }
