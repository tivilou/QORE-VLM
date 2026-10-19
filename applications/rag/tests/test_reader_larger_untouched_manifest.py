import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from applications.rag.reader_larger_untouched import cohort_identity, sha256_text
from applications.rag.reader_larger_untouched_manifest import (
    array_record,
    build_scoring_manifest,
    canonical_json_sha256,
    file_sha256,
    validate_scoring_manifest,
    verify_sequential_checkpoint,
)


def _write_fixture(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "method.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    cohort_dir = tmp_path / "cohort"
    cohort_dir.mkdir()
    cases = []
    for case_index in range(256):
        question = f"question {case_index}?"
        candidates = []
        for candidate_index in range(50):
            candidate_id = f"{case_index}:{candidate_index}"
            candidates.append({
                "id": candidate_id,
                "candidate_id_sha256": sha256_text(candidate_id),
                "retrieved_rank": candidate_index + 1,
                "retrieval_score": float(50 - candidate_index),
                "title": f"title {candidate_index}",
                "text": f"passage {case_index} {candidate_index}",
            })
        cases.append({
            "dataset_index": case_index + 512,
            "question": question,
            "question_sha256": sha256_text(question),
            "candidates": candidates,
        })
    identity = cohort_identity(cases)
    cohort = {
        "schema_version": "rag.reader_larger_untouched_cohort.v1",
        "cohort_identity_sha256": identity,
        "cases": cases,
    }
    cohort_path = cohort_dir / "cohort.json"
    cohort_path.write_text(json.dumps(cohort, separators=(",", ":")), encoding="utf-8")
    lock = {
        "schema_version": "sample-trace.v2",
        "artifact_type": "reader_larger_untouched_cohort_lock",
        "run_id": "fixture-run",
        "cohort_sha256": file_sha256(cohort_path),
        "cohort_identity_sha256": identity,
        "label_boundary": {"answers_read": False, "support_annotations_read": False},
        "samples": [{
            "question_sha256": case["question_sha256"],
            "candidate_id_sha256": [row["candidate_id_sha256"] for row in case["candidates"]],
            "candidate_retrieved_ranks": list(range(1, 51)),
        } for case in cases],
    }
    lock_path = cohort_dir / "cohort_lock.json"
    lock_path.write_text(json.dumps(lock, separators=(",", ":")), encoding="utf-8")

    def source_archive(directory: Path, name: str, member: str, payload: bytes):
        source_hashes = {member: __import__("hashlib").sha256(payload).hexdigest()}
        archive_path = directory / name
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(member, payload)
            archive.writestr("source_manifest.json", json.dumps({
                "schema_version": "source-snapshot.v1", "source_hashes": source_hashes,
            }))
        return archive_path, source_hashes

    cohort_source, cohort_source_hashes = source_archive(
        cohort_dir, "executed_sources.zip", "cohort_source.py", b"COHORT = 1\n"
    )

    checkpoint_path = tmp_path / "sequential_values.npz"
    arrays = {}
    for family, final_value in (("quantum", 0.125), ("classical", -0.25)):
        arrays[family + "__final_theta"] = np.full((40, 5), final_value, dtype=np.float64)
        arrays[family + "__head__theta"] = np.zeros((40, 5), dtype=np.float32)
        arrays[family + "__head__weight"] = np.arange(768, dtype=np.float32)
        arrays[family + "__head__bias"] = np.asarray([0.5], dtype=np.float32)
        arrays[family + "__head__projection"] = np.arange(4 * 3840, dtype=np.float32).reshape(4, 3840)
        for index in range(40):
            arrays[f"{family}__head__low_{index}"] = np.asarray([index], dtype=np.int64)
            arrays[f"{family}__head__high_{index}"] = np.asarray([index + 1], dtype=np.int64)
    np.savez(checkpoint_path, **arrays)
    checkpoint_source, checkpoint_source_hashes = source_archive(
        checkpoint_path.parent, "checkpoint_sources.zip", "checkpoint_source.py", b"CHECKPOINT = 1\n"
    )

    reader = tmp_path / "reader"
    reader.mkdir()
    reader_files = {}
    for name, content in {
        "config.json": b"{}",
        "pytorch_model.bin": b"weights",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"{}",
        "vocab.txt": b"tokens",
    }.items():
        path = reader / name
        path.write_bytes(content)
        reader_files[name] = {"bytes": len(content), "sha256": file_sha256(path)}

    def state_identity(family: str) -> tuple[str, dict]:
        keys = sorted(key for key in arrays if key == family + "__final_theta" or key.startswith(family + "__head__"))
        entries = [{"key": key, **array_record(arrays[key])} for key in keys]
        identity_entries = [
            {key: entry[key] for key in ("key", "shape", "dtype", "sha256")}
            for entry in entries
        ]
        critical = {
            name: {"sha256": array_record(arrays[f"{family}__{name}"])["sha256"]}
            for name in ("final_theta", "head__theta", "head__weight", "head__bias", "head__projection")
        }
        return canonical_json_sha256(identity_entries), critical

    arm_specs = {}
    for family in ("quantum", "classical"):
        state_sha, critical = state_identity(family)
        arm_specs[family] = {"restore_state_sha256": state_sha, "critical_arrays": critical}
    spec = {
        "schema_version": "rag.reader_larger_untouched_scoring_spec.v1",
        "cohort": {
            "exchange_path": "fixture/cohort",
            "run_id": "fixture-run",
            "cases": 256,
            "cohort_identity_sha256": identity,
            "files": {
                "cohort.json": {"bytes": cohort_path.stat().st_size, "sha256": file_sha256(cohort_path)},
                "cohort_lock.json": {"bytes": lock_path.stat().st_size, "sha256": file_sha256(lock_path)},
            },
            "source_snapshot": {
                "file": cohort_source.name,
                "bytes": cohort_source.stat().st_size,
                "sha256": file_sha256(cohort_source),
                "git_revision": "2" * 40,
                "source_hashes": cohort_source_hashes,
            },
        },
        "reader": {
            "model_id": "fixture/reader",
            "revision": "3" * 40,
            "max_length": 350,
            "frozen_max_batch_size": 8,
            "files": reader_files,
        },
        "sequential_checkpoint": {
            "exchange_path": "fixture/sequential_values.npz",
            "run_id": "fixture-checkpoint",
            "file": {"bytes": checkpoint_path.stat().st_size, "sha256": file_sha256(checkpoint_path)},
            "source_snapshot": {
                "file": checkpoint_source.name,
                "bytes": checkpoint_source.stat().st_size,
                "sha256": file_sha256(checkpoint_source),
                "source_hashes": checkpoint_source_hashes,
            },
            "source_config_sha256": "5" * 64,
            "arms": arm_specs,
        },
        "arms": ["identity_reader", "quantum_sequential_final", "classical_sequential_final"],
        "head": {"layers": 2, "conditioning": 4, "seed": 20261013},
        "selection": {"candidate_count": 50, "selected_count": 5, "tie_break": "score_descending_then_candidate_index_ascending"},
        "transport_gate": {"min_quantum_changed_cases_for_review": 25, "min_quantum_classical_different_cases_for_review": 15},
        "label_boundary": {"answers_read": False, "support_annotations_read": False, "silver_oracle_read": False},
        "method_source_hashes": {"method.py": file_sha256(source)},
    }
    return root, cohort_dir, checkpoint_path, reader, spec


def test_real_shape_fixture_freezes_all_inputs_before_scoring(tmp_path: Path):
    root, cohort_dir, checkpoint, reader, spec = _write_fixture(tmp_path)
    manifest = build_scoring_manifest(
        spec,
        root=root,
        cohort_dir=cohort_dir,
        sequential_values=checkpoint,
        reader_snapshot=reader,
        source_snapshot={"file": "sources.zip", "bytes": 1, "sha256": "6" * 64, "source_hashes": {}},
        git_revision="7" * 40,
    )
    checks = validate_scoring_manifest(manifest)
    assert checks == {
        "status": "passed",
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "cases": 256,
        "top50_rows": 12800,
        "arms": 3,
        "scoring_started": False,
    }
    assert manifest["sequential_checkpoint"]["arms"]["quantum"]["entry_count"] == 85
    assert manifest["cohort"]["label_boundary_verified"]


def test_checkpoint_byte_drift_is_rejected(tmp_path: Path):
    _root, _cohort_dir, checkpoint, _reader, spec = _write_fixture(tmp_path)
    with checkpoint.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(ValueError, match="byte-size mismatch"):
        verify_sequential_checkpoint(checkpoint, spec)


def test_manifest_identity_tamper_is_rejected(tmp_path: Path):
    root, cohort_dir, checkpoint, reader, spec = _write_fixture(tmp_path)
    manifest = build_scoring_manifest(
        spec,
        root=root,
        cohort_dir=cohort_dir,
        sequential_values=checkpoint,
        reader_snapshot=reader,
        source_snapshot={"file": "sources.zip", "bytes": 1, "sha256": "6" * 64, "source_hashes": {}},
        git_revision="7" * 40,
    )
    tampered = copy.deepcopy(manifest)
    tampered["transport_gate"]["min_quantum_changed_cases_for_review"] = 1
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        validate_scoring_manifest(tampered)
