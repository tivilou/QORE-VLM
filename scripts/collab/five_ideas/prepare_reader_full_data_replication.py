#!/usr/bin/env python3
"""Freeze and upload the complete untouched NQ/Wiki-DPR formal population."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.rag.reader_full_data_replication import (
    COHORT_SCHEMA,
    LOCK_TYPE,
    PRE_SCORE_STAGES,
    TRACE_SCHEMA,
    canonical_json_sha256,
    cohort_identity,
    freeze_complete_population,
    load_exposure_registry,
    split_cohort_identities,
    validate_full_cohort,
    validate_population_artifacts,
)
from applications.rag.reader_larger_untouched import file_sha256, sha256_text
from scripts.collab.lib.source_snapshot import archive_sources


CONFIG = ROOT / "configs/experiments/reader_full_data_replication.json"
PLAN = ROOT / "configs/experiments/reader_full_data_replication_plan.json"
DEFAULT_OUTPUT = ROOT / "exchange/five_ideas/reader_full_data_replication"
DEFAULT_EXCHANGE_URL = "http://117.50.198.37:18083"
SOURCES = (
    "applications/rag/reader_full_data_replication.py",
    "scripts/collab/five_ideas/prepare_reader_full_data_replication.py",
    "scripts/collab/five_ideas/prepare_reader_full_data_replication.sh",
    "scripts/collab/lib/exchange_upload.py",
    "scripts/collab/lib/source_snapshot.py",
    "configs/experiments/reader_full_data_replication.json",
    "configs/experiments/reader_full_data_replication_plan.json",
)


class FullDataExportError(RuntimeError):
    pass


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_revision() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return value or "unavailable"


def load_contract(
    config_path: Path = CONFIG, plan_path: Path = PLAN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "rag.reader_full_data_replication.v1":
        raise FullDataExportError("unexpected full-data config schema")
    if cfg.get("authorization") != "implementation_validated_formal_run_not_authorized":
        raise FullDataExportError("full-data implementation authorization drift")
    if cfg.get("formal_run_authorized") is not False:
        raise FullDataExportError("formal run must remain unauthorized")
    population = cfg.get("formal_population", {})
    if population.get("dataset") != "nq_open" or population.get("split") != "validation":
        raise FullDataExportError("formal dataset identity drift")
    if (population.get("pool_start_inclusive"), population.get("pool_end_exclusive")) != (512, 3610):
        raise FullDataExportError("formal source interval drift")
    if population.get("pool_cases_before_exclusion") != 3098:
        raise FullDataExportError("formal source count drift")
    if population.get("realized_count_bounds") != [2308, 2742]:
        raise FullDataExportError("formal population bounds drift")
    confirmations = cfg.get("independent_confirmations", {})
    if confirmations.get("split_seed") != 2026101907 or confirmations.get("required_units") != ["split_A", "split_B"]:
        raise FullDataExportError("independent split contract drift")
    retrieval = cfg.get("retrieval", {})
    required_retrieval = {
        "encoder_backend": "dpr",
        "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed",
        "nprobe": 64,
        "top_k": 50,
    }
    if any(retrieval.get(key) != value for key, value in required_retrieval.items()):
        raise FullDataExportError("retrieval contract drift")
    boundary = cfg.get("label_boundary", {}).get("before_all_scores_and_top5_are_frozen")
    if not isinstance(boundary, Mapping) or not boundary or any(value is not False for value in boundary.values()):
        raise FullDataExportError("label boundary must remain closed")
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v2":
        raise FullDataExportError("unexpected plugin-plan schema")
    if plan.get("authorization") != "approved":
        raise FullDataExportError("plugin plan is not approved for implementation")
    expected_plugins = [
        "complete_unexposed_population_freezer",
        "complete_frozen_reader_baseline_transport",
        "complete_frozen_head_transport",
        "split_sealed_blind_semantic_evaluator",
    ]
    if plan.get("discovery", {}).get("allowlist") != expected_plugins:
        raise FullDataExportError("plugin allowlist drift")
    if plan.get("composition", {}).get("order") != expected_plugins:
        raise FullDataExportError("plugin composition drift")
    return cfg, plan


def ensure_prerequisite(
    spec: Mapping[str, Any], destination: Path, *, exchange_url: str, token_env: str,
) -> Path:
    if destination.is_file():
        if destination.stat().st_size == int(spec["bytes"]) and file_sha256(destination) == spec["sha256"]:
            return destination
        raise FullDataExportError(f"existing prerequisite has wrong identity: {destination}")
    if destination.exists():
        raise FullDataExportError(f"prerequisite path is not a regular file: {destination}")
    from scripts.collab.lib.exchange_upload import download_exchange_file
    download_exchange_file(
        str(spec["exchange_path"]), destination, base_url=exchange_url,
        token_env=token_env, expected_size=int(spec["bytes"]),
        expected_sha256=str(spec["sha256"]),
    )
    return destination


def retrieve_rows(manager: Any, embedding: Any, *, top_k: int) -> list[dict[str, Any]]:
    import numpy as np

    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise FullDataExportError("Wiki-DPR manager does not expose nearest-example retrieval")
    scores, retrieved = dataset.get_nearest_examples(
        "embeddings", np.asarray(embedding, dtype=np.float32), k=top_k,
    )
    if len(scores) != top_k or any(key not in retrieved for key in ("id", "title", "text")):
        raise FullDataExportError("Wiki-DPR did not return a complete Top-50")
    output: list[dict[str, Any]] = []
    for index in range(top_k):
        raw_id = retrieved["id"][index]
        if raw_id is None or not str(raw_id).strip():
            raise FullDataExportError("Wiki-DPR returned an empty passage identity")
        candidate_id = str(raw_id)
        text = str(retrieved["text"][index] or "").strip()
        if not text:
            raise FullDataExportError("Wiki-DPR returned an empty passage")
        output.append({
            "id": candidate_id,
            "candidate_id_sha256": sha256_text(candidate_id),
            "title": str(retrieved["title"][index] or "").strip(),
            "text": text,
            "retrieved_rank": index + 1,
            "retrieval_score": float(scores[index]),
        })
    if len({row["id"] for row in output}) != top_k:
        raise FullDataExportError("Wiki-DPR returned duplicate passage identities")
    return output


def synthetic_inputs(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    end = int(cfg["formal_population"]["pool_end_exclusive"])
    start = int(cfg["formal_population"]["pool_start_inclusive"])
    rows = [{"id": f"synthetic-q-{index}", "question": f"Synthetic full question {index}?"}
            for index in range(end)]
    cursor = start
    sources: dict[str, set[str]] = {}
    for name, count in (("protected434", 434), ("historical100", 100), ("spent256", 256)):
        sources[name] = {
            sha256_text(rows[index]["question"]) for index in range(cursor, cursor + count)
        }
        cursor += count
    audit = {
        "source_identity_counts": {name: len(values) for name, values in sources.items()},
        "union_identities": len(set().union(*sources.values())),
        "registry_identity_sha256": canonical_json_sha256({
            name: sorted(values) for name, values in sources.items()
        }),
    }
    return rows, sources, audit


def synthetic_cases(population: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in population:
        candidates = []
        for rank in range(1, 51):
            candidate_id = f"synthetic-{row['dataset_index']}-{rank}"
            candidates.append({
                "id": candidate_id,
                "candidate_id_sha256": sha256_text(candidate_id),
                "title": f"Synthetic title {rank}",
                "text": f"Synthetic passage {rank} for question {row['dataset_index']}.",
                "retrieved_rank": rank,
                "retrieval_score": float(51 - rank),
            })
        cases.append({**row, "candidates": candidates})
    return cases


def materialize(
    population: list[dict[str, Any]],
    population_audit: Mapping[str, Any],
    registry_audit: Mapping[str, Any],
    cases: list[dict[str, Any]],
    *,
    cfg: Mapping[str, Any],
    output_root: Path,
    synthetic: bool,
) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    bounds = cfg["formal_population"]["realized_count_bounds"]
    identity = cohort_identity(cases)
    split_identities = split_cohort_identities(cases)
    boundary = dict(cfg["label_boundary"]["before_all_scores_and_top5_are_frozen"])
    cohort = {
        "schema_version": COHORT_SCHEMA,
        "run_id": run_id,
        "synthetic_only": synthetic,
        "formal_population": dict(cfg["formal_population"]),
        "independent_confirmations": dict(cfg["independent_confirmations"]),
        "retrieval": dict(cfg["retrieval"]),
        "exposure_registry_audit": dict(registry_audit),
        "population_audit": dict(population_audit),
        "cohort_identity_sha256": identity,
        "split_cohort_identity_sha256": split_identities,
        "label_boundary": boundary,
        "cases": cases,
    }
    checks = validate_full_cohort(cohort, count_bounds=bounds)
    cohort_path = run_dir / "cohort.json"
    write_json(cohort_path, cohort)
    lock = {
        "schema_version": TRACE_SCHEMA,
        "artifact_type": LOCK_TYPE,
        "run_id": run_id,
        "synthetic_only": synthetic,
        "population": {
            "complete": checks["cases"],
            "split_A": checks["split_counts"]["split_A"],
            "split_B": checks["split_counts"]["split_B"],
            "training": 0,
        },
        "stages": list(PRE_SCORE_STAGES),
        "later_stages": {
            "scoring": "not authorized in FD-02",
            "selection": "not authorized in FD-02",
            "evaluation": "not authorized in FD-02",
        },
        "sample_selection_rule": cfg["formal_population"]["selection_rule"],
        "population_audit": dict(population_audit),
        "samples": [{
            "population_position": row["population_position"],
            "dataset_index": row["dataset_index"],
            "question_sha256": row["question_sha256"],
            "confirmation_split": row["confirmation_split"],
            "candidate_id_sha256": [item["candidate_id_sha256"] for item in case["candidates"]],
            "candidate_retrieved_ranks": [item["retrieved_rank"] for item in case["candidates"]],
        } for row, case in zip(population, cases)],
        "cohort_identity_sha256": identity,
        "split_cohort_identity_sha256": split_identities,
        "cohort_sha256": file_sha256(cohort_path),
        "label_boundary": boundary,
        "checks": checks,
        "provenance": {
            "config_sha256": file_sha256(CONFIG),
            "plan_sha256": file_sha256(PLAN),
            "git_revision": git_revision(),
            "python": platform.python_version(),
        },
    }
    write_json(run_dir / "full_population_lock.json", lock)
    source_snapshot = archive_sources(ROOT, SOURCES, run_dir / "executed_sources.zip")
    summary = {
        "schema_version": "rag.reader_full_data_replication_result.v1",
        "run_id": run_id,
        "synthetic_only": synthetic,
        "status": "complete_population_frozen_pending_independent_verification" if not synthetic else "synthetic_fixture",
        "cases": checks["cases"],
        "top50_rows": checks["top50_rows"],
        "split_counts": checks["split_counts"],
        "cohort_identity_sha256": identity,
        "split_cohort_identity_sha256": split_identities,
        "population_identity_sha256": population_audit["population_identity_sha256"],
        "registry_identity_sha256": registry_audit["registry_identity_sha256"],
        "pre_score_validation": {
            "structural": "passed",
            "semantic": "passed",
            "target_boundary": "passed",
            "deterministic_replay": "passed",
        },
        "formal_run_authorized": False,
        "claim_ceiling": "fixture_only" if synthetic else "L0_complete_population_freeze_only",
        "next_step": "independently verify every hash before freezing a scoring manifest",
        "source_snapshot": source_snapshot,
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    if summary_path.stat().st_size > int(cfg["artifact_policy"]["github_max_result_bytes"]):
        raise FullDataExportError("compact summary exceeds the GitHub threshold")
    target = f"{cfg['output_namespace']}/{run_id}"
    exchange_names = ["cohort.json", "full_population_lock.json", "executed_sources.zip", "summary.json"]
    manifest = {
        "schema_version": "collaborator-exchange.manifest.v1",
        "target_directory": target,
        "status": "validated",
        "upload_status": "pending_until_receipts",
        "exchange_files": [{
            "name": name,
            "bytes": (run_dir / name).stat().st_size,
            "sha256": file_sha256(run_dir / name),
        } for name in exchange_names],
        "github_files": ["summary.json", "upload_manifest.json", "upload_receipts.json"],
        "exchange_only": ["cohort.json", "full_population_lock.json", "executed_sources.zip"],
        "privacy": {
            "raw_question_and_passage_text_exchange_only": True,
            "answers_or_semantic_targets_present": False,
            "method_outputs_present": False,
        },
    }
    write_json(run_dir / "upload_manifest.json", manifest)
    validate_population_artifacts(run_dir, count_bounds=bounds)
    return run_dir


def run(args: argparse.Namespace) -> Path | None:
    cfg, _plan = load_contract(args.config, args.plan)
    population_cfg = cfg["formal_population"]
    bounds = population_cfg["realized_count_bounds"]
    if args.validate_only:
        print(json.dumps({
            "status": "valid",
            "pool": [population_cfg["pool_start_inclusive"], population_cfg["pool_end_exclusive"]],
            "pool_cases": population_cfg["pool_cases_before_exclusion"],
            "realized_count_bounds": bounds,
            "model_loaded": False,
            "dataset_loaded": False,
            "wiki_dpr_started": False,
            "formal_run_authorized": False,
        }, sort_keys=True))
        return None

    if args.synthetic:
        rows, exposure_sources, registry_audit = synthetic_inputs(cfg)
        population, population_audit = freeze_complete_population(
            rows, exposure_sources,
            pool_start=int(population_cfg["pool_start_inclusive"]),
            pool_end=int(population_cfg["pool_end_exclusive"]),
            split_seed=int(cfg["independent_confirmations"]["split_seed"]),
            count_bounds=bounds,
        )
        run_dir = materialize(
            population, population_audit, registry_audit, synthetic_cases(population),
            cfg=cfg, output_root=args.output, synthetic=True,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="qduet-full-population-") as temporary:
            cache = Path(temporary)
            registry = cfg["exposure_registry"]
            support = ensure_prerequisite(
                registry["protected434"], args.support_bundle or cache / "protected434.zip",
                exchange_url=args.exchange_url, token_env=args.token_env,
            )
            historical = ensure_prerequisite(
                registry["historical100"], args.historical_panel or cache / "historical100.json",
                exchange_url=args.exchange_url, token_env=args.token_env,
            )
            spent = ensure_prerequisite(
                registry["spent256"], args.spent_cohort or cache / "spent256.json",
                exchange_url=args.exchange_url, token_env=args.token_env,
            )
            exposure_sources, registry_audit = load_exposure_registry(
                support, historical, spent, registry,
            )
            from applications.rag.data import load_dataset_for_rag, make_corpus_manager
            from applications.rag.retrieval import make_encoder

            rows = list(load_dataset_for_rag(
                population_cfg["dataset"], population_cfg["split"],
                int(population_cfg["pool_end_exclusive"]),
            ))
            population, population_audit = freeze_complete_population(
                rows, exposure_sources,
                pool_start=int(population_cfg["pool_start_inclusive"]),
                pool_end=int(population_cfg["pool_end_exclusive"]),
                split_seed=int(cfg["independent_confirmations"]["split_seed"]),
                count_bounds=bounds,
            )
            retrieval = cfg["retrieval"]
            encoder = make_encoder(retrieval["encoder_backend"])
            manager = make_corpus_manager(retrieval["corpus_mode"], {
                "wiki_dpr_config": retrieval["wiki_dpr_config"],
                "nprobe": int(retrieval["nprobe"]),
            })
            manager.build(population)
            cases = []
            for number, row in enumerate(population, 1):
                embedding = encoder.encode_queries([row["question"]])[0]
                candidates = retrieve_rows(manager, embedding, top_k=int(retrieval["top_k"]))
                cases.append({**row, "candidates": candidates})
                if number % 16 == 0 or number == len(population):
                    print(f"complete population retrieval {number}/{len(population)}", flush=True)
            run_dir = materialize(
                population, population_audit, registry_audit, cases,
                cfg=cfg, output_root=args.output, synthetic=False,
            )

    validation = validate_population_artifacts(run_dir, count_bounds=bounds)
    if args.upload:
        from scripts.collab.lib.exchange_upload import _upload_one, upload_manifest
        receipts = upload_manifest(
            run_dir / "upload_manifest.json", base_url=args.exchange_url,
            token_env=args.token_env,
        )
        manifest = json.loads((run_dir / "upload_manifest.json").read_text(encoding="utf-8"))
        expected = {
            f"{manifest['target_directory']}/{row['name']}": row
            for row in manifest["exchange_files"]
        }
        if len(receipts) != len(expected) or any(
            receipt.get("path") not in expected
            or receipt.get("sha256") != expected[receipt["path"]]["sha256"]
            or int(receipt.get("size_bytes", -1)) != expected[receipt["path"]]["bytes"]
            for receipt in receipts
        ):
            raise FullDataExportError("exchange receipt mismatch")
        receipt_path = run_dir / "upload_receipts.json"
        write_json(receipt_path, {"status": "uploaded", "receipts": receipts})
        token = os.environ.get(args.token_env)
        if not token:
            raise FullDataExportError(f"missing bearer token environment variable: {args.token_env}")
        _upload_one(
            args.exchange_url, token, run_dir / "upload_manifest.json",
            f"{manifest['target_directory']}/upload_manifest.json",
        )
        _upload_one(
            args.exchange_url, token, receipt_path,
            f"{manifest['target_directory']}/upload_receipts.json",
        )
    print(json.dumps({
        "output": str(run_dir),
        "summary": json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
        "validation": validation,
        "uploaded": bool(args.upload),
    }, ensure_ascii=False, indent=2))
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--support-bundle", type=Path, default=None)
    parser.add_argument("--historical-panel", type=Path, default=None)
    parser.add_argument("--spent-cohort", type=Path, default=None)
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", DEFAULT_EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())
