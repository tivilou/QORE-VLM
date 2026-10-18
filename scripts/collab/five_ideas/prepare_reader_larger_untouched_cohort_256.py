#!/usr/bin/env python3
"""Export a fresh label-blind NQ/Wiki-DPR 256x50 cohort and upload it."""
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

from applications.rag.reader_larger_untouched import (
    cohort_identity,
    exclusion_question_hashes,
    file_sha256,
    select_fresh_questions,
    sha256_text,
    validate_cohort,
)
from scripts.collab.lib.source_snapshot import archive_sources


CONFIG = ROOT / "configs/experiments/reader_larger_untouched_utility_256.json"
PLAN = ROOT / "configs/experiments/reader_larger_untouched_utility_256_plan.json"
DEFAULT_OUTPUT = ROOT / "exchange/five_ideas/reader_larger_untouched_cohort_256"
DEFAULT_EXCHANGE_URL = "http://117.50.198.37:18083"
SOURCES = (
    "applications/rag/reader_larger_untouched.py",
    "scripts/collab/five_ideas/prepare_reader_larger_untouched_cohort_256.py",
    "scripts/collab/lib/exchange_upload.py",
    "scripts/collab/lib/source_snapshot.py",
    "configs/experiments/reader_larger_untouched_utility_256.json",
    "configs/experiments/reader_larger_untouched_utility_256_plan.json",
)


class CohortExportError(RuntimeError):
    pass


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_revision() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return value or "unavailable"


def load_contract(config_path: Path = CONFIG, plan_path: Path = PLAN) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dataset = cfg.get("dataset", {})
    retrieval = cfg.get("retrieval", {})
    if cfg.get("schema_version") != "rag.reader_larger_untouched_utility.v1":
        raise CohortExportError("unexpected larger-untouched config schema")
    if cfg.get("authorization") != "cohort_export_only_then_freeze_hash":
        raise CohortExportError("cohort export authorization drift")
    if dataset != {
        "name": "nq_open", "split": "validation", "pool_start": 512,
        "pool_end": 3610, "cohort_cases": 256, "selection_seed": 20261018,
        "selection_rule": "sort eligible questions by SHA256(seed:dataset_index:question_sha256), then take first256",
    }:
        raise CohortExportError("fresh question selection contract drift")
    if retrieval != {
        "encoder_backend": "dpr", "corpus_mode": "wiki_dpr",
        "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64, "top_k": 50,
    }:
        raise CohortExportError("retrieval contract drift")
    if any(cfg.get("label_boundary", {}).values()):
        raise CohortExportError("label boundary must be entirely false")
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v2" or plan.get("authorization") != "approved":
        raise CohortExportError("plugin plan must remain approved until trace readiness")
    expected_plugins = ["fresh_question_identity_filter", "frozen_wiki_dpr_top50", "frozen_reader_head_transport"]
    if plan.get("discovery", {}).get("allowlist") != expected_plugins or plan.get("composition", {}).get("order") != expected_plugins:
        raise CohortExportError("plugin allowlist/order drift")
    return cfg, plan


def ensure_prerequisite(
    spec: Mapping[str, Any], destination: Path, *, exchange_url: str, token_env: str,
) -> Path:
    if destination.is_file() and file_sha256(destination) == spec["sha256"]:
        return destination
    if destination.exists():
        raise CohortExportError(f"existing prerequisite has wrong hash: {destination}")
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
        raise CohortExportError("Wiki-DPR manager does not expose nearest-example retrieval")
    scores, retrieved = dataset.get_nearest_examples(
        "embeddings", np.asarray(embedding, dtype=np.float32), k=top_k,
    )
    if len(scores) != top_k or any(key not in retrieved for key in ("id", "title", "text")):
        raise CohortExportError("Wiki-DPR did not return a complete Top-50")
    output = []
    for index in range(top_k):
        raw_candidate_id = retrieved["id"][index]
        if raw_candidate_id is None or not str(raw_candidate_id).strip():
            raise CohortExportError("Wiki-DPR returned an empty passage identity")
        candidate_id = str(raw_candidate_id)
        text = str(retrieved["text"][index] or "").strip()
        if not text:
            raise CohortExportError("Wiki-DPR returned an empty passage")
        output.append({
            "id": candidate_id,
            "candidate_id_sha256": sha256_text(candidate_id),
            "title": str(retrieved["title"][index] or "").strip(),
            "text": text,
            "retrieved_rank": index + 1,
            "retrieval_score": float(scores[index]),
        })
    if len({row["id"] for row in output}) != top_k:
        raise CohortExportError("Wiki-DPR returned duplicate passage identities")
    return output


def synthetic_cases(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in selected:
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
    selected: list[dict[str, Any]], cases: list[dict[str, Any]], exclusions: Mapping[str, int],
    *, cfg: Mapping[str, Any], output_root: Path, synthetic: bool,
) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    identity = cohort_identity(cases)
    cohort = {
        "schema_version": "rag.reader_larger_untouched_cohort.v1",
        "run_id": run_id,
        "synthetic_only": synthetic,
        "dataset": dict(cfg["dataset"]),
        "retrieval": dict(cfg["retrieval"]),
        "cohort_identity_sha256": identity,
        "label_boundary": dict(cfg["label_boundary"]),
        "cases": cases,
    }
    checks = validate_cohort(cohort, expected_cases=int(cfg["dataset"]["cohort_cases"]))
    write_json(run_dir / "cohort.json", cohort)
    lock = {
        "schema_version": "sample-trace.v2",
        "artifact_type": "reader_larger_untouched_cohort_lock",
        "run_id": run_id,
        "synthetic_only": synthetic,
        "population": {"evaluation": len(cases), "training": 0, "validation": 0},
        "stages": ["data", "preprocess", "retrieval", "diagnosis"],
        "sample_selection_rule": cfg["dataset"]["selection_rule"],
        "exclusion_counts": dict(exclusions),
        "samples": [{
            "dataset_index": row["dataset_index"],
            "question_sha256": row["question_sha256"],
            "candidate_id_sha256": [item["candidate_id_sha256"] for item in case["candidates"]],
            "candidate_retrieved_ranks": [item["retrieved_rank"] for item in case["candidates"]],
        } for row, case in zip(selected, cases)],
        "cohort_identity_sha256": identity,
        "cohort_sha256": file_sha256(run_dir / "cohort.json"),
        "label_boundary": dict(cfg["label_boundary"]),
        "checks": checks,
        "provenance": {
            "config_sha256": file_sha256(CONFIG),
            "plan_sha256": file_sha256(PLAN),
            "git_revision": git_revision(),
            "python": platform.python_version(),
        },
    }
    write_json(run_dir / "cohort_lock.json", lock)
    snapshot = archive_sources(ROOT, SOURCES, run_dir / "executed_sources.zip")
    summary = {
        "schema_version": "rag.reader_larger_untouched_cohort_result.v1",
        "run_id": run_id,
        "synthetic_only": synthetic,
        "status": "cohort_frozen_pending_transport" if not synthetic else "synthetic_fixture",
        "cases": len(cases),
        "top50_rows": len(cases) * 50,
        "cohort_identity_sha256": identity,
        "cohort_sha256": file_sha256(run_dir / "cohort.json"),
        "exclusion_counts": dict(exclusions),
        "claim_ceiling": "fixture_only" if synthetic else cfg["claim_ceiling"],
        "next_step": "verify cohort lock, then authorize one-shot Reader/head transport without inspecting text",
        "source_snapshot": snapshot,
    }
    write_json(run_dir / "summary.json", summary)
    target = f"{cfg['output_namespace']}/{run_id}"
    exchange_names = ["cohort.json", "cohort_lock.json", "executed_sources.zip", "summary.json"]
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
        "exchange_only": ["cohort.json", "cohort_lock.json", "executed_sources.zip"],
        "privacy": {
            "raw_question_and_passage_text_exchange_only": True,
            "answers_or_semantic_targets_present": False,
            "method_outputs_present": False,
        },
    }
    if (run_dir / "summary.json").stat().st_size > 1_048_576:
        raise CohortExportError("compact summary exceeds the 1 MiB GitHub threshold")
    write_json(run_dir / "upload_manifest.json", manifest)
    return run_dir


def run(args: argparse.Namespace) -> Path | None:
    cfg, _plan = load_contract(args.config, args.plan)
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "cases": cfg["dataset"]["cohort_cases"],
            "pool": [cfg["dataset"]["pool_start"], cfg["dataset"]["pool_end"]],
            "model_loaded": False, "wiki_dpr_started": False,
        }, sort_keys=True))
        return None

    if args.synthetic:
        rows = [{"id": f"q-{index}", "question": f"Synthetic fresh question {index}?"}
                for index in range(int(cfg["dataset"]["pool_end"]))]
        exclusions = {sha256_text("Synthetic fresh question 700?")}
        exclusion_counts = {"support_cases": 1, "support_unique": 1, "historical_panel_cases": 0,
                            "historical_panel_unique": 0, "union_unique": 1}
        selected = select_fresh_questions(
            rows, exclusions, pool_start=int(cfg["dataset"]["pool_start"]),
            pool_end=int(cfg["dataset"]["pool_end"]), count=int(cfg["dataset"]["cohort_cases"]),
            seed=int(cfg["dataset"]["selection_seed"]),
        )
        run_dir = materialize(selected, synthetic_cases(selected), exclusion_counts,
                              cfg=cfg, output_root=args.output, synthetic=True)
    else:
        with tempfile.TemporaryDirectory(prefix="qduet-fresh-cohort-") as temporary:
            cache = Path(temporary)
            support_spec = cfg["exclusions"]["support_bundle"]
            panel_spec = cfg["exclusions"]["historical_panel"]
            support = ensure_prerequisite(
                support_spec, args.support_bundle or cache / "frozen_support_bundle.zip",
                exchange_url=args.exchange_url, token_env=args.token_env,
            )
            panel = ensure_prerequisite(
                panel_spec, args.historical_panel or cache / "historical_panel.json",
                exchange_url=args.exchange_url, token_env=args.token_env,
            )
            exclusions, exclusion_counts = exclusion_question_hashes(
                support, panel, support_sha256=support_spec["sha256"], panel_sha256=panel_spec["sha256"],
            )
            if exclusion_counts["support_cases"] != int(support_spec["expected_cases"]):
                raise CohortExportError("support exclusion coverage drift")
            if exclusion_counts["historical_panel_cases"] != int(panel_spec["expected_cases"]):
                raise CohortExportError("historical-panel exclusion coverage drift")
            from applications.rag.data import load_dataset_for_rag, make_corpus_manager
            from applications.rag.retrieval import make_encoder

            rows = list(load_dataset_for_rag(
                cfg["dataset"]["name"], cfg["dataset"]["split"], int(cfg["dataset"]["pool_end"]),
            ))
            if len(rows) != int(cfg["dataset"]["pool_end"]):
                raise CohortExportError("NQ validation population is shorter than the frozen pool")
            selected = select_fresh_questions(
                rows, exclusions, pool_start=int(cfg["dataset"]["pool_start"]),
                pool_end=int(cfg["dataset"]["pool_end"]), count=int(cfg["dataset"]["cohort_cases"]),
                seed=int(cfg["dataset"]["selection_seed"]),
            )
            encoder = make_encoder(cfg["retrieval"]["encoder_backend"])
            manager = make_corpus_manager(cfg["retrieval"]["corpus_mode"], {
                "wiki_dpr_config": cfg["retrieval"]["wiki_dpr_config"],
                "nprobe": int(cfg["retrieval"]["nprobe"]),
            })
            manager.build(selected)
            cases = []
            for number, row in enumerate(selected, 1):
                embedding = encoder.encode_queries([row["question"]])[0]
                candidates = retrieve_rows(manager, embedding, top_k=int(cfg["retrieval"]["top_k"]))
                cases.append({**row, "candidates": candidates})
                if number % 8 == 0 or number == len(selected):
                    print(f"fresh cohort retrieval {number}/{len(selected)}", flush=True)
            run_dir = materialize(selected, cases, exclusion_counts, cfg=cfg,
                                  output_root=args.output, synthetic=False)

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
            raise CohortExportError("exchange receipt mismatch")
        receipt_path = run_dir / "upload_receipts.json"
        write_json(receipt_path, {"status": "uploaded", "receipts": receipts})
        token = os.environ.get(args.token_env)
        if not token:
            raise CohortExportError(f"missing bearer token environment variable: {args.token_env}")
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
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", DEFAULT_EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    run(parser.parse_args())
