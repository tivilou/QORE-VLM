#!/usr/bin/env python3
"""Export all registered old-training cases for separate-axis support review.

Reuses local cached retrieval and frozen Reader, never heads or training.
The full case trace and unfilled review are private exchange artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import semantic_supervision_audit as audit
from applications.rag import support_supervision_axes as axes
from scripts.collab.five_ideas import run_semantic_reader_supervision_audit as old

CONFIG = ROOT / "configs/experiments/support_supervision_cohort_export.json"
NAMESPACE = "five_ideas/support_supervision_cohort_export"
RAW_FILES = ("case_study.json", "case_study.md", "support_axes_review.json", "target_draft.json", "future_split.json")
COMPACT_FILES = ("summary.json", "report.md", "run_metadata.json", "upload_manifest.json")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(path):
    cfg = load(path)
    axes.require(cfg == load(CONFIG) and cfg["schema_version"] == "rag.support_cohort_export_config.v1", "use committed export config")
    axes.require(cfg["case_count"] == 434 and cfg["candidate_count"] == 50 and
                 cfg["source_manifest_sha256"] == old.SOURCE_MANIFEST_SHA and
                 cfg["allowlist"] == ["frozen_reader_export", "pending_axes_template", "offline_target_draft"] and
                 cfg["training_allowed"] is False and cfg["generator_allowed"] is False, "export boundary changed")
    return cfg


def prepare_input(directory, args):
    """Download only the locked small cohort, not checkpoints or corpora."""
    directory = directory or ROOT / ".cache/semantic_reader_supervision_audit/input"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "input_manifest.json"
    def get(name, **checks):
        from scripts.collab.lib.exchange_upload import download_exchange_file
        download_exchange_file(old.SOURCE_NAMESPACE + "/" + name, directory / name,
                               base_url=args.exchange_url, token_env=args.token_env, **checks)
    if not manifest.exists():
        get("input_manifest.json", expected_sha256=old.SOURCE_MANIFEST_SHA)
    axes.require(not manifest.is_symlink() and audit.digest_file(manifest) == old.SOURCE_MANIFEST_SHA, "source manifest mismatch")
    rows = load(manifest)["files"]
    axes.require([r["name"] for r in rows] == ["cohort.json", "head_checkpoints.npz"], "input allowlist changed")
    row = rows[0]
    path = directory / "cohort.json"
    if not path.exists():
        get("cohort.json", expected_size=row["bytes"], expected_sha256=row["sha256"])
    axes.require(not path.is_symlink() and path.stat().st_size == row["bytes"] and
                 audit.digest_file(path) == row["sha256"], "source cohort mismatch")
    return path


def reader_scores(torch, tokenizer, reader, device, question, texts, spec):
    """Same input construction as prior semantic Reader; relevance only, no hook."""
    result = []
    axes.require(not reader.training and all(not p.requires_grad for p in reader.parameters()), "Reader must be frozen/eval")
    with torch.no_grad():
        for start in range(0, len(texts), spec["batch_size"]):
            batch = texts[start:start + spec["batch_size"]]
            encoded = tokenizer(questions=[question] * len(batch), texts=batch,
                                padding=True, truncation=True, max_length=spec["max_length"], return_tensors="pt")
            output = reader(input_ids=encoded["input_ids"].to(device),
                            attention_mask=encoded["attention_mask"].to(device), return_dict=True)
            result.extend(output.relevance_logits.detach().cpu().reshape(-1).tolist())
    axes.require(len(result) == 50, "Reader score coverage mismatch")
    return result


def record_case(number, identity, question, candidates, scores, role):
    mask, reason = audit.weak_labels([old.online_text(p) for p in candidates], question["answers"])
    axes.require(reason == "ok", "registered mixed weak target changed")
    audit.check_identity(question["question"], candidates, mask, identity)
    ranking = audit.rank_order(scores, candidates)
    ranks = {i: r + 1 for r, i in enumerate(ranking)}
    qid = identity["question_sha256"]
    ps = []
    for i, p in enumerate(candidates):
        # Producer-owned numeric values are all captured, no learned intervention.
        ps.append({"id": str(p["id"]), "retrieved_rank": p["retrieved_rank"],
                   "retrieval_score": float(p["retrieval_score"]), "title": p["title"], "text": p["text"],
                   "text_sha256": axes.text_hash(p["text"]), "title_sha256": axes.text_hash(p["title"]),
                   "review_item_id": axes.text_hash(qid + ":" + str(p["id"])),
                   "reader_score": float(scores[i]), "reader_rank": ranks[i], "weak_positive_diagnostic_only": mask[i]})
    return {"usable_case_index": number, "question_sha256": qid, "question": question["question"],
            "answers": question["answers"], "lane": "previous_training_diagnostic_not_heldout",
            "future_role": role, "candidates": ps, "reader_top5_indices": ranking[:5]}


def make_trace(cases, cohort, provenance):
    return {"schema_version": axes.TRACE_SCHEMA, "case_count": len(cases), "training_authorized": False,
            "evaluation_question_sha256": cohort["evaluation_question_sha256"], "provenance": provenance,
            "stages": ["data", "retrieval", "frozen_reader_scoring", "baseline_selection", "pending_review"],
            "numeric_capture": {"reader_score": "complete [case_count,50] float values",
                                "retrieval_score": "complete [case_count,50] float values"},
            "hidden_tensor_capture": False, "cases": cases}


def validate_outputs(directory, cohort):
    """Reopen every raw role; replay scores, identity, split and pending masks."""
    trace = load(directory / "case_study.json")
    cases = axes.validate_trace(trace)
    identities = audit.validate_cohort(cohort)
    axes.require(len(cases) == 434 and len(identities) == 434, "full434 coverage required")
    _, future = audit.sample_cohort(cohort)
    axes.require(load(directory / "future_split.json") == future, "prospective split changed")
    for c, identity, role in zip(cases, identities, future["assignments"]):
        axes.require(c["future_role"] == role["future_role"], "prospective assignment changed")
        mask, reason = audit.weak_labels([old.online_text(p) for p in c["candidates"]], c["answers"])
        axes.require(reason == "ok" and mask == [p["weak_positive_diagnostic_only"] for p in c["candidates"]], "weak diagnostic changed")
        audit.check_identity(c["question"], c["candidates"], mask, identity)
        axes.require(c["reader_top5_indices"] == audit.rank_order([p["reader_score"] for p in c["candidates"]], c["candidates"])[:5], "baseline replay changed")
    annotations = load(directory / "support_axes_review.json")
    axes.require(annotations == axes.pending_template(trace), "export review must stay entirely unfilled")
    draft = axes.compile_draft(trace, annotations)
    axes.require(load(directory / "target_draft.json") == draft and draft["draft_pair_count"] == 0, "pending targets changed")
    md = (directory / "case_study.md").read_text(encoding="utf-8")
    axes.require(md.count("## 问题 ") == 434 and md.count("### 检索段 ") == 21700, "human trace incomplete")
    return {"status": "passed", "cases": 434, "passages": 21700, "baseline_replayed": True,
            "train_count": 347, "validation_count": 87, "qualified_pairs": 0, "training_authorized": False}


def write_outputs(directory, cases, cohort, provenance):
    directory.mkdir(parents=True, exist_ok=False)
    trace = make_trace(cases, cohort, provenance)
    audit.write_json(directory / "case_study.json", trace)
    review = axes.pending_template(trace)
    audit.write_json(directory / "support_axes_review.json", review)
    audit.write_json(directory / "target_draft.json", axes.compile_draft(trace, review))
    _, future = audit.sample_cohort(cohort)
    audit.write_json(directory / "future_split.json", future)
    with (directory / "case_study.md").open("w", encoding="utf-8", newline="\n") as target:
        target.write("# 全训练队列支持监督导出\n\n434 道旧训练题；不训练、不生成。参考答案并非完整有效答案集合。\n")
        for c in cases:
            target.write(f"\n## 问题 {c['usable_case_index']}\n\n{c['question']}\n\n参考答案：{' / '.join(c['answers'])}\n\n")
            target.write(f"Reader Top-5 检索序号：{[i + 1 for i in c['reader_top5_indices']]}；未来分区：{c['future_role']}（旧头已经见过）\n\n")
            for p in c["candidates"]:
                target.write(f"### 检索段 {p['retrieved_rank']} · {p['title']}\n\nReader 排名：{p['reader_rank']}；Reader 分数：{p['reader_score']}; 检索分数：{p['retrieval_score']}\n\n{p['text']}\n\n")
    verified = validate_outputs(directory, cohort)
    summary = {"schema_version": "rag.support_cohort_export_summary.v1", "status": "exported_pending_review",
               "fixture": provenance.get("fixture", False), "validation": verified,
               "annotation_count": 21700, "qualified_pairs": 0, "training_authorized": False,
               "training_consumable": False, "source_manifest_sha256": old.SOURCE_MANIFEST_SHA,
               "historical_full_text_parity_established": False, "previous_heads_trained_on_both_roles": True,
               "independent_validation_claim": False, "head_checkpoints_downloaded": False,
               "corpus_or_model_downloaded": False, "generator_called": False, "support_quality_measured": False}
    audit.write_json(directory / "summary.json", summary)
    (directory / "report.md").write_text("# 全队列导出完成，监督资格仍待审\n\n434 题 / 21700 段；未来 347/87 分区不是旧模型的独立验证。\n四个判断轴分开、全部待填；合格训练对 0，训练权限 false。\n本次锁定当前文本，不证明历史 432 题文本一致，也不证明筛选提升。\n", encoding="utf-8")
    names = [*RAW_FILES, "summary.json", "report.md"]
    records = [{"name": n, "bytes": (directory / n).stat().st_size, "sha256": audit.digest_file(directory / n)} for n in names]
    audit.write_json(directory / "run_metadata.json", {"files": records, "provenance": provenance, "validation": verified})
    names.append("run_metadata.json")
    manifest = {"target_directory": NAMESPACE + "/" + directory.name, "github_files": list(COMPACT_FILES),
                "exchange_only": list(RAW_FILES), "exchange_files": [
                    {"name": n, "bytes": (directory / n).stat().st_size, "sha256": audit.digest_file(directory / n)} for n in names]}
    audit.write_json(directory / "upload_manifest.json", manifest)
    for n in COMPACT_FILES:
        axes.require((directory / n).stat().st_size <= 1048576, "compact file exceeds 1 MiB")
    return summary


def upload_outputs(directory, args):
    from scripts.collab.lib.exchange_upload import upload_manifest, _upload_one
    m = load(directory / "upload_manifest.json")
    axes.require(m["target_directory"] == NAMESPACE + "/" + directory.name and
                 m["github_files"] == list(COMPACT_FILES) and m["exchange_only"] == list(RAW_FILES) and
                 [r["name"] for r in m["exchange_files"]] == [*RAW_FILES, "summary.json", "report.md", "run_metadata.json"], "upload allowlist mismatch")
    for r in m["exchange_files"]:
        p = directory / r["name"]
        axes.require(not p.is_symlink() and p.stat().st_size == r["bytes"] and audit.digest_file(p) == r["sha256"], "artifact changed before upload")
    for n in COMPACT_FILES:
        axes.require((directory / n).stat().st_size <= 1048576, "compact file exceeds 1 MiB")
    receipts = upload_manifest(directory / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
    receipts.append(_upload_one(args.exchange_url, os.environ[args.token_env], directory / "upload_manifest.json", m["target_directory"] + "/upload_manifest.json"))
    audit.write_json(directory / "upload_receipts.json", {"receipts": receipts})
    return receipts


def fixture(directory):
    cohort = {"usable_cases": 434, "requested_cases": 512,
              "evaluation_question_sha256": [axes.text_hash("EXCLUDED " + str(i)) for i in range(100)],
              "training_identity": []}
    cases = []
    for i in range(434):
        question = {"question": "Fixture question " + str(i + 1), "answers": ["fixture answer"]}
        ps = [{"id": f"{i + 1}:{j}", "retrieved_rank": j + 1, "retrieval_score": 50. - j,
               "title": f"Fixture title {j}", "text": "fixture answer explicitly supported" if j % 7 == 0 else "Unrelated fixture passage"} for j in range(50)]
        mask, _ = audit.weak_labels([old.online_text(p) for p in ps], question["answers"])
        identity = {"question_sha256": axes.text_hash(question["question"]),
                    "candidate_id_sha256": [axes.text_hash(p["id"]) for p in ps], "positive_mask": mask}
        cohort["training_identity"].append(identity)
        cases.append(record_case(i + 1, identity, question, ps, [float(50 - j) for j in range(50)], "training"))
    _, future = audit.sample_cohort(cohort)
    for c, role in zip(cases, future["assignments"]):
        c["future_role"] = role["future_role"]
    result = write_outputs(directory, cases, cohort, {"fixture": True, "offline": True, "no_real_model_passes": True})
    return result, cohort


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--preflight", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange" / NAMESPACE)
    p.add_argument("--wiki-dpr-cache")
    p.add_argument("--device")
    p.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", "http://117.50.198.37:18083"))
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--upload-only", type=Path)
    args = p.parse_args(argv)
    validate_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "valid", "cases": 434, "training_authorized": False})); return
    if args.preflight:
        print(json.dumps(fixture(args.preflight)[0]["validation"])); return
    if args.upload or args.upload_only:
        axes.require(bool(os.environ.get(args.token_env)), "configure QORE_EXCHANGE_TOKEN before exporting")
    if args.upload_only:
        print(json.dumps({"uploaded_files": len(upload_outputs(args.upload_only, args))})); return
    cohort_path = prepare_input(args.input_dir, args)
    cohort = load(cohort_path)
    identities = audit.validate_cohort(cohort)
    axes.require(len(identities) == 434 and cohort["requested_cases"] == 512, "locked434 cohort required")
    _, future = audit.sample_cohort(cohort)
    torch, tokenizer, reader, device, by_hash, retrieve, provenance = old.local_backends(cohort, args)
    spec = cohort["source_config"]["reader"]
    axes.require(provenance["reader"]["config_sha256"] == cohort["reader"]["config_sha256"], "Reader config identity mismatch")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance.update(source_run=cohort["source_run"], source_manifest_sha256=old.SOURCE_MANIFEST_SHA,
                      cohort_sha256=audit.digest_file(cohort_path), code_revision=revision,
                      config_sha256=audit.digest_file(args.config), reader_input_spec=spec,
                      source_hashes={str(x.relative_to(ROOT)).replace("\\", "/"): audit.digest_file(x) for x in
                          (Path(__file__), ROOT / "applications/rag/support_supervision_axes.py",
                           ROOT / "applications/rag/semantic_supervision_audit.py", Path(old.__file__),
                           ROOT / "scripts/collab/five_ideas/run_qarcg_reader_screen_100.py")},
                      python=sys.version.split()[0], torch=torch.__version__, fixture=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_root / run_id
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Persist each completed case for failure inspection; never publish an incomplete run.
    partial = args.output_root / (run_id + ".partial.jsonl")
    cases = []
    with partial.open("x", encoding="utf-8", newline="\n") as sink:
        for number, (identity, assignment) in enumerate(zip(identities, future["assignments"]), 1):
            question = by_hash[identity["question_sha256"]]
            candidates = retrieve(question["question"])
            mask, _ = audit.weak_labels([old.online_text(x) for x in candidates], question["answers"])
            audit.check_identity(question["question"], candidates, mask, identity)
            scores = reader_scores(torch, tokenizer, reader, device, question["question"], [old.online_text(x) for x in candidates], spec)
            case = record_case(number, identity, question, candidates, scores, assignment["future_role"])
            cases.append(case)
            sink.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n"); sink.flush()
            if number % 10 == 0 or number == 434:
                print(f"exported {number}/434", flush=True)
    summary = write_outputs(output, cases, cohort, provenance)
    partial.unlink()  # Exact producer-owned scratch file, only after full readback passed.
    if args.upload:
        upload_outputs(output, args)
    print(json.dumps({"output_dir": str(output), "uploaded": args.upload, "validation": summary["validation"]}))


if __name__ == "__main__":
    main()
