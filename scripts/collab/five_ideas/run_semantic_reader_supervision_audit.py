#!/usr/bin/env python3
"""Export 32 train-only support cases and replay existing heads; never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from applications.rag import semantic_supervision_audit as audit

CONFIG = ROOT / "configs/experiments/semantic_reader_supervision_audit.json"
RAW_FILES = ("case_study.json", "case_study.md", "support_review.json", "audit_values.npz", "future_split.json")
COMPACT_FILES = ("summary.json", "report.md", "run_metadata.json", "upload_manifest.json")
SOURCE_NAMESPACE = "five_ideas/semantic_reader_supervision_audit_input/repair_20261008T145652Z"
SOURCE_MANIFEST_SHA = "0464da7792e0fd2a0615c2e5a9058b641ef0d3b3300af6d0b39450b44fd39a81"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(path):
    cfg = load_json(path)
    if cfg != load_json(CONFIG) or cfg["schema_version"] != "rag.semantic_reader_supervision_audit.v1":
        raise ValueError("use the committed audit configuration")
    if cfg["training_allowed"] or cfg["generator_allowed"] or cfg["evaluation_allowed"]:
        raise ValueError("audit boundary changed")
    if tuple(cfg["methods"]) != audit.METHODS or cfg["sample_count"] != 32 or cfg["sample_seed"] != 20261009 or cfg["source_manifest_sha256"] != SOURCE_MANIFEST_SHA:
        raise ValueError("sampling/method/source identity changed")
    plan = load_json(ROOT / "configs/experiments/semantic_reader_supervision_audit_plan.json")
    allowlist = ["train_cohort_export", *audit.METHODS]
    if plan["discovery"]["allowlist"] != allowlist or plan["composition"]["order"] != allowlist:
        raise ValueError("diagnostic plugin allowlist/order mismatch")
    recovery = plan["recovery_contract"]
    if audit.digest_file(ROOT / recovery["dossier_path"]) != recovery["dossier_sha256"]:
        raise ValueError("recovery evidence identity mismatch")
    return cfg


def prepare_input(directory, args):
    from scripts.collab.lib.exchange_upload import download_exchange_file
    directory = directory or ROOT / ".cache/semantic_reader_supervision_audit/input"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "input_manifest.json"
    if not manifest.exists():
        download_exchange_file(SOURCE_NAMESPACE + "/input_manifest.json", manifest,
                               base_url=args.exchange_url, token_env=args.token_env, expected_sha256=SOURCE_MANIFEST_SHA)
    if audit.digest_file(manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError("audit input manifest hash mismatch")
    records = load_json(manifest)["files"]
    if [r["name"] for r in records] != ["cohort.json", "head_checkpoints.npz"]:
        raise ValueError("input file allowlist changed")
    for row in records:
        path = directory / row["name"]
        if not path.exists():
            download_exchange_file(SOURCE_NAMESPACE + "/" + row["name"], path, base_url=args.exchange_url,
                                   token_env=args.token_env, expected_size=row["bytes"], expected_sha256=row["sha256"])
        if path.stat().st_size != row["bytes"] or audit.digest_file(path) != row["sha256"]:
            raise ValueError("audit input artifact mismatch: " + row["name"])
    return directory


def force_offline():
    # Set before datasets/transformers imports, not merely setdefault.
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"


def local_backends(cohort, args):
    """Reuse cached dataset/index/models; never build a new index or download."""
    force_offline()
    import torch
    from datasets import load_dataset, DownloadConfig
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRReader, DPRReaderTokenizer
    from scripts.collab.five_ideas import run_qarcg_reader_screen_100 as old
    from types import SimpleNamespace
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    qa = load_dataset("nq_open", split="train", download_config=DownloadConfig(local_files_only=True))
    if len(qa) < 512:
        raise ValueError("cached NQ train slice is incomplete")
    questions = [{"question": r["question"], "answers": r["answer"]} for r in qa.select(range(512))]
    all_hashes = [audit.digest_text(q["question"]) for q in questions]
    if len(set(all_hashes)) != 512 or set(all_hashes) & set(cohort["evaluation_question_sha256"]):
        raise ValueError("train/evaluation overlap or duplicate training questions")
    by_hash = {audit.digest_text(q["question"]): q for q in questions}
    if any(r["question_sha256"] not in by_hash for r in cohort["training_identity"]):
        raise ValueError("cached first-512 NQ population differs from prior training")
    ds = load_dataset("facebook/wiki_dpr", "psgs_w100.nq.compressed", split="train",
                      cache_dir=args.wiki_dpr_cache, trust_remote_code=True,
                      download_config=DownloadConfig(local_files_only=True))
    if "embeddings" not in ds.list_indexes():
        raise ValueError("cached Wiki-DPR compressed index missing; no index building is permitted")
    index = ds.get_index("embeddings").faiss_index
    if not hasattr(index, "nprobe"):
        raise ValueError("expected the existing compressed Wiki-DPR index")
    index.nprobe = 64
    name = "facebook/dpr-question_encoder-single-nq-base"
    query_model = DPRQuestionEncoder.from_pretrained(name, local_files_only=True).to(device).eval()
    query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(name, local_files_only=True)
    spec = cohort["source_config"]["reader"]
    kwargs = {"revision": spec["revision"], "local_files_only": True}
    reader = DPRReader.from_pretrained(spec["model_id"], **kwargs).to(device).eval()
    tokenizer = DPRReaderTokenizer.from_pretrained(spec["model_id"], **kwargs)
    for model in (reader, query_model):
        for p in model.parameters():
            p.requires_grad_(False)
    reader_identity = old._reader_identity(reader, tokenizer, spec["model_id"], spec["revision"], device)
    if reader_identity["resolved_revision"] != spec["revision"]:
        raise ValueError("Reader revision mismatch")
    def retrieve(question):
        encoded = query_tokenizer([question], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            embedding = query_model(**{k: v.to(device) for k, v in encoded.items()}).pooler_output.cpu().numpy()[0]
        records, _ = old._retrieve(SimpleNamespace(_dataset=ds), embedding, 50)
        return records
    provenance = {"reader": reader_identity,
                  "question_encoder": {"model_id": name, "resolved_revision": getattr(query_model.config, "_commit_hash", None)},
                  "nq_fingerprint": qa._fingerprint, "wiki_dpr_fingerprint": ds._fingerprint,
                  "wiki_dpr_config": "psgs_w100.nq.compressed", "nprobe": 64,
                  "offline": True, "index_built": False}
    return torch, tokenizer, reader, device, by_hash, retrieve, provenance


def restore_heads(torch, cohort, archive, hidden_size, device):
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    heads = screen.make_heads(torch, hidden_size, cohort["source_config"], device)
    for method, head in heads.items():
        screen.restore(torch, head, method, "3", archive)
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
    return heads


def record_case(torch, heads, row, question, records, values, arrays, device, prefix):
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    mask, reason = audit.weak_labels([online_text(p) for p in records], question["answers"])
    if reason != "ok":
        raise ValueError("registered training case no longer has its mixed target")
    for name, value in values.items():
        if value is not None:
            arrays[prefix + "__" + name] = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    base = values["base"].numpy()
    arrays[prefix + "__weak_mask"] = np.asarray(mask, dtype=bool)
    ranking = audit.rank_order(base, records)
    ranks = {i: j + 1 for j, i in enumerate(ranking)}
    candidates = []
    for i, p in enumerate(records):
        candidates.append({**p, "question_passage_review_id": audit.digest_text(row["question_sha256"] + ":" + str(p["id"])),
                           "text_sha256": audit.digest_text(p["text"]), "online_text_sha256": audit.digest_text(online_text(p)),
                           "reader_score": float(base[i]), "reader_rank": ranks[i], "weak_positive": mask[i],
                           "weak_match_witnesses": audit.containment_witnesses(online_text(p), question["answers"])})
    case = {**row, "question": question["question"], "answers": question["answers"],
            "lane": "previous_training_diagnostic_not_heldout", "value_prefix": prefix,
            "candidates": candidates, "methods": {}, "stages": ["data", "retrieval", "scoring", "selection", "diagnosis"]}
    outputs = {"frozen_reader_topk": (values["base"],),
               "monotonic_compression_control": (values["base"] * .5,)}
    with torch.no_grad():
        for method, head in heads.items():
            outputs[method] = screen.head_forward(head, method, values, device)
    for method in audit.METHODS:
        output = outputs[method]
        for name, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
            arrays[prefix + "__" + method + "__" + name] = value.detach().cpu().numpy()
        scores = output[0].cpu().numpy()
        selected = audit.rank_order(scores, records)[:5]
        case["methods"][method] = {"scores": scores.tolist(), "selected_indices": selected,
             "selected_retrieved_ranks": [records[i]["retrieved_rank"] for i in selected],
             "weak_positive_count": sum(mask[i] for i in selected),
             "compression": audit.compression_diagnostic(base, scores),
             "boundary": audit.boundary_diagnostic(base, scores, mask, records)}
    return case


def online_text(p):
    title, text = p["title"].strip(), p["text"].strip()
    if not text:
        raise ValueError("empty candidate text")
    return f"{title}. {text}" if title else text


def validate_trace(directory, cohort, source_arrays, *, fixture=False):
    import torch
    from applications.rag.quantum_semantic_reader import semantic_pool
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    trace = load_json(directory / "case_study.json")
    if trace["schema_version"] != "sample-trace.v2" or trace["values_sha256"] != audit.digest_file(directory / "audit_values.npz"):
        raise ValueError("trace/archive identity mismatch")
    selected, future = audit.sample_cohort(cohort)
    if load_json(directory / "future_split.json") != future or len(trace["cases"]) != 32:
        raise ValueError("trace split/sample coverage mismatch")
    with np.load(directory / "audit_values.npz", allow_pickle=False) as archive:
        hidden_size = archive[trace["cases"][0]["value_prefix"] + "__pooled"].shape[1] // 5
        heads = restore_heads(torch, cohort, source_arrays, hidden_size, torch.device("cpu"))
        required = set()
        for expected, case in zip(selected, trace["cases"]):
            if any(case.get(k) != v for k, v in expected.items()) or case.get("stages") != ["data", "retrieval", "scoring", "selection", "diagnosis"]:
                raise ValueError("trace sample rule/stage order mismatch")
            identity = cohort["training_identity"][case["usable_case_index"] - 1]
            candidates = case["candidates"]
            mask, reason = audit.weak_labels([online_text(p) for p in candidates], case["answers"])
            audit.check_identity(case["question"], candidates, mask, identity)
            prefix = case["value_prefix"]
            for p, m in zip(candidates, mask):
                if p["text_sha256"] != audit.digest_text(p["text"]) or p["online_text_sha256"] != audit.digest_text(online_text(p)) or p["weak_positive"] is not m:
                    raise ValueError("raw/weak trace mismatch")
                if p["weak_match_witnesses"] != audit.containment_witnesses(online_text(p), case["answers"]):
                    raise ValueError("containment witness mismatch")
            values = {key: torch.as_tensor(archive[prefix + "__" + key]) for key in ("base", "pooled", "scalar")}
            if values["base"].shape != (50,) or values["pooled"].shape != (50, hidden_size * 5) or values["scalar"].shape != (50, 4):
                raise ValueError("complete coordinate coverage mismatch")
            np.testing.assert_array_equal(archive[prefix + "__weak_mask"], mask)
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise ValueError("nonfinite trace states")
            number = case["usable_case_index"]
            if number <= 2:
                for key in ("base", "pooled", "scalar"):
                    torch.testing.assert_close(values[key], torch.as_tensor(source_arrays[f"training_{number:03d}__{key}"]), rtol=1e-4, atol=1e-4)
                hidden = torch.as_tensor(archive[prefix + "__hidden"])[None]
                qm, pm = [torch.as_tensor(archive[prefix + "__" + k])[None] for k in ("question_mask", "passage_mask")]
                pooled, weights = semantic_pool(hidden, qm, pm)
                torch.testing.assert_close(pooled[0], values["pooled"][0], rtol=1e-4, atol=1e-4)
                torch.testing.assert_close(weights[0], torch.as_tensor(archive[prefix + "__pooling_weights"]), rtol=1e-4, atol=1e-4)
                if archive[prefix + "__input_ids"].shape != archive[prefix + "__question_mask"].shape:
                    raise ValueError("hidden token coordinate mismatch")
            baseline = values["base"].numpy()
            order = audit.rank_order(baseline, candidates)
            for reader_rank, candidate_index in enumerate(order, 1):
                p = candidates[candidate_index]
                if p["reader_rank"] != reader_rank or p["reader_score"] != float(baseline[candidate_index]):
                    raise ValueError("candidate Reader score/rank trace mismatch")
            # Restore all initial heads and prove exact null on every selected sample.
            with torch.no_grad():
                for method, head in heads.items():
                    screen.restore(torch, head, method, "initial", source_arrays)
                    null = screen.head_forward(head, method, values, torch.device("cpu"))[0]
                    torch.testing.assert_close(null, values["base"], rtol=0, atol=0)
                    screen.restore(torch, head, method, "3", source_arrays)
                outputs = {"frozen_reader_topk": (values["base"],), "monotonic_compression_control": (values["base"] * .5,)}
                outputs.update({m: screen.head_forward(h, m, values, torch.device("cpu")) for m, h in heads.items()})
            if tuple(case["methods"]) != audit.METHODS:
                raise ValueError("method allowlist/order mismatch")
            for method, output in outputs.items():
                for key, value in zip(("scores", "gate", "residual", "observables", "encoded"), output):
                    torch.testing.assert_close(value, torch.as_tensor(archive[prefix + "__" + method + "__" + key]), rtol=1e-4, atol=1e-4)
                saved = archive[prefix + "__" + method + "__scores"]
                row = case["methods"][method]
                np.testing.assert_array_equal(saved, row["scores"])
                selected_indices = audit.rank_order(saved, candidates)[:5]
                if selected_indices != row["selected_indices"] or row["selected_retrieved_ranks"] != [candidates[i]["retrieved_rank"] for i in selected_indices]:
                    raise ValueError("selection replay mismatch")
                if row["weak_positive_count"] != sum(mask[i] for i in selected_indices):
                    raise ValueError("weak diagnostic mismatch")
                if row["boundary"] != audit.boundary_diagnostic(baseline, saved, mask, candidates) or row["compression"] != audit.compression_diagnostic(baseline, saved):
                    raise ValueError("diagnostic arithmetic mismatch")
                if method in audit.TRAINABLE and np.max(np.abs(saved - baseline)) > .25 * max(np.std(baseline), 1e-6) + 1e-4:
                    raise ValueError("effective correction violates source bound")
            if case["methods"]["frozen_reader_topk"]["selected_indices"] != case["methods"]["monotonic_compression_control"]["selected_indices"]:
                raise ValueError("monotonic null changed selection")
            required.update(k for k in archive.files if k.startswith(prefix + "__"))
        if set(archive.files) != required:
            raise ValueError("unexpected detached trace coordinates")
    if load_json(directory / "support_review.json") != audit.blinded_review(trace["cases"]):
        raise ValueError("blinded review leaks/changes producer-owned inputs")
    return {k: "pass" for k in ("project_v2_structure", "sample_split_identity", "complete_detached_values",
               "all_head_replay", "initial_null", "compression_null", "weak_metric_replay", "prior_two_witness_parity",
               "hidden_pooling_slices", "raw_input_hashes", "blinded_target_boundary")}


def write_outputs(directory, cases, arrays, cohort, future, provenance, source_arrays, *, fixture=False):
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "audit_values.npz", **arrays)
    trace = {"schema_version": "sample-trace.v2", "artifact_type": "train_only_semantic_reader_supervision_audit",
             "cases": cases, "provenance": provenance, "values_sha256": audit.digest_file(directory / "audit_values.npz"),
             "training_called": False, "generator_called": False, "evaluation_called": False,
             "support_certified": False, "independent_validation": False, "fixture": fixture}
    audit.write_json(directory / "case_study.json", trace)
    audit.write_json(directory / "support_review.json", audit.blinded_review(cases))
    audit.write_json(directory / "future_split.json", future)
    verified = validate_trace(directory, cohort, source_arrays, fixture=fixture)
    summary = {"schema_version": "rag.semantic_reader_supervision_audit.summary.v1", "claim_ceiling": "L0_diagnostic",
       "sample_cases": len(cases), "candidate_count": len(cases) * 50,
       "weak_target_not_support_gold": True, "support_review_status": "pending_review",
       "training_called": False, "generator_called": False, "evaluation_called": False,
       "existing_heads_trained_on_all_cases": True, "fixture": fixture,
       "methods": {m: {"weak_positive_selected_total": sum(c["methods"][m]["weak_positive_count"] for c in cases),
            "changed_top5_cases": sum(set(c["methods"][m]["selected_indices"]) != set(c["methods"]["frozen_reader_topk"]["selected_indices"]) for c in cases),
            "boundary_pairs": sum(c["methods"][m]["boundary"]["pair_count"] for c in cases),
            "strict_crossings": sum(c["methods"][m]["boundary"]["strict_crossings"] for c in cases)} for m in audit.METHODS},
       "validation": verified, "provenance": provenance,
       "limitations": ["Containment is not certified support; pending blinded review blocks a support-quality conclusion.",
          "This is a deterministic diagnostic sample of the previous training cohort, not independent validation.",
          "Prospective validation split applies only to future newly initialized training; old heads saw all these cases.",
          "Original raw text hashes for 432 training inputs were absent; exact IDs/order/weak masks are checked, but full historical text parity is unproven.",
          "No task utility, generalization or quantum advantage is inferred."]}
    audit.write_json(directory / "summary.json", summary)
    (directory / "report.md").write_text("# Train-only supervision audit\n\n32 cases x 50 passages; replay only; no training/Generator/evaluation.\n\n"
       + "\n".join(f"- {m}: {v['weak_positive_selected_total']} weak-positive selected; {v['changed_top5_cases']} changed sets." for m, v in summary["methods"].items())
       + "\n\n**Weak counts are not support accuracy.** Blinded support review is pending. Old heads trained on every sampled case.\n", encoding="utf-8")
    md = ["# 训练监督与筛选边界审计", "", "只重放上一轮检查点，不训练。所有答案字符串标签均为弱标签，不代表真正支持。",
          "请先独立审阅 support_review.json，再查看下方分数和标签。", ""]
    for c in cases:
        md.extend([f"## 原训练问题 {c['usable_case_index']}", c["question"], "参考答案：" + " / ".join(c["answers"]),
                   "未来分区：" + c["future_role"] + "（旧检查点已经见过此题，不是当前留出验证）", "",
                   "|方法|选中检索序号|弱正例数|", "|---|---|---|"])
        for m, r in c["methods"].items():
            md.append(f"|{m}|{r['selected_retrieved_ranks']}|{r['weak_positive_count']}|")
        md.append("")
        for p in sorted(c["candidates"], key=lambda p: p["reader_rank"]):
            md.extend([f"### Reader第{p['reader_rank']}名 · 检索第{p['retrieved_rank']}段 · {p['title']}",
                       f"ID：{p['id']}；Reader分数：{p['reader_score']:.7f}；字符串弱标签：{p['weak_positive']}", p["text"], ""])
    (directory / "case_study.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    names = [*RAW_FILES, "summary.json", "report.md"]
    records = [{"name": n, "bytes": (directory / n).stat().st_size, "sha256": audit.digest_file(directory / n)} for n in names]
    audit.write_json(directory / "run_metadata.json", {"files": records, "provenance": provenance, "validation": verified})
    names.append("run_metadata.json")
    target = "five_ideas/semantic_reader_supervision_audit/" + directory.name
    manifest = {"target_directory": target,
                "exchange_files": [{"name": n, "bytes": (directory / n).stat().st_size, "sha256": audit.digest_file(directory / n)} for n in names],
                "github_files": list(COMPACT_FILES), "exchange_only": list(RAW_FILES), "provenance": provenance}
    audit.write_json(directory / "upload_manifest.json", manifest)
    for name in COMPACT_FILES:
        if (directory / name).stat().st_size > 1_048_576:
            raise ValueError("compact publication exceeds 1 MiB")
    return summary


def produce_fixture(directory, *, return_context=False):
    import torch
    from applications.rag.quantum_semantic_reader import semantic_pool
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    cfg = load_json(ROOT / "configs/experiments/semantic_reader_training_repair_100.json")
    cohort = {"usable_cases": 434, "requested_cases": 512, "source_config": cfg,
       "evaluation_question_sha256": [audit.digest_text("EXCLUDED " + str(i)) for i in range(100)],
       "training_identity": [{"question_sha256": audit.digest_text("Fixture question " + str(i + 1)),
             "candidate_id_sha256": [audit.digest_text(f"{i + 1}:{j}") for j in range(50)],
             "positive_mask": [j % 7 == 0 for j in range(50)]} for i in range(434)]}
    selected, future = audit.sample_cohort(cohort)
    heads = screen.make_heads(torch, 8, cfg, torch.device("cpu"))
    checkpoints = {}
    for m, h in heads.items():
        screen.checkpoint(h, m, "initial", checkpoints)
        with torch.no_grad():
            # Synthetic fixture intervention; no optimizer/backward or real-data fitting.
            interaction = getattr(h, "interaction", h)
            for name, p in interaction.named_parameters():
                if "readout" in name or "scale" in name:
                    p.add_(.1)
        screen.checkpoint(h, m, "3", checkpoints)
    rng = np.random.default_rng(20261009)
    arrays, cases = {}, []
    for k, row in enumerate(selected):
        n = row["usable_case_index"]
        q = {"question": "Fixture question " + str(n), "answers": ["fixture answer"]}
        records = [{"id": f"{n}:{j}", "retrieved_rank": j + 1, "retrieval_score": 50. - j,
                    "title": f"Fixture title {j}", "text": "fixture answer explicitly supported" if j % 7 == 0 else "Unrelated fixture passage"} for j in range(50)]
        values = {"base": torch.linspace(4, -4, 50), "pooled": torch.tensor(rng.normal(size=(50, 40)), dtype=torch.float32),
                  "scalar": torch.tensor(rng.normal(size=(50, 4)), dtype=torch.float32)}
        if n <= 2:
            hidden = torch.tensor(rng.normal(size=(12, 8)), dtype=torch.float32)
            qm, pm = torch.zeros(12, dtype=torch.bool), torch.zeros(12, dtype=torch.bool)
            qm[1:4] = True; pm[5:11] = True
            pooled, weights = semantic_pool(hidden[None], qm[None], pm[None])
            values["pooled"][0] = pooled[0]
            values.update(hidden=hidden, input_ids=torch.arange(12), question_mask=qm, passage_mask=pm, pooling_weights=weights[0])
            for key in ("base", "pooled", "scalar"):
                checkpoints[f"training_{n:03d}__{key}"] = values[key].numpy().copy()
        cases.append(record_case(torch, heads, row, q, records, values, arrays, torch.device("cpu"), f"audit_{k + 1:03d}"))
    summary = write_outputs(directory, cases, arrays, cohort, future, {"fixture": True}, checkpoints, fixture=True)
    return (summary, cohort, checkpoints) if return_context else summary


def upload_outputs(directory, args):
    """Hash-check pending local outputs; failed transport leaves a resumable run."""
    from scripts.collab.lib.exchange_upload import upload_manifest, _upload_one
    manifest = load_json(directory / "upload_manifest.json")
    allowed = [*RAW_FILES, "summary.json", "report.md", "run_metadata.json"]
    if [r["name"] for r in manifest["exchange_files"]] != allowed or manifest["github_files"] != list(COMPACT_FILES) or manifest["exchange_only"] != list(RAW_FILES):
        raise ValueError("upload output allowlist changed")
    expected_target = "five_ideas/semantic_reader_supervision_audit/" + directory.name
    if manifest["target_directory"] != expected_target:
        raise ValueError("upload namespace mismatch")
    for row in manifest["exchange_files"]:
        path = directory / row["name"]
        if path.is_symlink() or path.stat().st_size != row["bytes"] or audit.digest_file(path) != row["sha256"]:
            raise ValueError("pending artifact changed: " + row["name"])
    for name in COMPACT_FILES:
        if (directory / name).stat().st_size > 1_048_576:
            raise ValueError("compact publication exceeds 1 MiB")
    receipts = upload_manifest(directory / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
    receipts.append(_upload_one(args.exchange_url, os.environ[args.token_env], directory / "upload_manifest.json", expected_target + "/upload_manifest.json"))
    audit.write_json(directory / "upload_receipts.json", {"receipts": receipts})
    return receipts


def run(args):
    validate_config(args.config)
    if args.validate_only:
        print(json.dumps({"status": "valid", "cases": 32, "training_allowed": False, "methods": audit.METHODS})); return
    if args.preflight:
        print(json.dumps(produce_fixture(args.preflight)["validation"])); return
    if args.upload and not os.environ.get(args.token_env):
        raise ValueError("configure QORE_EXCHANGE_TOKEN before starting the export")
    if args.upload_only:
        receipts = upload_outputs(args.upload_only, args)
        print(json.dumps({"output_dir": str(args.upload_only), "uploaded": True, "files": len(receipts), "new_model_passes": 0})); return
    input_dir = prepare_input(args.input_dir, args)
    cohort = load_json(input_dir / "cohort.json")
    audit.validate_cohort(cohort)
    selected, future = audit.sample_cohort(cohort)
    torch, tokenizer, reader, device, by_hash, retrieve, provenance = local_backends(cohort, args)
    from applications.rag.quantum_semantic_reader import reader_semantic_forward
    from scripts.collab.five_ideas import run_quantum_semantic_reader_screen_100 as screen
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError("a Git checkout is required to record executed code provenance")
    provenance.update(source_run=cohort["source_run"], source_manifest_sha256=SOURCE_MANIFEST_SHA,
                      source_checkpoint_sha256=audit.digest_file(input_dir / "head_checkpoints.npz"),
                      code_revision=revision, config_sha256=audit.digest_file(args.config),
                      source_hashes={str(p.relative_to(ROOT)).replace("\\", "/"): audit.digest_file(p) for p in
                                     (Path(__file__), ROOT / "applications/rag/semantic_supervision_audit.py",
                                      ROOT / "applications/rag/quantum_semantic_reader.py", ROOT / "applications/rag/qarcg_reader.py",
                                      ROOT / "scripts/collab/five_ideas/run_quantum_semantic_reader_screen_100.py")},
                      python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__)
    arrays, cases = {}, []
    with np.load(input_dir / "head_checkpoints.npz", allow_pickle=False) as source_arrays:
        heads = restore_heads(torch, cohort, source_arrays, reader.config.hidden_size, device)
        for k, row in enumerate(selected):
            question = by_hash[row["question_sha256"]]
            records = retrieve(question["question"])
            texts = [online_text(p) for p in records]
            mask, reason = audit.weak_labels(texts, question["answers"])
            audit.check_identity(question["question"], records, mask, cohort["training_identity"][row["usable_case_index"] - 1])
            spec = cohort["source_config"]["reader"]
            base, pooled, scalar, capture = reader_semantic_forward(torch, tokenizer, reader, device, question["question"], texts,
                  max_length=spec["max_length"], batch_size=spec["batch_size"], capture_first=row["usable_case_index"] <= 2)
            values = {"base": base, "pooled": pooled, "scalar": scalar, **(capture or {})}
            cases.append(record_case(torch, heads, row, question, records, values, arrays, device, f"audit_{k + 1:03d}"))
            print(f"Train-only export: {k + 1}/32", flush=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = args.output_root / run_id
        write_outputs(directory, cases, arrays, cohort, future, provenance, source_arrays)
    if args.upload:
        upload_outputs(directory, args)
    print(json.dumps({"output_dir": str(directory), "uploaded": args.upload, "training_called": False,
                      "support_review": "pending", "github_files": list(COMPACT_FILES)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=ROOT / "exchange/five_ideas/semantic_reader_supervision_audit")
    p.add_argument("--wiki-dpr-cache", help="optional existing nondefault HF dataset cache; never downloaded")
    p.add_argument("--device")
    p.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", "http://117.50.198.37:18083"))
    p.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--upload-only", type=Path, help="resume uploads of an unchanged completed local run; no model/data pass")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--preflight", type=Path)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Supervision audit failed: {type(exc).__name__}: {exc}; existing local outputs are preserved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
