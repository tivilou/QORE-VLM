#!/usr/bin/env python3
"""Create a deterministic score/rank/method-blind review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/reader_unseen_blind_review_87.json"
BANNED_REVIEW_KEYS = {
    "answers", "candidate_id", "candidate_id_sha256", "retrieved_rank",
    "reader_rank", "reader_score", "retrieval_score", "score", "scores",
    "identity_top5", "baseline_top5_indices", "quantum_top5", "classical_top5",
    "selected_by", "method", "methods", "source_index",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def opaque(prefix: str, seed: int, *parts: str) -> str:
    raw = "\n".join([prefix, str(seed), *parts]).encode("utf-8")
    return prefix + "-" + hashlib.sha256(raw).hexdigest()[:20]


def key_names(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found.update(key_names(item))
    elif isinstance(value, list):
        for item in value:
            found.update(key_names(item))
    return found


def build_packet(bundle: Path, trace_path: Path, output: Path, config_path: Path = CONFIG) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if digest(bundle) != cfg["source_bundle_sha256"]:
        raise ValueError("source bundle hash mismatch")
    if digest(trace_path) != cfg["transport_run"]["trace_sha256"]:
        raise ValueError("transport trace hash mismatch")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    expected = cfg["transport_run"]
    if trace.get("run_id") != expected["run_id"] or trace.get("population", {}).get("validation") != expected["expected_validation_cases"]:
        raise ValueError("transport identity drift")
    if not trace.get("gate", {}).get("advance_to_blind_review"):
        raise ValueError("transport gate did not authorize packet construction")
    if any(trace.get("label_boundary", {}).values()):
        raise ValueError("transport trace crossed label boundary")

    with zipfile.ZipFile(bundle) as archive:
        source = json.loads(archive.read("source.json"))
    cases = [case for case in source["cases"] if case.get("future_role") == "validation"]
    if len(cases) != expected["expected_validation_cases"] or len(trace["samples"]) != len(cases):
        raise ValueError("validation coverage drift")

    seed = int(cfg["blind_seed"])
    packet_cases: list[dict] = []
    key_cases: list[dict] = []
    for number, (case, sample) in enumerate(zip(cases, trace["samples"]), 1):
        question = case.get("question")
        candidates = case.get("candidates")
        if not isinstance(question, str) or not question.strip() or not isinstance(candidates, list) or len(candidates) != 50:
            raise ValueError(f"invalid source case {number}")
        question_sha = hashlib.sha256(question.encode()).hexdigest()
        if question_sha != sample["question_sha256"] or question_sha != case["question_sha256"]:
            raise ValueError(f"question identity drift at {number}")
        source_digests = [hashlib.sha256(str(item["id"]).encode()).hexdigest() for item in candidates]
        if source_digests != sample["candidate_id_sha256"]:
            raise ValueError(f"candidate identity drift at {number}")
        selections = {
            "identity_reader": [int(v) for v in sample["identity_top5"]],
            "quantum": [int(v) for v in sample["quantum_top5"]],
            "classical": [int(v) for v in sample["classical_top5"]],
        }
        if len({tuple(sorted(v)) for v in selections.values()}) == 1:
            continue
        union = sorted(set().union(*(set(v) for v in selections.values())))
        case_id = opaque("case", seed, question_sha)
        items = []
        key_items = []
        for index in union:
            candidate = candidates[index]
            item_id = opaque("item", seed, question_sha, source_digests[index])
            title = str(candidate.get("title") or "").strip()
            passage = candidate.get("text")
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError(f"empty passage at {number}:{index}")
            items.append({"review_item_id": item_id, "title": title, "passage": passage.strip()})
            key_items.append({
                "review_item_id": item_id,
                "source_index": index,
                "candidate_id_sha256": source_digests[index],
                "selected_by": {method: index in chosen for method, chosen in selections.items()},
            })
        random.Random(seed ^ int(question_sha[:8], 16)).shuffle(items)
        packet_cases.append({"review_case_id": case_id, "question": question.strip(), "items": items})
        key_cases.append({"review_case_id": case_id, "source_case_number": number, "question_sha256": question_sha,
                          "selections": selections, "items": key_items})

    if len(packet_cases) != expected["expected_changed_cases"]:
        raise ValueError("changed-case count drift")
    ordering = list(range(len(packet_cases)))
    random.Random(seed).shuffle(ordering)
    packet_cases = [packet_cases[i] for i in ordering]
    key_lookup = {case["review_case_id"]: case for case in key_cases}
    key_cases = [key_lookup[case["review_case_id"]] for case in packet_cases]
    packet = {
        "schema_version": "rag.reader_unseen_blind_packet.v1",
        "packet_id": "validation87-transport-20261018-v1",
        "claim_ceiling": cfg["claim_ceiling"],
        "instructions": {
            "answer_support": "direct if this passage itself gives explicit evidence answering the question; partial if substantively helpful but incomplete; none if it does not support an answer; uncertain only when the text cannot be judged",
            "scope_match": "yes if entity, time, relation and requested answer type match; partial if only part matches; no if the passage answers a different scope; uncertain only when genuinely unresolved",
            "independence": "Do not search externally, infer method identity, reorder by perceived rank, or consult the sealed method key"
        },
        "cases": packet_cases,
    }
    leaked = sorted(BANNED_REVIEW_KEYS & key_names(packet))
    if leaked:
        raise ValueError("review packet leaks banned fields: " + ", ".join(leaked))

    output.mkdir(parents=True, exist_ok=False)
    packet_path = output / "blind_packet.json"
    write_json(packet_path, packet)
    template = {
        "schema_version": "rag.reader_unseen_blind_review.v1",
        "packet_sha256": digest(packet_path),
        "reviewer_id": None,
        "completed": False,
        "judgments": [
            {"review_case_id": case["review_case_id"], "review_item_id": item["review_item_id"],
             "answer_support": None, "scope_match": None, "candidate_answer": "", "evidence_text": "", "rationale": ""}
            for case in packet_cases for item in case["items"]
        ],
    }
    write_json(output / "review_template.json", template)
    sealed_key = {
        "schema_version": "rag.reader_unseen_sealed_method_key.v1",
        "packet_sha256": digest(packet_path),
        "transport_trace_sha256": digest(trace_path),
        "cases": key_cases,
        "opening_rule": cfg["blinding"]["unblind_only_after"],
    }
    write_json(output / "sealed_method_key.json", sealed_key)
    item_count = sum(len(case["items"]) for case in packet_cases)
    manifest = {
        "schema_version": "rag.reader_unseen_blind_packet_manifest.v1",
        "packet_id": packet["packet_id"],
        "changed_cases": len(packet_cases),
        "review_items": item_count,
        "files": {},
        "blind_audit": {"banned_keys_absent": True, "answers_read_for_packet": False, "scores_exposed": False,
                        "ranks_exposed": False, "methods_exposed": False},
    }
    for name in ("blind_packet.json", "review_template.json", "sealed_method_key.json"):
        path = output / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    write_json(output / "packet_manifest.json", manifest)
    (output / "TASK.md").write_text(
        "# Validation87 blind passage review\n\n"
        "Review `blind_packet.json` without opening `sealed_method_key.json`. Copy `review_template.json` "
        "to a reviewer-specific output and fill every semantic field. Use only the supplied question and passage. "
        "Do not search externally. `uncertain` is reserved for genuinely unresolvable text, not ordinary doubt.\n",
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.bundle, args.trace, args.output, args.config), ensure_ascii=False, indent=2))
