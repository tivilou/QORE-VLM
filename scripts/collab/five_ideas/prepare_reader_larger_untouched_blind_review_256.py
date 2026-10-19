#!/usr/bin/env python3
"""Create the hash-locked 256-question transport blind-review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/experiments/reader_larger_untouched_blind_review_256.json"
BANNED_REVIEW_KEYS = {
    "answers", "answer", "gold", "silver", "support_annotations",
    "candidate_id", "candidate_id_sha256", "source_id_sha256", "source_index",
    "dataset_index", "retrieved_rank", "candidate_retrieved_ranks", "reader_rank",
    "reader_score", "retrieval_score", "score", "scores", "prefix",
    "identity_top5", "baseline_top5_indices", "quantum_top5", "classical_top5",
    "selections", "selected_by", "method", "methods", "input_sha256",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _label_boundary_closed(trace: dict) -> bool:
    boundary = trace.get("label_boundary", {})
    required_false = {
        "answers_read", "support_annotations_read", "silver_oracle_read",
        "generator_called", "semantic_review_started",
    }
    return all(boundary.get(name) is False for name in required_false)


def build_packet(cohort_path: Path, trace_path: Path, output: Path, config_path: Path = CONFIG) -> dict:
    cfg = load(config_path)
    if digest(cohort_path) != cfg["cohort"]["sha256"]:
        raise ValueError("cohort hash mismatch")
    if digest(trace_path) != cfg["transport_run"]["trace_sha256"]:
        raise ValueError("transport trace hash mismatch")
    cohort, trace = load(cohort_path), load(trace_path)
    expected_questions = int(cfg["cohort"]["expected_questions"])
    expected = cfg["transport_run"]
    cases, samples = cohort.get("cases"), trace.get("samples")
    if trace.get("run_id") != expected["run_id"]:
        raise ValueError("transport run identity drift")
    if trace.get("population", {}).get("unseen_transport") != expected_questions:
        raise ValueError("transport population drift")
    if not isinstance(cases, list) or not isinstance(samples, list) or len(cases) != expected_questions or len(samples) != expected_questions:
        raise ValueError("cohort or trace coverage drift")
    if not trace.get("gate", {}).get("advance_to_blind_review"):
        raise ValueError("transport gate did not authorize packet construction")
    if not _label_boundary_closed(trace):
        raise ValueError("transport trace crossed the frozen label boundary")

    seed = int(cfg["blind_seed"])
    packet_cases: list[dict] = []
    key_cases: list[dict] = []
    for number, (case, sample) in enumerate(zip(cases, samples), 1):
        question, candidates = case.get("question"), case.get("candidates")
        if not isinstance(question, str) or not question.strip() or not isinstance(candidates, list) or len(candidates) != 50:
            raise ValueError(f"invalid source case {number}")
        question_sha = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if question_sha != case.get("question_sha256") or question_sha != sample.get("question_sha256"):
            raise ValueError(f"question identity drift at case {number}")
        source_digests = [hashlib.sha256(str(item["id"]).encode("utf-8")).hexdigest() for item in candidates]
        if source_digests != sample.get("candidate_id_sha256"):
            raise ValueError(f"candidate identity drift at case {number}")
        for index, candidate in enumerate(candidates):
            if candidate.get("candidate_id_sha256") != source_digests[index]:
                raise ValueError(f"cohort candidate digest drift at case {number}:{index}")
        selections = {
            "identity_reader": [int(value) for value in sample["identity_top5"]],
            "quantum": [int(value) for value in sample["quantum_top5"]],
            "classical": [int(value) for value in sample["classical_top5"]],
        }
        if any(len(values) != 5 or len(set(values)) != 5 or min(values) < 0 or max(values) >= 50
               for values in selections.values()):
            raise ValueError(f"invalid Top5 at case {number}")
        if len({tuple(sorted(values)) for values in selections.values()}) == 1:
            continue
        union = sorted(set().union(*(set(values) for values in selections.values())))
        case_id = opaque("case", seed, question_sha)
        items, key_items = [], []
        for index in union:
            candidate = candidates[index]
            title = str(candidate.get("title") or "").strip()
            passage = candidate.get("text")
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError(f"empty passage at case {number}:{index}")
            item_id = opaque("item", seed, question_sha, source_digests[index])
            items.append({"review_item_id": item_id, "title": title, "passage": passage.strip()})
            key_items.append({
                "review_item_id": item_id,
                "source_index": index,
                "candidate_id_sha256": source_digests[index],
                "selected_by": {method: index in values for method, values in selections.items()},
            })
        random.Random(seed ^ int(question_sha[:8], 16)).shuffle(items)
        packet_cases.append({"review_case_id": case_id, "question": question.strip(), "items": items})
        key_cases.append({
            "review_case_id": case_id,
            "source_case_number": number,
            "question_sha256": question_sha,
            "selections": selections,
            "items": key_items,
        })

    if len(packet_cases) != int(expected["expected_changed_cases"]):
        raise ValueError("changed-case count drift")
    item_count = sum(len(case["items"]) for case in packet_cases)
    if item_count != int(expected["expected_review_items"]):
        raise ValueError("review-item count drift")
    ordering = list(range(len(packet_cases)))
    random.Random(seed).shuffle(ordering)
    packet_cases = [packet_cases[index] for index in ordering]
    keyed = {case["review_case_id"]: case for case in key_cases}
    key_cases = [keyed[case["review_case_id"]] for case in packet_cases]

    packet = {
        "schema_version": "rag.reader_larger_untouched_blind_packet.v1",
        "packet_id": "larger-untouched-256-transport-20261019-v1",
        "claim_ceiling": cfg["claim_ceiling"],
        "instructions": {
            "answer_support": "direct if this passage itself gives explicit evidence answering the question; partial if substantively helpful but incomplete; none if it does not support an answer; uncertain only when the supplied text cannot be judged",
            "scope_match": "yes if entity, time, relation and requested answer type match; partial if only part matches; no if the passage addresses a different scope; uncertain only when genuinely unresolved",
            "independence": "Use only the supplied question and passage. Do not search externally, infer method identity, consult another reviewer, or access the sealed method key.",
        },
        "cases": packet_cases,
    }
    leaked = sorted(BANNED_REVIEW_KEYS & key_names(packet))
    if leaked:
        raise ValueError("review packet leaks banned fields: " + ", ".join(leaked))

    output.mkdir(parents=True, exist_ok=False)
    public_dir, private_dir = output / "public", output / "private"
    public_dir.mkdir()
    private_dir.mkdir()
    packet_path = public_dir / "blind_packet.json"
    write_json(packet_path, packet)
    template = {
        "schema_version": "rag.reader_larger_untouched_blind_review.v1",
        "packet_sha256": digest(packet_path),
        "reviewer_id": None,
        "completed": False,
        "judgments": [
            {
                "review_case_id": case["review_case_id"],
                "review_item_id": item["review_item_id"],
                "answer_support": None,
                "scope_match": None,
                "candidate_answer": "",
                "evidence_text": "",
                "rationale": "",
            }
            for case in packet_cases for item in case["items"]
        ],
    }
    write_json(public_dir / "review_template.json", template)
    sealed_key = {
        "schema_version": "rag.reader_larger_untouched_sealed_method_key.v1",
        "packet_sha256": digest(packet_path),
        "transport_trace_sha256": digest(trace_path),
        "cohort_sha256": digest(cohort_path),
        "population_questions": expected_questions,
        "cases": key_cases,
        "opening_rule": cfg["blinding"]["unblind_only_after"],
    }
    write_json(private_dir / "sealed_method_key.json", sealed_key)
    task = (
        "# Larger untouched blind passage review\n\n"
        "Judge every item in `blind_packet.json` and fill a private copy of `review_template.json`. "
        "Use only the supplied question, title, and passage. Do not search externally, consult another "
        "reviewer, inspect project traces, or open any sealed/private file. `uncertain` is reserved for "
        "genuinely unresolvable supplied text, not ordinary doubt. Set `completed=true` only after all items "
        "have both semantic codes and a non-empty rationale.\n"
    )
    (public_dir / "TASK.md").write_text(task, encoding="utf-8")
    manifest = {
        "schema_version": "rag.reader_larger_untouched_blind_packet_manifest.v1",
        "packet_id": packet["packet_id"],
        "population_questions": expected_questions,
        "changed_cases": len(packet_cases),
        "unchanged_cases": expected_questions - len(packet_cases),
        "review_items": item_count,
        "files": {},
        "blind_audit": {
            "banned_keys_absent": True,
            "answers_read_for_packet": False,
            "scores_exposed": False,
            "ranks_exposed": False,
            "methods_exposed": False,
            "sealed_key_separated_from_public_packet": True,
        },
    }
    for relative in ("public/blind_packet.json", "public/review_template.json", "public/TASK.md",
                     "private/sealed_method_key.json"):
        path = output / relative
        manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    write_json(output / "packet_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_packet(args.cohort, args.trace, args.output, args.config), ensure_ascii=False, indent=2))
