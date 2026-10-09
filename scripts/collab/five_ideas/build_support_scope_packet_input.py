"""Extract the locked9/54 witness union locally; never infer or copy labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--records-dir", type=Path, required=True)
    p.add_argument("--packet", type=Path, required=True)
    args = p.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    source = args.source_dir / "case_study.json"
    locked = load(args.records_dir / "semantic_supervision_20261009T045728Z_complete_review_manifest.json")
    if hashlib.sha256(source.read_bytes()).hexdigest() != locked["source_case_sha256"]:
        raise ValueError("immutable source mismatch")
    trace = load(source)
    draft = load(args.records_dir / "support_boundary_targets_20261009_boundary_target_draft.json")
    cases = []
    for row in draft["cases"]:
        if row["audit_case_number"] not in [5, 6, 8, 13, 16, 17, 21, 26, 27]:
            continue
        c = trace["cases"][row["audit_case_number"] - 1]
        indices = set(row["provisional_direct_indices"] + [i for pair in row["proposal_pairs"] for i in pair])
        items = [{"review_item_id": c["candidates"][i]["question_passage_review_id"],
                  "source_text_sha256": c["candidates"][i]["text_sha256"],
                  "title": c["candidates"][i]["title"], "text": c["candidates"][i]["text"]} for i in indices]
        items.sort(key=lambda item: item["review_item_id"])
        cases.append({"audit_case_number": row["audit_case_number"], "review_case_id": c["question_sha256"],
                      "question": c["question"], "reference_answers": c["answers"], "items": items})
    data = {"schema_version": "rag.blinded_scope_witness_input.v1", "training_authorized": False,
            "scores_and_prior_labels_included": False,
            "selection": "predeclared priority9 boundary/retention union,54 witnesses; selected by earlier provisional review,not representative",
            "source_case_sha256": locked["source_case_sha256"], "cases": cases}
    if len(cases) != 9 or sum(len(c["items"]) for c in cases) != 54:
        raise ValueError("locked witness population changed")
    target = args.packet / "context/evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"questions": 9, "witnesses": 54, "bytes": target.stat().st_size,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
