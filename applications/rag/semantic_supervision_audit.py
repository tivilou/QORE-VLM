"""Train-only observation helpers. No fitting, Silver access or selector mutation."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

METHODS = ("frozen_reader_topk", "quantum_semantic", "classical_semantic",
           "quantum_scalar_control", "monotonic_compression_control")
TRAINABLE = METHODS[1:4]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
SUPPORT_LABELS = ("direct", "partial", "irrelevant", "contradictory", "uncertain")
# Compressed Wiki-DPR inner products collapse to float32 resolution, so a genuine
# near-tie can differ by ~1 ULP (~8e-6) and is not order-stable across runs. Two
# candidates within this tolerance are interchangeable in retrieval order; real
# retrieval differences are orders of magnitude larger, so the gate stays strict.
IDENTITY_SCORE_TOLERANCE = 1.0e-3


def digest_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def normalized(value):
    return " ".join(TOKEN_RE.findall(str(value).lower().replace("-", " ")))


def weak_labels(texts, answers):
    """Exactly reproduce the old substring target, including its limitations."""
    aliases = [normalized(a) for a in answers if normalized(a)]
    mask = [any(a in normalized(t) for a in aliases) for t in texts]
    reason = ("no_nonempty_answers" if not aliases else "no_containment_hit" if not any(mask)
              else "all_candidates_positive" if all(mask) else "ok")
    return mask, reason


def containment_witnesses(text, answers):
    value = normalized(text)
    return [{"alias": str(a), "normalized_alias": normalized(a),
             "normalized_substring_start": value.find(normalized(a)),
             "match_is_support_proof": False}
            for a in answers if normalized(a) and normalized(a) in value]


def rank_order(scores, candidates):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (50,) or len(candidates) != 50 or not np.isfinite(scores).all():
        raise ValueError("expected 50 finite candidate scores")
    if [p["retrieved_rank"] for p in candidates] != list(range(1, 51)):
        raise ValueError("retrieval order changed")
    if len({str(p["id"]) for p in candidates}) != 50:
        raise ValueError("duplicate candidate IDs")
    return sorted(range(50), key=lambda i: (-float(scores[i]), candidates[i]["retrieved_rank"], str(candidates[i]["id"])))


def validate_cohort(cohort):
    identities = cohort["training_identity"]
    if len(identities) != cohort["usable_cases"] or len(identities) < 32:
        raise ValueError("source training cohort size mismatch")
    questions = [row["question_sha256"] for row in identities]
    if len(set(questions)) != len(questions) or set(questions) & set(cohort["evaluation_question_sha256"]):
        raise ValueError("cohort duplicate or training/evaluation leakage")
    if len(set(cohort["evaluation_question_sha256"])) != 100:
        raise ValueError("evaluation exclusion identity mismatch")
    for row in identities:
        ids, mask = row["candidate_id_sha256"], row["positive_mask"]
        if len(ids) != 50 or len(set(ids)) != 50 or len(mask) != 50 or any(type(v) is not bool for v in mask):
            raise ValueError("cohort candidate/weak-target schema mismatch")
        if not any(mask) or all(mask):
            raise ValueError("source usable case has no mixed weak target")
    return identities


def sample_cohort(cohort, *, seed=20261009, count=32):
    """Two replay witnesses + 15 each from prospective split; score-independent."""
    rows = validate_cohort(cohort)
    if count != 32:
        raise ValueError("sampling count is preregistered at 32")
    ordered = sorted(range(len(rows)), key=lambda i: digest_text(f"{seed}:future-split:{rows[i]['question_sha256']}"))
    validation = set(ordered[:math.ceil(len(rows) * .2)])
    selected = [0, 1]
    for role in (False, True):
        pool = [i for i in ordered if (i in validation) == role and i not in selected]
        pool.sort(key=lambda i: digest_text(f"{seed}:audit:{rows[i]['question_sha256']}"))
        if len(pool) < 15:
            raise ValueError("insufficient prospective split for fixed audit rule")
        selected.extend(pool[:15])
    future = {"rule": "lowest 20% rounded up of SHA256(seed:future-split:question_sha256)",
              "seed": seed, "validation_count": len(validation),
              "previous_heads_trained_on_both_roles": True,
              "independent_validation_claim": False,
              "assignments": [{"usable_case_index": i + 1, "question_sha256": r["question_sha256"],
                               "future_role": "validation" if i in validation else "training"}
                              for i, r in enumerate(rows)]}
    return [{"usable_case_index": i + 1, "question_sha256": rows[i]["question_sha256"],
             "future_role": "validation" if i in validation else "training",
             "sample_role": "prior_replay_witness" if i < 2 else "hash_sample"} for i in selected], future


def check_identity(question, candidates, mask, identity, *, score_tolerance=IDENTITY_SCORE_TOLERANCE):
    """Confirm the current retrieval reproduces the registered training case.

    The 50 candidate IDs must be the same multiset as the prior run. Order may
    differ only between candidates whose retrieval scores tie within
    ``score_tolerance``; a reordering of genuinely distinct scores, a missing
    candidate or a changed weak target all fail. The weak target is matched per
    candidate identity, so a tolerated tie swap never reassigns a weak label.
    """
    if digest_text(question) != identity["question_sha256"]:
        raise ValueError("question identity mismatch")
    current = [digest_text(p["id"]) for p in candidates]
    prior = list(identity["candidate_id_sha256"])
    if current != prior:
        if sorted(current) != sorted(prior):
            raise ValueError("retrieved candidate identity/order differs from prior run")
        try:
            score_by_id = {digest_text(p["id"]): float(p["retrieval_score"]) for p in candidates}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retrieved candidate identity/order differs from prior run") from exc
        ordered = [score_by_id[cid] for cid in prior]
        if any(later > earlier + score_tolerance for earlier, later in zip(ordered, ordered[1:])):
            raise ValueError("retrieved candidate identity/order differs from prior run")
    prior_mask = dict(zip(prior, identity["positive_mask"]))
    if any(prior_mask.get(digest_text(p["id"])) != bool(m) for p, m in zip(candidates, mask)):
        raise ValueError("prior/current weak target mismatch")


def compression_diagnostic(base, scores):
    x, y = np.asarray(base, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if x.shape != (50,) or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid compression inputs")
    centered = x - x.mean()
    slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered)) if np.any(centered) else 0.0
    intercept = float(y.mean() - slope * x.mean())
    corr = float(np.corrcoef(x, y - x)[0, 1]) if np.std(x) > 0 and np.std(y - x) > 0 else None
    return {"score_on_base_slope": slope, "intercept": intercept,
            "delta_base_pearson": corr,
            "non_affine_rmse": float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2))),
            "delta_std": float(np.std(y - x)), "fitted_slope_preserves_order": slope > 0}


def boundary_diagnostic(base, scores, mask, candidates, *, residual_bound=.25):
    order = rank_order(base, candidates)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (50,):
        raise ValueError("invalid weak-target shape")
    base, scores = np.asarray(base, dtype=float), np.asarray(scores, dtype=float)
    rank_order(scores, candidates)  # finite/shape checks even for no-pair cases
    # Strongest distractor + weakest replaceable negative; two near and one deep positive.
    all_negatives = [i for i in order[:5] if not mask[i]]
    all_positives = [i for i in order[5:] if mask[i]]
    negatives = list(dict.fromkeys(all_negatives[:1] + all_negatives[-1:]))
    positives = list(dict.fromkeys(all_positives[:2] + all_positives[-1:]))
    ranks = {i: j + 1 for j, i in enumerate(order)}
    pairs = []
    for p in positives:
        band = (6, 10) if ranks[p] <= 10 else (11, 20) if ranks[p] <= 20 else (21, 50)
        controls = [i for i in order[5:] if not mask[i] and band[0] <= ranks[i] <= band[1]]
        c = min(controls, key=lambda i: (abs(base[i] - base[p]), ranks[i])) if controls else None
        for n in negatives:
            margin = float(scores[p] - scores[n])
            delta = float((scores[p] - base[p]) - (scores[n] - base[n]))
            control = None if c is None else {"candidate_index": c, "reader_rank": ranks[c],
                       "score_distance_to_positive": float(abs(base[c] - base[p])),
                       "relative_correction": float((scores[c] - base[c]) - (scores[n] - base[n]))}
            pairs.append({"weak_positive_index": p, "weak_negative_index": n,
                          "positive_reader_rank": ranks[p], "negative_reader_rank": ranks[n],
                          "base_margin": float(base[p] - base[n]), "new_margin": margin,
                          "relative_correction": delta, "crossed": margin > 0,
                          "max_pairwise_bound": float(2 * residual_bound * max(np.std(base), 1e-6)),
                          "within_bound_necessary_not_sufficient": bool(base[n] - base[p] <= 2 * residual_bound * max(np.std(base), 1e-6)),
                          "matched_weak_negative_control": control,
                          "labels_are_not_certified_support": True})
    return {"pairs": pairs, "pair_count": len(pairs),
            "positive_direction_pairs": sum(p["relative_correction"] > 0 for p in pairs),
            "strict_crossings": sum(p["crossed"] for p in pairs)}


def blinded_review(cases, seed=20261009):
    """No ranks, scores, weak labels or method identities in the review view."""
    output = []
    for case in cases:
        items = sorted(case["candidates"], key=lambda p: digest_text(f"{seed}:blind:{case['question_sha256']}:{p['id']}"))
        output.append({"review_case_id": case["question_sha256"], "question": case["question"],
                       "reference_answers": case["answers"],
                       "passages": [{"review_item_id": digest_text(case["question_sha256"] + ":" + str(p["id"])),
                                     "title": p["title"], "text": p["text"],
                                     "support_label": None, "support_quote": None, "rationale": None}
                                    for p in items]})
    return {"schema_version": "rag.train_support_review.v1", "status": "pending_review",
            "target_is_official_gold": False, "allowed_labels": list(SUPPORT_LABELS),
            "instructions": "Judge support for the question, not answer-string occurrence or topical similarity. "
                            "Direct requires sufficient explicit support; partial is helpful but insufficient; "
                            "irrelevant is no supporting relation; contradictory conflicts with the answer; "
                            "uncertain preserves missing/time-dependent context. Quote the passage verbatim. "
                            "No label should be filled merely because the reference answer occurs.",
            "cases": output}


def validate_reviews(template, completed):
    """A later review must retain exact blinded inputs and a genuine source quote."""
    if len(template["cases"]) != len(completed.get("cases", [])):
        raise ValueError("review case coverage mismatch")
    result = {k: 0 for k in SUPPORT_LABELS}
    for before, after in zip(template["cases"], completed["cases"]):
        for key in ("review_case_id", "question", "reference_answers"):
            if before[key] != after.get(key):
                raise ValueError("review case input changed")
        if len(before["passages"]) != len(after.get("passages", [])):
            raise ValueError("review passage coverage mismatch")
        for p, q in zip(before["passages"], after["passages"]):
            if any(p[k] != q.get(k) for k in ("review_item_id", "title", "text")):
                raise ValueError("review passage input changed")
            label, quote = q.get("support_label"), q.get("support_quote")
            if label not in SUPPORT_LABELS or not isinstance(q.get("rationale"), str) or not q["rationale"].strip():
                raise ValueError("review label/rationale missing")
            if label in ("direct", "partial", "contradictory") and (not isinstance(quote, str) or not quote.strip() or quote not in p["text"]):
                raise ValueError("support claim requires a nonempty verbatim text quote")
            result[label] += 1
    return result
