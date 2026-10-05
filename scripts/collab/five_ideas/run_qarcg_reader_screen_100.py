#!/usr/bin/env python3
"""Train and screen the Q-ARCG Reader head on a fixed 100-case panel.

The frozen DPR Reader is the only model that produces the base relevance and
span logits.  Q-ARCG and its matched classical control are trained on a
separate ``nq_open/train`` weak-supervision slice, where a passage is positive
only when an answer string occurs in its retrieved text.  This target is
explicitly not official passage gold.  The registered 100 x 50 detail input
is evaluation-only; Silver labels are opened only after all online rankings
have been produced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/experiments/qarcg_reader_screen_100.json"
DEFAULT_PLAN = ROOT / "configs/experiments/qarcg_reader_screen_100_plan.json"
DEFAULT_INPUT = ROOT / "research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_INPUT_EXCHANGE_PATH = "five_ideas/selector_replay_100_historical_input/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_INPUT_BYTES = 8046318
DEFAULT_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/qarcg_reader_screen_100"
DEFAULT_EXCHANGE_URL = "http://117.50.198.37:18083"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
TOP5 = 5
MAX_GITHUB_BYTES = 1_048_576
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
FORBIDDEN_COMPACT_FIELDS = frozenset({
    "question", "text", "passages", "gold_answers", "answers", "prediction",
    "raw_prompt", "prompt", "token_ids", "selected_ids", "retrieved_ids",
})


class ScreenError(RuntimeError):
    """Raised when the real-data screen contract cannot be satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScreenError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ScreenError("JSON root must be an object")
    return value


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _validate_config(path: Path) -> dict[str, Any]:
    document = _load_json(path)
    phase = document.get("phase")
    if not isinstance(phase, dict) or phase.get("name") != "qarcg_reader_screen_100":
        raise ScreenError("unexpected Q-ARCG screen config")
    if int(phase.get("schema_version", -1)) != 1 or phase.get("authorization") != "approved":
        raise ScreenError("screen must use the approved schema v1 config")
    if phase.get("diagnostic_only") is not True or phase.get("selection_mutation") is not False:
        raise ScreenError("screen must remain diagnostic-only and non-mutating")
    dataset = phase.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") != "nq_open":
        raise ScreenError("dataset identity is not frozen")
    if int(dataset.get("evaluation_cases", -1)) != EXPECTED_CASES or int(dataset.get("evaluation_top_k", -1)) != EXPECTED_TOP50:
        raise ScreenError("evaluation must be exactly 100 cases x 50 candidates")
    if dataset.get("evaluation_input_sha256") != DEFAULT_INPUT_SHA256:
        raise ScreenError("evaluation detail input hash changed")
    training = phase.get("training")
    required_training = {
        "weak_target": "answer_string_containment_in_retrieved_passage",
        "weak_target_is_not_official_gold": True,
        "seed": 20261005,
        "max_questions": 512,
    }
    if not isinstance(training, dict) or any(training.get(key) != value for key, value in required_training.items()):
        raise ScreenError("training weak-supervision contract changed")
    reader = phase.get("reader")
    if not isinstance(reader, dict) or reader.get("model_id") != "facebook/dpr-reader-single-nq-base":
        raise ScreenError("DPR Reader identity is not frozen")
    head = phase.get("head")
    if not isinstance(head, dict) or int(head.get("n_qubits", -1)) != 4 or not bool(head.get("same_budget_classical_control")):
        raise ScreenError("Q-ARCG head/control budget is not frozen")
    boundary = phase.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("silver_used_for_training_or_selection") is not False:
        raise ScreenError("online label boundary is not frozen")
    return document


def _validate_plan(path: Path, phase: Mapping[str, Any]) -> dict[str, Any]:
    plan = _load_json(path)
    if plan.get("schema_version") != "research-plugin-architecture.plugin-plan.v2" or plan.get("authorization") != "approved":
        raise ScreenError("plugin plan must use approved schema v2")
    if plan.get("project") != "Q-DUET-VLM/rag-selector":
        raise ScreenError("plugin plan project identity changed")
    if plan.get("candidate_id", "q_arcg_reader_integrated_residual") != phase.get("candidate_id"):
        raise ScreenError("plugin plan and runtime config identify different candidates")
    plugins = plan.get("plugins")
    expected = ["frozen_reader_topk", "trained_q_arcg", "trained_same_budget_classical_control"]
    if not isinstance(plugins, list) or [item.get("id") for item in plugins if isinstance(item, Mapping)] != expected:
        raise ScreenError("plugin allowlist/order changed")
    if plan.get("training", {}).get("target_status") != "weak_supervision_only_not_official_gold":
        raise ScreenError("training target provenance is not explicit")
    if plan.get("evaluation", {}).get("silver_diagnostics") != "posthoc_only":
        raise ScreenError("Silver must remain posthoc-only")
    return plan


def _normalize_answer(value: Any) -> str:
    tokens = TOKEN_RE.findall(str(value).lower().replace("-", " "))
    return " ".join(tokens)


def _weak_positive_mask(passages: Sequence[str], answers: Sequence[Any]) -> tuple[list[bool], str]:
    normalized_answers = [_normalize_answer(answer) for answer in answers]
    normalized_answers = [answer for answer in normalized_answers if answer]
    if not normalized_answers:
        return [False] * len(passages), "no_nonempty_answers"
    mask: list[bool] = []
    for passage in passages:
        normalized_passage = _normalize_answer(passage)
        mask.append(any(answer in normalized_passage for answer in normalized_answers))
    if not any(mask):
        return mask, "no_containment_hit"
    if all(mask):
        return mask, "all_candidates_positive"
    return mask, "ok"


def _contains_forbidden(value: Any, path: str = "$root") -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_COMPACT_FIELDS:
                return True
            if _contains_forbidden(child, f"{path}.{key}"):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden(child, f"{path}[{index}]") for index, child in enumerate(value))
    return False


def _load_eval_cases(path: Path) -> list[Mapping[str, Any]]:
    document = _load_json(path)
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ScreenError(f"evaluation input must contain exactly {EXPECTED_CASES} cases")
    for case_number, case in enumerate(cases, 1):
        if not isinstance(case, Mapping) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ScreenError(f"case {case_number} has no valid question")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise ScreenError(f"case {case_number} must contain exactly 50 candidates")
        ranks = [item.get("retrieved_rank", item.get("rank")) for item in top50 if isinstance(item, Mapping)]
        ids = [str(item.get("id", item.get("passage_id"))) for item in top50 if isinstance(item, Mapping)]
        if len(ranks) != EXPECTED_TOP50 or sorted(ranks) != list(range(1, EXPECTED_TOP50 + 1)) or len(set(ids)) != EXPECTED_TOP50:
            raise ScreenError(f"case {case_number} failed candidate identity/rank contract")
        if any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in top50):
            raise ScreenError(f"case {case_number} has empty candidate text")
    return cases


def _prepare_input(input_path: Path | None, *, exchange_url: str, token_env: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    candidates = [
        input_path.resolve() if input_path is not None else None,
        DEFAULT_INPUT,
        ROOT / "exchange" / DEFAULT_INPUT_EXCHANGE_PATH,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and not candidate.is_symlink():
            if _sha256(candidate) != DEFAULT_INPUT_SHA256:
                raise ScreenError(f"evaluation input hash mismatch: {candidate}")
            return candidate, None
    try:
        from scripts.collab.lib.exchange_upload import download_exchange_file
    except ImportError as exc:
        raise ScreenError("registered detail input is absent; exchange helper unavailable") from exc
    temporary = tempfile.TemporaryDirectory(prefix="qarcg-reader-input-")
    destination = Path(temporary.name) / Path(DEFAULT_INPUT_EXCHANGE_PATH).name
    try:
        download_exchange_file(
            DEFAULT_INPUT_EXCHANGE_PATH, destination,
            base_url=exchange_url, token_env=token_env,
            expected_size=DEFAULT_INPUT_BYTES, expected_sha256=DEFAULT_INPUT_SHA256,
        )
    except Exception:
        temporary.cleanup()
        raise
    return destination, temporary


def _online_text(candidate: Mapping[str, Any]) -> str:
    title = str(candidate.get("title") or "").strip()
    body = str(candidate.get("text") or "").strip()
    if not body:
        raise ScreenError("candidate text is empty")
    return f"{title}. {body}" if title else body


def _reader_identity(reader: Any, tokenizer: Any, model_id: str, revision: str | None, device: Any) -> dict[str, Any]:
    config_dict = reader.config.to_dict() if hasattr(reader.config, "to_dict") else dict(reader.config)
    config_bytes = json.dumps(config_dict, sort_keys=True, default=str).encode("utf-8")
    resolved_revision = (
        getattr(reader.config, "_commit_hash", None)
        or getattr(reader, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or revision
    )
    return {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "device": str(device),
        "reader_frozen": True,
    }


def _load_reader(phase: Mapping[str, Any], device_name: str | None):
    try:
        import torch
        from transformers import DPRReader, DPRReaderTokenizer
    except ImportError as exc:
        raise ScreenError("Torch and Transformers are required for the collaborator screen") from exc
    reader_spec = phase["reader"]
    kwargs = {"revision": reader_spec["revision"]} if reader_spec.get("revision") else {}
    tokenizer = DPRReaderTokenizer.from_pretrained(reader_spec["model_id"], **kwargs)
    reader = DPRReader.from_pretrained(reader_spec["model_id"], **kwargs)
    device = torch.device(device_name) if device_name else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader.to(device)
    reader.eval()
    return torch, tokenizer, reader, device, _reader_identity(reader, tokenizer, reader_spec["model_id"], reader_spec.get("revision"), device)


def _reader_forward(torch: Any, tokenizer: Any, reader: Any, device: Any, question: str, texts: Sequence[str], max_length: int, max_answer_tokens: int) -> dict[str, Any]:
    from applications.rag.qarcg_reader import torch_reader_features

    encoded = tokenizer(
        questions=[question] * len(texts), texts=list(texts), return_tensors="pt",
        padding=True, truncation=True, max_length=max_length,
    )
    if "token_type_ids" not in encoded:
        raise ScreenError("DPR Reader tokenizer did not return token_type_ids")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    passage_mask = (encoded["token_type_ids"] == 1) & (encoded["attention_mask"] == 1)
    special_mask = torch.zeros_like(passage_mask)
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        special_mask |= encoded["input_ids"] == int(token_id)
    passage_mask &= ~special_mask
    with torch.no_grad():
        outputs = reader(**encoded, return_dict=True)
        features, span_summary = torch_reader_features(
            outputs.relevance_logits, outputs.start_logits, outputs.end_logits,
            passage_mask, max_answer_tokens=max_answer_tokens,
        )
    return {
        "base_scores": outputs.relevance_logits.detach(),
        "features": features.detach(),
        "span_summary": {key: value.detach().cpu().numpy() for key, value in span_summary.items()},
    }


def _retrieve(manager: Any, query_embedding: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    dataset = getattr(manager, "_dataset", None)
    if dataset is None or not hasattr(dataset, "get_nearest_examples"):
        raise ScreenError("Wiki-DPR manager does not expose get_nearest_examples")
    scores, retrieved = dataset.get_nearest_examples("embeddings", np.asarray(query_embedding, dtype=np.float32), k=top_k)
    required = ("id", "title", "text")
    if any(key not in retrieved for key in required) or len(scores) != top_k:
        raise ScreenError("Wiki-DPR retrieval did not return the required Top-50")
    records = []
    for rank in range(top_k):
        records.append({
            "id": str(retrieved["id"][rank]),
            "title": str(retrieved["title"][rank] or ""),
            "text": str(retrieved["text"][rank] or ""),
            "retrieved_rank": rank + 1,
            "retrieval_score": float(scores[rank]),
        })
    return records, np.asarray(scores, dtype=np.float32)


def _stable_top(scores: Any, candidates: Sequence[Mapping[str, Any]], k: int = TOP5) -> list[int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.shape != (EXPECTED_TOP50,) or len(candidates) != EXPECTED_TOP50 or not np.all(np.isfinite(values)):
        raise ScreenError("selector scores must be finite and have length 50")
    return sorted(range(len(values)), key=lambda index: (
        -float(values[index]),
        int(candidates[index].get("retrieved_rank", candidates[index].get("rank"))),
        str(candidates[index].get("id", candidates[index].get("passage_id"))),
    ))[:k]


def _train_head(torch: Any, module: Any, cases: Sequence[Mapping[str, Any]], *, phase: Mapping[str, Any], seed: int) -> dict[str, Any]:
    training = phase["training"]
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    losses: list[float] = []
    epochs = int(training["epochs"])
    for _epoch in range(epochs):
        epoch_losses: list[float] = []
        for case in cases:
            optimizer.zero_grad(set_to_none=True)
            scores, _gate, _residual, _obs = module(case["base_scores"], case["features"])
            from applications.rag.qarcg_reader import torch_pairwise_rank_loss
            ranking = torch_pairwise_rank_loss(scores, case["positive_mask"])
            anchor = (scores - case["base_scores"]).square().mean()
            loss = ranking + float(training["anchor_penalty"]) * anchor
            if not bool(torch.isfinite(loss)):
                raise ScreenError("non-finite Q-ARCG training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return {"epochs": epochs, "mean_loss_by_epoch": [round(value, 8) for value in losses], "case_count": len(cases)}


def _bootstrap_lower(values: Sequence[float], seed: int, reps: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, array.size, size=(reps, array.size))
    return float(np.quantile(array[sampled].mean(axis=1), 0.025))


def _posthoc_silver_ids(case: Mapping[str, Any]) -> list[str]:
    for selector in case.get("selectors", []):
        if isinstance(selector, Mapping) and selector.get("selector_id") == "silver_oracle_common_order":
            return [str(item.get("id", item.get("passage_id"))) for item in selector.get("selected_top_5", [])]
    raise ScreenError("evaluation case is missing silver_oracle_common_order")


def _evaluate_cases(torch: Any, tokenizer: Any, reader: Any, device: Any, q_head: Any, c_head: Any, cases: Sequence[Mapping[str, Any]], phase: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reader_spec = phase["reader"]
    q_head.eval()
    c_head.eval()
    aggregate: dict[str, list[float]] = {"frozen_reader_topk": [], "trained_q_arcg": [], "trained_same_budget_classical_control": []}
    trace_cases: list[dict[str, Any]] = []
    for case_number, case in enumerate(cases, 1):
        question = str(case["question"])
        candidates = list(case["top_50"])
        texts = [_online_text(candidate) for candidate in candidates]
        with torch.inference_mode():
            reader_pass = _reader_forward(torch, tokenizer, reader, device, question, texts, int(reader_spec["max_length"]), int(reader_spec["max_answer_tokens"]))
            q_scores, q_gate, q_residual, _ = q_head(reader_pass["base_scores"], reader_pass["features"])
            c_scores, c_gate, c_residual, _ = c_head(reader_pass["base_scores"], reader_pass["features"])
        base = reader_pass["base_scores"].detach().cpu().numpy().astype(np.float64)
        q_values = q_scores.detach().cpu().numpy().astype(np.float64)
        c_values = c_scores.detach().cpu().numpy().astype(np.float64)
        methods = {
            "frozen_reader_topk": {"scores": base, "gate": np.zeros(EXPECTED_TOP50), "residual": np.zeros(EXPECTED_TOP50)},
            "trained_q_arcg": {"scores": q_values, "gate": q_gate.detach().cpu().numpy(), "residual": q_residual.detach().cpu().numpy()},
            "trained_same_budget_classical_control": {"scores": c_values, "gate": c_gate.detach().cpu().numpy(), "residual": c_residual.detach().cpu().numpy()},
        }
        case_trace: dict[str, Any] = {
            "case_number": case_number,
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "candidate_id_sha256": [hashlib.sha256(str(candidate.get("id", candidate.get("passage_id"))).encode("utf-8")).hexdigest() for candidate in candidates],
            "methods": {},
        }
        for method_id, values in methods.items():
            indices = _stable_top(values["scores"], candidates)
            selected_ids = [str(candidates[index].get("id", candidates[index].get("passage_id"))) for index in indices]
            case_trace["methods"][method_id] = {
                "selected_retrieved_ranks": [int(candidates[index].get("retrieved_rank", candidates[index].get("rank"))) for index in indices],
                "selected_id_sha256": [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in selected_ids],
                "score_vector": [round(float(value), 7) for value in values["scores"]],
                "gate_mean": round(float(np.mean(values["gate"])), 7),
                "residual_max_abs": round(float(np.max(np.abs(values["residual"]))), 7),
            }
        trace_cases.append(case_trace)
    # All online rankings are complete. Only now open Silver labels for the
    # separate diagnostic pass; they never enter the head or later rankings.
    for case, case_trace in zip(cases, trace_cases):
        silver_ids = set(_posthoc_silver_ids(case))
        case_trace["posthoc_silver"] = {"silver_count": len(silver_ids)}
        for method_id, method_trace in case_trace["methods"].items():
            selected_hashes = set(method_trace["selected_id_sha256"])
            selected_ids = [
                str(candidate.get("id", candidate.get("passage_id")))
                for candidate in case["top_50"]
                if hashlib.sha256(str(candidate.get("id", candidate.get("passage_id"))).encode("utf-8")).hexdigest() in selected_hashes
            ]
            overlap = float(len(set(selected_ids).intersection(silver_ids)))
            method_trace["silver_overlap"] = int(overlap)
            aggregate[method_id].append(overlap)
    summary: dict[str, Any] = {}
    for method_id, values in aggregate.items():
        summary[method_id] = {"mean_silver_overlap": round(float(np.mean(values)), 6), "case_count": len(values), "overlap_histogram": dict(sorted(Counter(int(value) for value in values).items()))}
    baseline = np.asarray(aggregate["frozen_reader_topk"], dtype=np.float64)
    q_values = np.asarray(aggregate["trained_q_arcg"], dtype=np.float64)
    c_values = np.asarray(aggregate["trained_same_budget_classical_control"], dtype=np.float64)
    reps = int(phase["gates"]["bootstrap_replicates"])
    seed = int(phase["gates"]["bootstrap_seed"])
    q_delta = q_values - baseline
    c_delta = c_values - baseline
    q_c_delta = q_values - c_values
    result = {
        "methods": summary,
        "paired_deltas": {
            "q_arcg_minus_frozen_reader": {"mean": round(float(np.mean(q_delta)), 6), "bootstrap_ci95_lower": round(_bootstrap_lower(q_delta, seed, reps), 6)},
            "classical_minus_frozen_reader": {"mean": round(float(np.mean(c_delta)), 6), "bootstrap_ci95_lower": round(_bootstrap_lower(c_delta, seed + 1, reps), 6)},
            "q_arcg_minus_classical": {"mean": round(float(np.mean(q_c_delta)), 6), "bootstrap_ci95_lower": round(_bootstrap_lower(q_c_delta, seed + 2, reps), 6)},
        },
        "per_case_overlap": {key: [int(value) for value in values] for key, values in aggregate.items()},
    }
    return result, trace_cases


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Q-ARCG Reader-integrated 100-case screen", "",
        "This is an L0 diagnostic screen. The Reader is frozen; training uses only answer-string containment weak supervision on a separate NQ-Open train slice. Silver overlap is post-hoc and no Generator/evaluator was called.", "",
        "## Methods", "",
    ]
    for method_id, values in summary["evaluation"]["methods"].items():
        lines.append(f"- `{method_id}`: mean Silver overlap `{values['mean_silver_overlap']:.6f}/5` over `{values['case_count']}` cases")
    lines.extend(["", "## Paired diagnostics", ""])
    for name, values in summary["evaluation"]["paired_deltas"].items():
        lines.append(f"- `{name}`: mean `{values['mean']:.6f}`, bootstrap 95% lower `{values['bootstrap_ci95_lower']:.6f}`")
    gates = summary["gates"]
    lines.extend(["", "## Gates", "", f"- Utility gate: `{gates['utility']['pass']}`", f"- Quantum attribution gate: `{gates['quantum_attribution']['pass']}`", f"- Weak training coverage: `{gates['training_coverage']['pass']}`", "", "## Limitations", "", "- Answer-string containment is noisy weak supervision, not official passage gold.", "- Silver overlap is a post-hoc diagnostic and cannot establish population utility or L1/L2.", "- The classical arm is the required same-budget attribution control; a Q-ARCG win without it is not a quantum-advantage claim."])
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path | None:
    config_path = (args.config if args.config.is_absolute() else ROOT / args.config).resolve()
    document = _validate_config(config_path)
    phase = document["phase"]
    plan_path = (args.plan if args.plan.is_absolute() else ROOT / args.plan).resolve()
    _validate_plan(plan_path, phase)
    if args.validate_only:
        print(json.dumps({"status": "valid", "phase": phase["name"], "candidate_id": phase["candidate_id"], "evaluation_cases": EXPECTED_CASES, "training_max_questions": phase["training"]["max_questions"], "weak_target": phase["training"]["weak_target"]}, sort_keys=True))
        return None
    if args.model_id:
        phase["reader"]["model_id"] = args.model_id
    if args.revision:
        phase["reader"]["revision"] = args.revision
    input_path, temporary = _prepare_input(args.input, exchange_url=args.exchange_url, token_env=args.token_env)
    try:
        cases = _load_eval_cases(input_path)
        torch, tokenizer, reader, device, reader_identity = _load_reader(phase, args.device)
        from applications.rag.data import load_dataset_for_rag, make_corpus_manager
        from applications.rag.retrieval import make_encoder
        from applications.rag.qarcg_reader import QARCGConfig, make_torch_qarcg, make_torch_classical_control

        train_count = int(phase["training"]["max_questions"])
        training_questions = list(load_dataset_for_rag(phase["dataset"]["name"], phase["dataset"]["training_split"], train_count))[:train_count]
        if len(training_questions) != train_count:
            raise ScreenError(f"expected {train_count} training questions, got {len(training_questions)}")
        evaluation_question_hashes = {hashlib.sha256(str(case["question"]).encode("utf-8")).hexdigest() for case in cases}
        if any(hashlib.sha256(str(item.get("question", "")).encode("utf-8")).hexdigest() in evaluation_question_hashes for item in training_questions):
            raise ScreenError("training/evaluation question overlap detected")
        encoder = make_encoder("dpr")
        manager = make_corpus_manager("wiki_dpr", {"wiki_dpr_config": phase["retrieval"]["wiki_dpr_config"], "nprobe": int(phase["retrieval"]["nprobe"])})
        manager.build(training_questions)
        training_cases: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        started_training = datetime.now(timezone.utc).isoformat()
        for index, item in enumerate(training_questions, 1):
            question = str(item.get("question", ""))
            answers = item.get("answers", [])
            if not question.strip() or not isinstance(answers, (list, tuple)):
                skipped["invalid_question_or_answers"] += 1
                continue
            query_embedding = encoder.encode_queries([question])[0]
            records, _retrieval_scores = _retrieve(manager, query_embedding, EXPECTED_TOP50)
            texts = [_online_text(record) for record in records]
            positive, reason = _weak_positive_mask(texts, answers)
            if reason != "ok":
                skipped[reason] += 1
                continue
            reader_pass = _reader_forward(torch, tokenizer, reader, device, question, texts, int(phase["reader"]["max_length"]), int(phase["reader"]["max_answer_tokens"]))
            training_cases.append({
                "base_scores": reader_pass["base_scores"].detach(),
                "features": reader_pass["features"].detach(),
                "positive_mask": torch.tensor(positive, dtype=torch.bool, device=device),
            })
            if args.progress and (index % 32 == 0 or index == len(training_questions)):
                print(f"  weak training feature extraction: {index}/{len(training_questions)}", flush=True)
        coverage = len(training_cases) / max(1, len(training_questions))
        if coverage < float(phase["gates"]["minimum_training_coverage"]):
            raise ScreenError(f"weak training coverage {coverage:.4f} is below the predeclared minimum")
        head_config = QARCGConfig(n_qubits=int(phase["head"]["n_qubits"]), depth=int(phase["head"]["depth"]), residual_bound=float(phase["head"]["residual_bound"]), gate_temperature=float(phase["head"]["gate_temperature"]))
        q_head = make_torch_qarcg(head_config).to(device)
        c_head = make_torch_classical_control(head_config).to(device)
        q_training = _train_head(torch, q_head, training_cases, phase=phase, seed=int(phase["training"]["seed"]))
        c_training = _train_head(torch, c_head, training_cases, phase=phase, seed=int(phase["training"]["seed"]))
        evaluation, trace_cases = _evaluate_cases(torch, tokenizer, reader, device, q_head, c_head, cases, phase)
        q_reader = evaluation["paired_deltas"]["q_arcg_minus_frozen_reader"]
        q_classical = evaluation["paired_deltas"]["q_arcg_minus_classical"]
        gates = {
            "training_coverage": {"pass": coverage >= float(phase["gates"]["minimum_training_coverage"]), "coverage": round(coverage, 6), "minimum": float(phase["gates"]["minimum_training_coverage"])},
            "utility": {"pass": q_reader["mean"] > float(phase["gates"]["minimum_q_arcg_minus_reader_silver_overlap"]) and q_reader["bootstrap_ci95_lower"] >= 0.0, "delta": q_reader["mean"], "ci95_lower": q_reader["bootstrap_ci95_lower"]},
            "quantum_attribution": {"pass": q_classical["mean"] > float(phase["gates"]["minimum_q_arcg_minus_classical_silver_overlap"]) and q_classical["bootstrap_ci95_lower"] > 0.0, "delta": q_classical["mean"], "ci95_lower": q_classical["bootstrap_ci95_lower"]},
        }
        summary: dict[str, Any] = {
            "schema_version": "rag.qarcg_reader_screen_100.summary.v1",
            "artifact_type": "qarcg_reader_integrated_screen_summary",
            "diagnostic_only": True,
            "candidate_id": phase["candidate_id"],
            "evaluation": evaluation,
            "gates": gates,
            "decision": "pass_utility_gate" if gates["utility"]["pass"] else "stop_candidate_after_screen",
            "claim_ceiling": "L0_diagnostic",
            "limitations": ["Weak answer-string containment labels are training-only and not official passage gold.", "Silver labels are post-hoc only; no Generator/evaluator was called.", "A quantum attribution claim requires the matched classical control gate to pass."],
        }
        if _contains_forbidden(summary):
            raise ScreenError("compact summary contains a forbidden raw-content field")
        output_root = Path(args.output_root or phase["outputs"]["root"])
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / run_id
        suffix = 1
        while run_dir.exists():
            run_dir = output_root / f"{run_id}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        compact_summary = {
            **summary,
            "provenance": {"config": str(config_path.relative_to(ROOT)).replace("\\", "/"), "config_sha256": _sha256(config_path), "git_revision": _git_revision(), "evaluation_input_sha256": _sha256(input_path), "reader": reader_identity},
            "training": {"split": phase["dataset"]["training_split"], "requested_questions": train_count, "usable_cases": len(training_cases), "coverage": round(coverage, 6), "skip_reasons": dict(sorted(skipped.items())), "q_arcg": q_training, "classical_control": c_training, "weak_target": phase["training"]["weak_target"], "started_at_utc": started_training},
            "online_boundary_audit": {
                "silver_used_for_training_or_selection": False,
                "gold_answers_used_for_training_or_selection": False,
                "generated_answers_used_for_training_or_selection": False,
                "evaluator_outputs_used_for_training_or_selection": False,
            },
        }
        _write_json(run_dir / "summary.json", compact_summary)
        (run_dir / "report.md").write_text(_report(compact_summary), encoding="utf-8")
        trace = {"schema_version": "sample-trace.v2", "artifact_type": "qarcg_reader_screen_selector_trace", "diagnostic_only": True, "input_sha256": _sha256(input_path), "reader": reader_identity, "cases": trace_cases}
        _write_json(run_dir / "selector_trace.json", trace)
        metadata = {
            "schema_version": "rag.qarcg_reader_screen_100.run_metadata.v1",
            "run_id": run_dir.name,
            "output_files": ["summary.json", "report.md", "run_metadata.json", "upload_manifest.json", "selector_trace.json"],
            "exchange_only": ["selector_trace.json"],
            "summary_sha256": _sha256(run_dir / "summary.json"),
            "report_sha256": _sha256(run_dir / "report.md"),
            "trace_sha256": _sha256(run_dir / "selector_trace.json"),
            "python": platform.python_version(),
            "provenance": compact_summary["provenance"],
            "generator_called": False,
            "evaluator_called": False,
        }
        _write_json(run_dir / "run_metadata.json", metadata)
        if (run_dir / "summary.json").stat().st_size > MAX_GITHUB_BYTES or (run_dir / "report.md").stat().st_size > MAX_GITHUB_BYTES:
            raise ScreenError("compact output exceeds the 1 MiB GitHub threshold")
        target_dir = f"{phase['outputs']['exchange_namespace']}/{run_dir.name}"
        manifest = {
            "schema_version": "rag.qarcg_reader_screen_100.upload_manifest.v1",
            "artifact_type": "qarcg_reader_screen_100_upload_manifest",
            "status": "ready_for_github_and_authenticated_exchange",
            "target_directory": target_dir,
            "exchange_files": [{"name": "selector_trace.json", "bytes": (run_dir / "selector_trace.json").stat().st_size, "sha256": _sha256(run_dir / "selector_trace.json")}],
            "compact_files": [{"name": name, "bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in ("summary.json", "report.md", "run_metadata.json")],
            "privacy": {"raw_passage_text_in_compact": False, "answers_in_compact": False, "silver_used_online": False, "trace_exchange_only": True},
            "provenance": compact_summary["provenance"],
        }
        _write_json(run_dir / "upload_manifest.json", manifest)
        if args.upload:
            try:
                from scripts.collab.lib.exchange_upload import upload_manifest
                receipts = upload_manifest(run_dir / "upload_manifest.json", base_url=args.exchange_url, token_env=args.token_env)
            except Exception as exc:
                raise ScreenError(f"exchange upload failed: {type(exc).__name__}: {exc}") from exc
            _write_json(run_dir / "upload_receipts.json", {"receipts": receipts})
        print(json.dumps({"output_dir": str(run_dir), "summary": str(run_dir / "summary.json"), "uploaded": bool(args.upload), "decision": summary["decision"]}, ensure_ascii=False))
        return run_dir
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", DEFAULT_EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--model-id", default=None, help="optional Hugging Face DPR Reader override")
    parser.add_argument("--revision", default=None, help="optional Hugging Face model revision override")
    parser.add_argument("--device", default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (ScreenError, OSError, ValueError, ImportError) as exc:
        print(f"Q-ARCG Reader screen failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
