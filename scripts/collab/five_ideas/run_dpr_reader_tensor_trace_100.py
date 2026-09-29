#!/usr/bin/env python3
"""Trace the frozen DPR Reader on the registered 100 x 50 input.

The online pass exposes only question and candidate passage fields to the
Reader.  Silver/evidence and answer fields are attached after inference for
offline diagnosis.  A compact per-candidate trace and a small set of full
hidden-state/attention tensors are written to the authenticated exchange;
only aggregate summaries and provenance stay in the working tree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "research-web/apps/experiment-results/case-studies/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_INPUT_PATH = "five_ideas/selector_replay_100_historical_input/silver-oracle-top5-100-20260915T120525Z-detail.json"
DEFAULT_INPUT_BYTES = 8046318
DEFAULT_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/dpr_reader_tensor_trace_100"
DEFAULT_EXCHANGE_URL = "http://117.50.198.37:18083"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
MAX_GITHUB_BYTES = 1_048_576
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class TraceError(RuntimeError):
    """Raised when the fixed trace contract cannot be satisfied."""


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
        raise TraceError(f"cannot load JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise TraceError("detail JSON root must be an object")
    return value


def _validate(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise TraceError(f"expected exactly {EXPECTED_CASES} cases")
    for case_number, case in enumerate(cases, 1):
        if not isinstance(case, Mapping) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise TraceError(f"case {case_number} has an invalid question")
        top50 = case.get("top_50")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise TraceError(f"case {case_number} must contain exactly 50 candidates")
        ids: list[str] = []
        ranks: list[int] = []
        for candidate in top50:
            if not isinstance(candidate, Mapping):
                raise TraceError(f"case {case_number} candidate is not an object")
            identifier = candidate.get("id", candidate.get("passage_id"))
            rank = candidate.get("retrieved_rank", candidate.get("rank"))
            if not isinstance(identifier, (str, int)) or not str(identifier).strip():
                raise TraceError(f"case {case_number} has an invalid candidate id")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TraceError(f"case {case_number} has an invalid retrieved rank")
            if not isinstance(candidate.get("title"), str) or not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
                raise TraceError(f"case {case_number} has invalid title/text")
            ids.append(str(identifier))
            ranks.append(rank)
        if len(set(ids)) != EXPECTED_TOP50 or ranks != list(range(1, EXPECTED_TOP50 + 1)):
            raise TraceError(f"case {case_number} failed identity/rank contract")
    return cases


def _online_text(candidate: Mapping[str, Any]) -> str:
    title = str(candidate.get("title") or "").strip()
    text = str(candidate.get("text") or "").strip()
    return f"{title}. {text}" if title else text


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TraceError(f"non-finite {name}")
    return result


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _span_summary(start: np.ndarray, end: np.ndarray, passage_mask: np.ndarray, max_answer_tokens: int, top_k: int) -> dict[str, Any]:
    valid = np.asarray(passage_mask, dtype=bool)
    starts = np.where(valid)[0]
    candidates: list[tuple[float, int, int]] = []
    for left in starts:
        right_max = min(len(end) - 1, int(left) + max_answer_tokens - 1)
        for right in range(int(left), right_max + 1):
            if valid[right]:
                candidates.append((float(start[left] + end[right]), int(left), int(right)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best = candidates[0] if candidates else (0.0, -1, -1)
    second = candidates[1][0] if len(candidates) > 1 else best[0]
    top = [{"start": left, "end": right, "logit": round(score, 7)} for score, left, right in candidates[:top_k]]
    start_values = start[valid]
    end_values = end[valid]
    return {
        "best_span": {"start": best[1], "end": best[2], "logit": round(best[0], 7)},
        "span_margin": round(float(best[0] - second), 7),
        "start_argmax": int(starts[int(np.argmax(start_values))]) if starts.size else -1,
        "end_argmax": int(starts[int(np.argmax(end_values))]) if starts.size else -1,
        "start_entropy": round(_entropy(start_values), 7),
        "end_entropy": round(_entropy(end_values), 7),
        "top_spans": top,
        "start_topk": [{"index": int(starts[index]), "logit": round(float(start_values[index]), 7)} for index in np.argsort(-start_values)[:top_k]],
        "end_topk": [{"index": int(starts[index]), "logit": round(float(end_values[index]), 7)} for index in np.argsort(-end_values)[:top_k]],
    }


def _attention_summary(
    attention: np.ndarray,
    question_mask: np.ndarray,
    passage_mask: np.ndarray,
    title_mask: np.ndarray,
    body_mask: np.ndarray,
    local_mask: np.ndarray,
) -> dict[str, float]:
    mean_attention = np.asarray(attention, dtype=np.float32).mean(axis=0)
    q = np.asarray(question_mask, dtype=bool)
    p = np.asarray(passage_mask, dtype=bool)
    row_count = max(1, mean_attention.shape[0])
    q_rows = mean_attention[q] if np.any(q) else np.zeros((1, mean_attention.shape[1]), dtype=np.float32)
    p_rows = mean_attention[p] if np.any(p) else np.zeros((1, mean_attention.shape[1]), dtype=np.float32)
    cls_row = mean_attention[0] if mean_attention.shape[0] else np.zeros(mean_attention.shape[1], dtype=np.float32)
    title = np.asarray(title_mask, dtype=bool)
    body = np.asarray(body_mask, dtype=bool)
    local = np.asarray(local_mask, dtype=bool)
    return {
        "cls_to_question_mass": float(cls_row[q].sum()) if np.any(q) else 0.0,
        "cls_to_passage_mass": float(cls_row[p].sum()) if np.any(p) else 0.0,
        "question_to_passage_mass": float(q_rows[:, p].sum(axis=1).mean()) if np.any(p) else 0.0,
        "passage_to_question_mass": float(p_rows[:, q].sum(axis=1).mean()) if np.any(q) else 0.0,
        "question_self_mass": float(q_rows[:, q].sum(axis=1).mean()) if np.any(q) else 0.0,
        "passage_self_mass": float(p_rows[:, p].sum(axis=1).mean()) if np.any(p) else 0.0,
        "cls_to_title_mass": float(cls_row[title].sum()) if np.any(title) else 0.0,
        "cls_to_body_mass": float(cls_row[body].sum()) if np.any(body) else 0.0,
        "cls_to_best_span_mass": float(cls_row[local].sum()) if np.any(local) else 0.0,
        "attention_row_count": float(row_count),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else 0.0


def _layer_summary(
    hidden: np.ndarray,
    attention: np.ndarray | None,
    question_mask: np.ndarray,
    passage_mask: np.ndarray,
    title_mask: np.ndarray,
    body_mask: np.ndarray,
    local_mask: np.ndarray,
    final_cls: np.ndarray,
) -> dict[str, Any]:
    active = np.asarray(question_mask | passage_mask, dtype=bool)
    active_values = hidden[active] if np.any(active) else hidden
    question_values = hidden[question_mask] if np.any(question_mask) else hidden[:0]
    passage_values = hidden[passage_mask] if np.any(passage_mask) else hidden[:0]
    row: dict[str, Any] = {
        "hidden_mean": round(float(active_values.mean()), 7),
        "hidden_std": round(float(active_values.std()), 7),
        "hidden_l2_mean": round(float(np.linalg.norm(active_values, axis=1).mean()), 7),
        "cls_l2": round(float(np.linalg.norm(hidden[0])), 7),
        "cls_cosine_to_final": round(_cosine(hidden[0], final_cls), 7),
        "question_l2_mean": round(float(np.linalg.norm(question_values, axis=1).mean()), 7) if question_values.size else 0.0,
        "passage_l2_mean": round(float(np.linalg.norm(passage_values, axis=1).mean()), 7) if passage_values.size else 0.0,
    }
    if attention is not None:
        row["attention"] = _attention_summary(attention, question_mask, passage_mask, title_mask, body_mask, local_mask)
    return row


def _evidence_flags(candidate: Mapping[str, Any]) -> tuple[bool, bool]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        return False, False
    broad = evidence.get("positive_consensus")
    direct = evidence.get("direct_consensus")
    if not isinstance(broad, bool):
        broad = evidence.get("consensus_label") in {"direct", "partial"}
    if not isinstance(direct, bool):
        direct = evidence.get("consensus_label") == "direct"
    return bool(broad), bool(direct)


def _case_label(case: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    broad, direct = _evidence_flags(candidate)
    evidence = candidate.get("evidence")
    label = evidence.get("consensus_label") if isinstance(evidence, Mapping) else None
    return {"positive_consensus": broad, "direct_consensus": direct, "consensus_label": label}


def _prepare_input(path: Path | None, exchange_url: str, token_env: str) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if path is not None:
        return path.resolve(), None
    for candidate in (DEFAULT_INPUT, ROOT / "exchange" / DEFAULT_INPUT_PATH, ROOT / ".tmp_case_study_100/silver-oracle-detail.json"):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve(), None
    token = os.environ.get(token_env)
    if not token:
        raise TraceError(f"input is absent and {token_env} is not set")
    temporary = tempfile.TemporaryDirectory(prefix="qore-dpr-trace-input-")
    destination = Path(temporary.name) / Path(DEFAULT_INPUT_PATH).name
    _download(exchange_url, token, DEFAULT_INPUT_PATH, destination, DEFAULT_INPUT_BYTES, DEFAULT_INPUT_SHA256)
    return destination, temporary


def _connection(base_url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise TraceError("exchange URL must be a plain http(s) URL")
    conn: http.client.HTTPConnection
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    return conn, parsed.path.rstrip("/")


def _download(base_url: str, token: str, exchange_path: str, destination: Path, expected_size: int, expected_hash: str) -> None:
    conn, prefix = _connection(base_url, 300.0)
    try:
        conn.request("GET", prefix + "/files/" + quote(exchange_path, safe="/"), headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"})
        response = conn.getresponse()
        if response.status != 200:
            raise TraceError(f"exchange download failed (HTTP {response.status})")
        digest = hashlib.sha256()
        size = 0
        with destination.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                size += len(block)
        if size != expected_size or digest.hexdigest() != expected_hash:
            raise TraceError("downloaded detail artifact failed registered size/hash check")
    finally:
        conn.close()


def _upload(base_url: str, token: str, local_path: Path, exchange_path: str) -> dict[str, Any]:
    size = local_path.stat().st_size
    digest = _sha256(local_path)
    conn, prefix = _connection(base_url, 900.0)
    try:
        conn.putrequest("PUT", prefix + "/upload/" + quote(exchange_path, safe="/"))
        conn.putheader("Authorization", f"Bearer {token}")
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Content-Length", str(size))
        conn.putheader("Accept", "application/json")
        conn.endheaders()
        with local_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                conn.send(block)
        response = conn.getresponse()
        raw = response.read(2 * 1024 * 1024)
        if response.status not in {200, 201}:
            raise TraceError(f"exchange upload failed for {exchange_path} (HTTP {response.status})")
        receipt = json.loads(raw.decode("utf-8")) if raw else {}
        if receipt.get("path") != exchange_path or int(receipt.get("size_bytes", -1)) != size or receipt.get("sha256") != digest:
            raise TraceError(f"exchange receipt mismatch for {exchange_path}")
        return receipt
    finally:
        conn.close()


def _token_masks(encoded: Mapping[str, Any], tokenizer: Any, title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    attention = encoded["attention_mask"][0].detach().cpu().numpy().astype(bool)
    token_types = encoded.get("token_type_ids")
    if token_types is not None:
        token_types_array = token_types[0].detach().cpu().numpy()
        passage = attention & (token_types_array == 1)
        question = attention & ~passage
    else:
        ids = encoded["input_ids"][0].detach().cpu().numpy()
        sep_id = getattr(tokenizer, "sep_token_id", None)
        first_sep = int(np.where(ids == sep_id)[0][0]) if sep_id is not None and np.any(ids == sep_id) else max(1, len(ids) // 2)
        question = attention.copy()
        question[first_sep + 1:] = False
        passage = attention.copy()
        passage[: first_sep + 1] = False
    title_mask = np.zeros_like(passage, dtype=bool)
    if title.strip() and np.any(passage):
        try:
            title_encoded = tokenizer(title.strip() + ". ", add_special_tokens=False)
            title_ids = title_encoded.get("input_ids", [])
            if title_ids and isinstance(title_ids[0], list):
                title_ids = title_ids[0]
            passage_indices = np.where(passage)[0]
            passage_ids = encoded["input_ids"][0].detach().cpu().numpy()[passage_indices].tolist()
            title_ids = [int(value) for value in title_ids]
            for start in range(max(0, len(passage_ids) - len(title_ids) + 1)):
                if passage_ids[start : start + len(title_ids)] == title_ids:
                    title_mask[passage_indices[start : start + len(title_ids)]] = True
                    break
        except Exception:
            # Title localization is diagnostic metadata; failure must not alter
            # the Reader score or invalidate the question-passage pair.
            pass
    body_mask = passage & ~title_mask
    return question, passage, title_mask, body_mask, attention


def _forward(model: Any, tokenizer: Any, torch: Any, question: str, text: str, *, title: str = "", full: bool, max_length: int, max_answer_tokens: int, top_k: int) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    encoded = tokenizer(questions=[question], texts=[text], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    question_mask, passage_mask, title_mask, body_mask, attention_mask = _token_masks(encoded, tokenizer, title)
    with torch.inference_mode():
        outputs = model(**encoded, output_hidden_states=full, output_attentions=full, return_dict=True)
    relevance = float(outputs.relevance_logits.reshape(-1)[0].detach().cpu())
    start = outputs.start_logits.reshape(-1, outputs.start_logits.shape[-1])[0].detach().float().cpu().numpy()
    end = outputs.end_logits.reshape(-1, outputs.end_logits.shape[-1])[0].detach().float().cpu().numpy()
    span = _span_summary(start, end, passage_mask, max_answer_tokens, top_k)
    local_mask = np.zeros_like(passage_mask, dtype=bool)
    if span["best_span"]["start"] >= 0:
        local_mask[span["best_span"]["start"] : span["best_span"]["end"] + 1] = True
    active_tokens = int(attention_mask.sum())
    passage_tokens = int(passage_mask.sum())
    output: dict[str, Any] = {
        "relevance_logit": round(relevance, 7),
        "relevance_score": round(_sigmoid(relevance), 7),
        "token_count": active_tokens,
        "question_token_count": int(question_mask.sum()),
        "passage_token_count": passage_tokens,
        "title_token_count": int(title_mask.sum()),
        "body_token_count": int(body_mask.sum()),
        "truncated": bool(active_tokens >= max_length),
        "span": span,
    }
    if not full:
        return output, None
    hidden_values = getattr(outputs, "hidden_states", None)
    attention_values = getattr(outputs, "attentions", None)
    if not hidden_values:
        raise TraceError("DPR Reader did not return hidden_states")
    hidden = np.stack([value[0].detach().float().cpu().numpy() for value in hidden_values])
    attentions: list[np.ndarray] = []
    if attention_values:
        attentions = [value[0].detach().float().cpu().numpy() for value in attention_values]
    final_cls = hidden[-1, 0]
    layers = []
    for layer_index, values in enumerate(hidden):
        # hidden_states[0] is the embedding output; attentions[0] belongs to
        # the first Transformer block and therefore aligns with hidden_states[1].
        attention = attentions[layer_index - 1] if layer_index > 0 and layer_index - 1 < len(attentions) else None
        layers.append({"layer": layer_index, **_layer_summary(values, attention, question_mask, passage_mask, title_mask, body_mask, local_mask, final_cls)})
    output["layers"] = layers
    output["special_token_ids"] = [int(value) for value in encoded["input_ids"][0].detach().cpu().tolist()]
    arrays = {
        "input_ids": encoded["input_ids"][0].detach().cpu().numpy(),
        "attention_mask": encoded["attention_mask"][0].detach().cpu().numpy(),
        "question_mask": question_mask.astype(np.uint8),
        "passage_mask": passage_mask.astype(np.uint8),
        "title_mask": title_mask.astype(np.uint8),
        "body_mask": body_mask.astype(np.uint8),
        "best_span_mask": local_mask.astype(np.uint8),
        # Full tensors are diagnostic artifacts; float16 keeps the exchange
        # upload bounded while preserving the layer/attention structure.
        "hidden_states": hidden.astype(np.float16),
    }
    if attentions:
        arrays["attentions"] = np.stack(attentions).astype(np.float16)
    return output, arrays


def _representatives(rows: list[dict[str, Any]], cases: Sequence[Mapping[str, Any]], per_case: int, max_total: int) -> list[tuple[int, int, str]]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[int(row["case_number"])].append(row)
    selected: list[tuple[int, int, str]] = []
    # Six deterministic strata are chosen after the online pass.  Evidence is
    # consulted only here, after all 50 scores for the case are complete.
    strata = ("low_direct", "high_non_evidence", "span_strong_relevance_weak", "relevance_strong_span_weak", "rank_boundary", "low_any")
    candidates_by_case: dict[int, dict[str, dict[str, Any] | None]] = {}
    for case_number in sorted(by_case):
        pool = by_case[case_number]
        candidates: dict[str, dict[str, Any] | None] = {name: None for name in strata}
        for row in sorted(pool, key=lambda item: (item["score_rank"], item["retrieved_rank"], item["candidate_id"])):
            if row["direct_consensus"] and row["score_rank"] >= 21 and candidates["low_direct"] is None:
                candidates["low_direct"] = row
            if not row["positive_consensus"] and row["score_rank"] <= 5 and candidates["high_non_evidence"] is None:
                candidates["high_non_evidence"] = row
            if row["span_logit"] - row["relevance_logit"] > 0 and candidates["span_strong_relevance_weak"] is None:
                candidates["span_strong_relevance_weak"] = row
            if row["relevance_logit"] - row["span_logit"] > 0 and candidates["relevance_strong_span_weak"] is None:
                candidates["relevance_strong_span_weak"] = row
            if 4 <= row["score_rank"] <= 7 and candidates["rank_boundary"] is None:
                candidates["rank_boundary"] = row
            if candidates["low_any"] is None and row["score_rank"] >= 21:
                candidates["low_any"] = row
        candidates_by_case[case_number] = candidates
    selected_keys: set[tuple[int, int]] = set()
    selected_per_case: Counter[int] = Counter()
    selected_per_stratum: Counter[str] = Counter()
    stratum_quota = max(1, max_total // len(strata))
    # Round-robin strata across questions so the global cap does not fill up
    # on only the first diagnostic category.
    for name in strata:
        for case_number in sorted(candidates_by_case):
            if len(selected) >= max_total:
                return selected
            row = candidates_by_case[case_number][name]
            if row is None or (int(row["case_number"]), int(row["candidate_index"])) in selected_keys:
                continue
            if selected_per_case[case_number] >= per_case or selected_per_stratum[name] >= stratum_quota:
                continue
            key = (int(row["case_number"]), int(row["candidate_index"]))
            selected_keys.add(key)
            selected.append((key[0], key[1], name))
            selected_per_case[case_number] += 1
            selected_per_stratum[name] += 1
    # If a stratum was sparse, fill remaining slots from any unused candidate
    # while keeping the per-question cap deterministic.
    if len(selected) < max_total:
        for row in sorted(rows, key=lambda item: (item["score_rank"], item["case_number"], item["candidate_index"])):
            if len(selected) >= max_total:
                break
            key = (int(row["case_number"]), int(row["candidate_index"]))
            if key in selected_keys or selected_per_case[key[0]] >= per_case:
                continue
            selected_keys.add(key)
            selected.append((key[0], key[1], "fallback"))
            selected_per_case[key[0]] += 1
    return selected


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# DPR Reader internal tensor trace · fixed 100 × 50",
        "",
        "本实验只观察固定 Top-50 上的 DPR Reader 内部表示和输出。Silver/evidence 字段在模型调用完成后才用于分层诊断，未进入模型、训练、selector 或参数选择。",
        "",
        f"- 输入：`{report['input']['sha256']}`",
        f"- 规模：{summary['case_count']} 题 × {summary['candidate_count']} 段",
        f"- 完整张量样本：{summary['full_tensor_sample_count']} 个",
        f"- 低分 Silver 样本：{summary['low_rank_direct_consensus_count']} 个；高分非 Silver 样本：{summary['high_rank_non_evidence_count']} 个",
        "",
        "## 观察目标",
        "",
        "1. 检查局部 span 证据是否强而 relevance head 低估。",
        "2. 检查 relevance 高但 start/end 分布分散或不一致的主题相关段。",
        "3. 检查 rank 5 附近和近重复候选的竞争。",
        "",
        "## 结果摘要",
        "",
        "| 诊断量 | 数值 |",
        "|---|---:|",
        f"| relevance 分数均值 | {summary['relevance_score_mean']:.6f} |",
        f"| span 熵均值（start/end） | {summary['start_entropy_mean']:.4f} / {summary['end_entropy_mean']:.4f} |",
        f"| 低分且 direct consensus | {summary['low_rank_direct_consensus_count']} |",
        f"| Top-5 且无 positive consensus | {summary['high_rank_non_evidence_count']} |",
        f"| relevance/span 明显分歧 | {summary['signal_disagreement_count']} |",
        "",
        "## 解释边界",
        "",
        "这是 L0 的内部机制诊断。完整张量和逐候选 trace 只进入 18083；GitHub 只保留本报告、摘要、运行元数据和哈希。只有完成张量模式归因后，才决定是否实现量子 evidence gate；本轮不改变现有 Answer Scorer。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    input_path, temporary = _prepare_input(args.input, args.exchange_url, args.token_env)
    try:
        bundle = _load_json(input_path)
        cases = _validate(bundle)
        try:
            import torch
            from transformers import DPRReader, DPRReaderTokenizer
        except ImportError as exc:
            raise TraceError("torch and transformers are required in the collaborator environment") from exc
        model_kwargs = {"revision": args.revision} if args.revision else {}
        tokenizer = DPRReaderTokenizer.from_pretrained(args.model_id, **model_kwargs)
        model = DPRReader.from_pretrained(args.model_id, **model_kwargs)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        model.eval()
        resolved_revision = getattr(model.config, "_commit_hash", None) or getattr(model, "_commit_hash", None) or args.revision
        if not resolved_revision:
            raise TraceError("DPR model revision could not be resolved")
        rows: list[dict[str, Any]] = []
        for case_number, case in enumerate(cases, 1):
            question = str(case["question"])
            for candidate_index, candidate in enumerate(case["top_50"]):
                output, _ = _forward(model, tokenizer, torch, question, _online_text(candidate), title=str(candidate.get("title") or ""), full=False, max_length=args.max_length, max_answer_tokens=args.max_answer_tokens, top_k=args.top_k)
                broad, direct = _evidence_flags(candidate)
                span = output["span"]
                row = {
                    "case_number": case_number,
                    "candidate_index": candidate_index,
                    "candidate_id": str(candidate.get("id", candidate.get("passage_id"))),
                    "retrieved_rank": int(candidate.get("retrieved_rank", candidate.get("rank"))),
                    "relevance_logit": output["relevance_logit"],
                    "relevance_score": output["relevance_score"],
                    "token_count": output["token_count"],
                    "question_token_count": output["question_token_count"],
                    "passage_token_count": output["passage_token_count"],
                    "truncated": output["truncated"],
                    "span_logit": span["best_span"]["logit"],
                    "span_margin": span["span_margin"],
                    "start_entropy": span["start_entropy"],
                    "end_entropy": span["end_entropy"],
                    "span_start": span["best_span"]["start"],
                    "span_end": span["best_span"]["end"],
                    "positive_consensus": broad,
                    "direct_consensus": direct,
                    "consensus_label": _case_label(case, candidate)["consensus_label"],
                }
                rows.append(row)
            case_rows = rows[-EXPECTED_TOP50:]
            order = sorted(case_rows, key=lambda item: (-item["relevance_logit"], item["retrieved_rank"], item["candidate_id"]))
            for score_rank, item in enumerate(order, 1):
                item["score_rank"] = score_rank
            if args.progress and (case_number % 10 == 0 or case_number == EXPECTED_CASES):
                print(f"  DPR Reader compact pass: {case_number}/{EXPECTED_CASES}", flush=True)
        representatives = _representatives(rows, cases, args.full_candidates_per_case, args.max_full_samples)
        full_metadata: list[dict[str, Any]] = []
        tensor_arrays: dict[str, np.ndarray] = {}
        for sample_index, (case_number, candidate_index, stratum) in enumerate(representatives, 1):
            case = cases[case_number - 1]
            candidate = case["top_50"][candidate_index]
            output, arrays = _forward(model, tokenizer, torch, str(case["question"]), _online_text(candidate), title=str(candidate.get("title") or ""), full=True, max_length=args.max_length, max_answer_tokens=args.max_answer_tokens, top_k=args.top_k)
            if arrays is None:
                raise TraceError("full trace arrays are missing")
            prefix = f"sample_{sample_index:03d}"
            for name, array in arrays.items():
                tensor_arrays[f"{prefix}_{name}"] = array
            token_ids = arrays["input_ids"].tolist()
            full_metadata.append({
                "sample_index": sample_index,
                "array_prefix": prefix,
                "case_number": case_number,
                "candidate_index": candidate_index,
                "candidate_id": str(candidate.get("id", candidate.get("passage_id"))),
                "retrieved_rank": int(candidate.get("retrieved_rank", candidate.get("rank"))),
                "stratum": stratum,
                "question": str(case["question"]),
                "title": str(candidate.get("title") or ""),
                "text": str(candidate.get("text") or ""),
                "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                "posthoc": _case_label(case, candidate),
                "reader": output,
            })
        output_dir = args.output_root.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = 1
        while output_dir.exists():
            output_dir = args.output_root.resolve() / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)
        np.savez_compressed(output_dir / "tensor_trace.npz", **tensor_arrays)
        low_direct = [row for row in rows if row["direct_consensus"] and row["score_rank"] >= 21]
        high_non = [row for row in rows if not row["positive_consensus"] and row["score_rank"] <= 5]
        disagreement = [row for row in rows if abs(float(row["span_logit"]) - float(row["relevance_logit"])) > args.disagreement_threshold]
        summary = {
            "case_count": EXPECTED_CASES,
            "candidate_count": len(rows),
            "full_tensor_sample_count": len(full_metadata),
            "low_rank_direct_consensus_count": len(low_direct),
            "high_rank_non_evidence_count": len(high_non),
            "signal_disagreement_count": len(disagreement),
            "relevance_score_mean": float(np.mean([row["relevance_score"] for row in rows])),
            "start_entropy_mean": float(np.mean([row["start_entropy"] for row in rows])),
            "end_entropy_mean": float(np.mean([row["end_entropy"] for row in rows])),
            "low_direct_by_retrieved_rank": dict(sorted(Counter(row["retrieved_rank"] for row in low_direct).items())),
            "low_direct_by_score_rank_band": dict(sorted(Counter("1-5" if row["score_rank"] <= 5 else "6-10" if row["score_rank"] <= 10 else "11-20" if row["score_rank"] <= 20 else "21-50" for row in low_direct).items())),
            "full_trace_strata": dict(Counter(item["stratum"] for item in full_metadata)),
        }
        input_info = {"path_name": input_path.name, "bytes": input_path.stat().st_size, "sha256": _sha256(input_path)}
        report = {
            "schema_version": "rag.dpr_reader_tensor_trace_100.v1",
            "artifact_type": "dpr_reader_observation_only_tensor_trace",
            "diagnostic_only": True,
            "input": input_info,
            "model": {"model_id": args.model_id, "resolved_revision": resolved_revision, "config_sha256": hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True, default=str).encode()).hexdigest(), "device": str(device)},
            "protocol": {"case_count": EXPECTED_CASES, "top50_count": EXPECTED_TOP50, "online_fields": ["question", "title", "text", "id", "retrieved_rank", "retrieval_score"], "silver_labels_used_online": False, "gold_answers_used_online": False, "selector_called": False, "generator_called": False, "evaluator_called": False, "max_length": args.max_length, "max_answer_tokens": args.max_answer_tokens, "full_candidates_per_case": args.full_candidates_per_case, "max_full_samples": args.max_full_samples, "disagreement_threshold": args.disagreement_threshold},
            "summary": summary,
            "provenance": {"script": str(SCRIPT_PATH.relative_to(ROOT)), "script_sha256": _sha256(SCRIPT_PATH), "git_revision": _git_revision()},
            "next_step": "先按 hidden-state、attention、span/relevance 分歧归因，再决定是否实现量子 evidence gate；本轮不改变 Answer Scorer。",
        }
        _write_json(output_dir / "summary.json", report)
        _write_json(output_dir / "candidate_trace.json", {"schema_version": "rag.dpr_reader_tensor_trace_100.candidate_trace.v1", "input": input_info, "protocol": report["protocol"], "cases": rows})
        _write_json(output_dir / "tensor_trace_metadata.json", {"schema_version": "rag.dpr_reader_tensor_trace_100.tensor_metadata.v1", "input": input_info, "model": report["model"], "samples": full_metadata})
        (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        compact_files = ["summary.json", "report.md", "run_metadata.json", "upload_manifest.json"]
        exchange_files = ["candidate_trace.json", "tensor_trace.npz", "tensor_trace_metadata.json"]
        metadata = {"schema_version": "rag.dpr_reader_tensor_trace_100.run_metadata.v1", "run_timestamp_utc": output_dir.name, "input": input_info, "model": report["model"], "output_files": compact_files + exchange_files, "exchange_only": exchange_files}
        _write_json(output_dir / "run_metadata.json", metadata)
        manifest = {"schema_version": "rag.dpr_reader_tensor_trace_100.upload_manifest.v1", "artifact_type": "dpr_reader_tensor_trace_100_upload_manifest", "status": "ready_for_authenticated_exchange_upload", "target_directory": f"five_ideas/dpr_reader_tensor_trace_100/{output_dir.name}", "exchange_files": [{"name": name} for name in exchange_files], "compact_files": [{"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)} for name in compact_files], "privacy": {"raw_passage_text_in_compact": False, "silver_labels_used_online": False, "full_raw_text_exchange_only": True}, "provenance": report["provenance"]}
        _write_json(output_dir / "upload_manifest.json", manifest)
        if (output_dir / "summary.json").stat().st_size > MAX_GITHUB_BYTES:
            raise TraceError("compact summary exceeds 1 MiB")
        if args.upload:
            token = os.environ.get(args.token_env)
            if not token:
                raise TraceError(f"--upload requires {args.token_env}")
            receipts = [_upload(args.exchange_url, token, output_dir / name, f"five_ideas/dpr_reader_tensor_trace_100/{output_dir.name}/{name}") for name in exchange_files]
            _write_json(output_dir / "upload_receipts.json", {"receipts": receipts})
        print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "summary.json"), "full_tensor_samples": len(full_metadata), "uploaded": bool(args.upload)}, ensure_ascii=False))
        return output_dir
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange-url", default=os.environ.get("QORE_EXCHANGE_URL", DEFAULT_EXCHANGE_URL))
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    parser.add_argument("--model-id", default="facebook/dpr-reader-single-nq-base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=350)
    parser.add_argument("--max-answer-tokens", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full-candidates-per-case", type=int, default=1)
    parser.add_argument("--max-full-samples", type=int, default=12)
    parser.add_argument("--disagreement-threshold", type=float, default=2.0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 32 or args.max_answer_tokens < 1 or args.top_k < 1 or args.full_candidates_per_case < 1 or args.max_full_samples < 1:
        parser.error("invalid positive trace parameters")
    try:
        if args.validate_only:
            input_path, temporary = _prepare_input(args.input, args.exchange_url, args.token_env)
            try:
                bundle = _load_json(input_path)
                cases = _validate(bundle)
                print(json.dumps({"input": str(input_path), "bytes": input_path.stat().st_size, "sha256": _sha256(input_path), "case_count": len(cases), "top50_count": len(cases[0]["top_50"]), "model_calls": 0, "selector_called": False}, ensure_ascii=False))
                return 0
            finally:
                if temporary is not None:
                    temporary.cleanup()
        run(args)
    except TraceError as exc:
        print(f"DPR Reader tensor trace failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
