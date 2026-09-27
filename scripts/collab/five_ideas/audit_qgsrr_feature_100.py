#!/usr/bin/env python3
"""Audit Q-GSRR's predeclared multi-granularity inversion signal.

This is an observation-only audit. It reads the registered 100-case detail
artifact, uses only question and Top-50 passage text, and never reads evidence
labels, answers, Generator output, or evaluation fields. It deliberately does
not implement or invoke the selector.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
ROOT = next(
    (candidate for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents)
     if (candidate / "configs").is_dir() and (candidate / "applications").is_dir()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_INPUT = ROOT / ".tmp_case_study_100/silver-oracle-detail.json"
DEFAULT_EXCHANGE_INPUT_PATH = (
    "five_ideas/selector_replay_100_historical_input/"
    "silver-oracle-top5-100-20260915T120525Z-detail.json"
)
DEFAULT_EXCHANGE_INPUT_BYTES = 8046318
DEFAULT_EXCHANGE_INPUT_SHA256 = "669ce1018ec502f02bf2a4a76420c7cb9b4e2f17b250c5420b6bcf01fc1d5731"
DEFAULT_OUTPUT_ROOT = ROOT / "exchange/five_ideas/qgsrr_feature_audit_100"
EXPECTED_CASES = 100
EXPECTED_TOP50 = 50
BASE_RANK = 5
CHALLENGER_MIN = 6
CHALLENGER_MAX = 10
MARGIN = 0.0
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class AuditError(RuntimeError):
    """Raised when the frozen feature-audit contract is not met."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError("detail JSON root must be an object")
    return value


def _tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.lower()))


def _sentence_split(value: str) -> list[str]:
    # Fixed punctuation rule. No language model or adaptive segmentation.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", value.strip())
    return [part.strip() for part in parts if part.strip()] or ([value.strip()] if value.strip() else [])


def _coverage(query_tokens: set[str], value: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(value)) / len(query_tokens)


def _view_features(question: str, candidate: Mapping[str, Any]) -> dict[str, float | int]:
    query_tokens = _tokens(question)
    title = str(candidate.get("title") or "")
    text = str(candidate.get("text") or "")
    sentences = _sentence_split(text)
    one_sentence = max((_coverage(query_tokens, sentence) for sentence in sentences), default=0.0)
    two_sentence = max(
        (_coverage(query_tokens, " ".join(sentences[index:index + 2]))
         for index in range(max(1, len(sentences) - 1))),
        default=0.0,
    )
    return {
        "title_coverage": _coverage(query_tokens, title),
        "full_coverage": _coverage(query_tokens, text),
        "local_sentence_coverage": one_sentence,
        "local_two_sentence_coverage": two_sentence,
        "sentence_count": len(sentences),
        "query_token_count": len(query_tokens),
    }


def _validate_bundle(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise AuditError(f"expected exactly {EXPECTED_CASES} cases")
    for case_number, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise AuditError(f"case {case_number} is not an object")
        question = case.get("question")
        top50 = case.get("top_50")
        if not isinstance(question, str) or not question.strip():
            raise AuditError(f"case {case_number} has no question")
        if not isinstance(top50, list) or len(top50) != EXPECTED_TOP50:
            raise AuditError(f"case {case_number} must contain 50 candidates")
        ids: list[str] = []
        ranks: list[int] = []
        for candidate in top50:
            if not isinstance(candidate, Mapping):
                raise AuditError(f"case {case_number} contains a non-object candidate")
            identifier = candidate.get("id")
            title = candidate.get("title")
            text = candidate.get("text")
            rank = candidate.get("retrieved_rank")
            if not isinstance(identifier, (str, int)) or not str(identifier).strip():
                raise AuditError(f"case {case_number} contains an invalid candidate id")
            if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
                raise AuditError(f"case {case_number} contains invalid title/text")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise AuditError(f"case {case_number} contains an invalid rank")
            ids.append(str(identifier))
            ranks.append(rank)
        if len(set(ids)) != EXPECTED_TOP50 or ranks != list(range(1, EXPECTED_TOP50 + 1)):
            raise AuditError(f"case {case_number} candidate identity/rank contract failed")
    return cases


def _strictly_above(left: float, right: float) -> bool:
    return left >= right + MARGIN and left > right


def _audit_case(case_number: int, case: Mapping[str, Any]) -> dict[str, Any]:
    question = str(case["question"])
    top50 = case["top_50"]
    features = [_view_features(question, candidate) for candidate in top50]
    base = features[BASE_RANK - 1]
    inversions: list[dict[str, Any]] = []
    for rank in range(CHALLENGER_MIN, CHALLENGER_MAX + 1):
        challenger = features[rank - 1]
        # Primary trigger: the challenger is locally stronger in both fixed
        # sentence granularities while the rank-5 passage is stronger in the
        # full-passage view. Title is retained as a separate audit view and is
        # used by the strict all-view sensitivity analysis below.
        text_inversion = (
            _strictly_above(float(base["full_coverage"]), float(challenger["full_coverage"]))
            and _strictly_above(float(challenger["local_sentence_coverage"]), float(base["local_sentence_coverage"]))
            and _strictly_above(float(challenger["local_two_sentence_coverage"]), float(base["local_two_sentence_coverage"]))
        )
        all_view_inversion = text_inversion and (
            float(challenger["title_coverage"]) >= float(base["title_coverage"]) + MARGIN
        )
        if text_inversion or all_view_inversion:
            inversions.append({
                "challenger_rank": rank,
                "text_view_inversion": text_inversion,
                "all_view_inversion": all_view_inversion,
                "base": dict(base),
                "challenger": dict(challenger),
            })
    return {
        "case_number": case_number,
        "source_id": str(case.get("source_id") or case.get("question_id") or ""),
        "question_id": str(case.get("question_id") or ""),
        "candidate_id_order_sha256": hashlib.sha256(
            "\n".join(str(item["id"]) for item in top50).encode("utf-8")
        ).hexdigest(),
        "base_rank": BASE_RANK,
        "challenger_band": [CHALLENGER_MIN, CHALLENGER_MAX],
        "features": [
            {"rank": index + 1, **value}
            for index, value in enumerate(features)
        ],
        "inversions": inversions,
        "text_view_inversion": bool(any(item["text_view_inversion"] for item in inversions)),
        "all_view_inversion": bool(any(item["all_view_inversion"] for item in inversions)),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text_cases = [row for row in rows if row["text_view_inversion"]]
    all_cases = [row for row in rows if row["all_view_inversion"]]
    text_pairs = sum(sum(int(item["text_view_inversion"]) for item in row["inversions"]) for row in rows)
    all_pairs = sum(sum(int(item["all_view_inversion"]) for item in row["inversions"]) for row in rows)
    return {
        "case_count": len(rows),
        "text_view_inversion_cases": len(text_cases),
        "text_view_inversion_pairs": text_pairs,
        "all_view_inversion_cases": len(all_cases),
        "all_view_inversion_pairs": all_pairs,
        "preflight_gate_min_cases": 15,
        "text_view_gate_pass": len(text_cases) >= 15,
        "all_view_gate_pass": len(all_cases) >= 15,
        "canonical_gate": "all_view_inversion",
        "canonical_gate_pass": len(all_cases) >= 15,
        "decision": "proceed_to_selector_specification" if len(all_cases) >= 15 else "stop_before_selector_implementation",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Q-GSRR 100题 feature-only audit",
        "",
        "本报告只审计固定 Top-50 中的多粒度文本信号；没有运行 selector、Generator 或 evaluator，且没有读取 Evidence/Silver 标签。",
        "",
        f"- 输入：`{report['input']['sha256']}`",
        f"- 规模：{summary['case_count']} 题 × {EXPECTED_TOP50} 候选",
        f"- 视图：title、full passage、最高单句窗口、最高相邻双句窗口",
        f"- 边界：rank {BASE_RANK} 对 rank {CHALLENGER_MIN}--{CHALLENGER_MAX}，最多局部替换第 5 位",
        "",
        "## 结果",
        "",
        "| 触发定义 | 触发题数 | 触发候选对数 | 门槛 15 题 |",
        "|---|---:|---:|---:|",
        f"| full 反转 + 单句/双句局部反转 | {summary['text_view_inversion_cases']} | {summary['text_view_inversion_pairs']} | {'通过' if summary['text_view_gate_pass'] else '未通过'} |",
        f"| 上述条件 + title 不下降（canonical） | {summary['all_view_inversion_cases']} | {summary['all_view_inversion_pairs']} | {'通过' if summary['all_view_gate_pass'] else '未通过'} |",
        "",
        "## 决定",
        "",
        f"- 当前 canonical gate：`{summary['canonical_gate']}`。",
        f"- 结果：`{summary['decision']}`。",
        "- 该结果仍是 L0 mechanism diagnostic，不代表 Silver utility、最终答案正确率或量子优势。",
        "- 若继续，下一步只能是冻结量子/经典匹配控制和 identity/leakage contract；不能调阈值追样本。",
    ]
    return "\n".join(lines) + "\n"


def _prepare_input(input_path: Path | None, exchange_url: str | None, token_env: str) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if input_path is not None:
        return input_path.resolve(), None
    try:
        from scripts.collab.lib.exchange_upload import download_exchange_file
    except ImportError as exc:
        raise AuditError("exchange helper unavailable; pass --input explicitly") from exc
    temporary_dir = tempfile.TemporaryDirectory(prefix="qore-qgsrr-input-")
    local_path = Path(temporary_dir.name) / Path(DEFAULT_EXCHANGE_INPUT_PATH).name
    try:
        download_exchange_file(
            DEFAULT_EXCHANGE_INPUT_PATH,
            local_path,
            base_url=exchange_url,
            token_env=token_env,
            expected_size=DEFAULT_EXCHANGE_INPUT_BYTES,
            expected_sha256=DEFAULT_EXCHANGE_INPUT_SHA256,
        )
    except Exception as exc:
        temporary_dir.cleanup()
        raise AuditError("cannot download the registered detail artifact; pass --input explicitly") from exc
    return local_path, temporary_dir


def run(input_path: Path | None, output_root: Path, exchange_url: str | None = None, token_env: str = "QORE_EXCHANGE_TOKEN") -> Path:
    input_path, temporary_dir = _prepare_input(input_path, exchange_url, token_env)
    input_path = input_path.resolve()
    try:
        if not input_path.is_file() or input_path.is_symlink():
            raise AuditError(f"input is missing or is a symlink: {input_path}")
        bundle = _load_json(input_path)
        cases = _validate_bundle(bundle)
        rows = [_audit_case(number, case) for number, case in enumerate(cases, start=1)]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_root.resolve() / timestamp
        suffix = 1
        while output_dir.exists():
            output_dir = output_root.resolve() / f"{timestamp}_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True)
        report = {
        "schema_version": "rag.qgsrr_feature_audit_100.v1",
        "artifact_type": "qgsrr_observation_only_feature_audit",
        "diagnostic_only": True,
        "selector_called": False,
        "generator_called": False,
        "evaluator_called": False,
        "silver_labels_read": False,
        "input": {"path_name": input_path.name, "bytes": input_path.stat().st_size, "sha256": _sha256(input_path)},
        "protocol": {
            "case_count": EXPECTED_CASES,
            "top50_count": EXPECTED_TOP50,
            "base_rank": BASE_RANK,
            "challenger_band": [CHALLENGER_MIN, CHALLENGER_MAX],
            "margin": MARGIN,
            "tokenization": TOKEN_RE.pattern,
            "sentence_split": "fixed punctuation split on .!? followed by whitespace and uppercase/number/quote",
            "views": ["title", "full_passage", "max_single_sentence", "max_adjacent_two_sentence"],
            "label_access": "forbidden and structurally ignored",
        },
        "summary": _summary(rows),
        "provenance": {
            "script": str(SCRIPT_PATH.relative_to(ROOT)) if SCRIPT_PATH.is_relative_to(ROOT) else str(SCRIPT_PATH),
            "script_sha256": _sha256(SCRIPT_PATH),
        },
    }
        _write_json(output_dir / "summary.json", report)
        _write_json(output_dir / "feature_trace.json", {
        "schema_version": "rag.qgsrr_feature_audit_100.trace.v1",
        "diagnostic_only": True,
        "input": report["input"],
        "protocol": report["protocol"],
        "cases": rows,
        })
        (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        _write_json(output_dir / "upload_manifest.json", {
        "schema_version": "rag.qgsrr_feature_audit_100.upload_manifest.v1",
        "artifact_type": "qgsrr_observation_only_feature_audit_upload_manifest",
        "status": "ready_for_authenticated_exchange_upload",
        "target_directory": f"five_ideas/qgsrr_feature_audit_100/{output_dir.name}",
        "exchange_files": [{"name": "feature_trace.json"}],
        "compact_files": [
            {"name": name, "bytes": (output_dir / name).stat().st_size, "sha256": _sha256(output_dir / name)}
            for name in ("summary.json", "report.md")
        ],
        "privacy": {"raw_passage_text_in_compact": False, "silver_labels_used_online": False},
        })
        return output_dir
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="optional local detail JSON; default downloads the registered exchange artifact")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exchange-url", default=None)
    parser.add_argument("--token-env", default="QORE_EXCHANGE_TOKEN")
    args = parser.parse_args(argv)
    try:
        output_dir = run(args.input, args.output_root, args.exchange_url, args.token_env)
    except AuditError as exc:
        print(f"Q-GSRR feature audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(output_dir), "canonical_gate_pass": json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["summary"]["canonical_gate_pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
