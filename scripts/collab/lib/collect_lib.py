#!/usr/bin/env python3
"""Shared pieces for the exchange/ collection scripts.

Each workstream has its own thin collector (result shapes differ: the RAG
diagnosis writes gamma_*/result.json via the tuning framework, KV cache runs
eval_kv_cache.py directly and produces no status.json). What they share is here.

The point of these scripts is not saving keystrokes on `cp`. It is making
provenance checks mandatory. The 2026-07-27 round shipped four reports in which
neither of its two real problems was visible: the yaml had been hand-edited to
add --skip_generation (so no F1 at all), and two diagnosis scripts were stale
(so two reports' numbers were meaningless). Both were only found by reading
status.json's command field and diffing report titles afterwards. A collector
that records git state and refuses on a missing F1 would have surfaced both at
packaging time.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


class CollectError(RuntimeError):
    """Refuse to collect — the artifacts would be misleading."""


# ── git / 环境来源 ────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=cwd)
        return r.stdout.strip()
    except Exception:
        return ""


def git_provenance(repo: Path) -> dict:
    """git HEAD + whether the tree is dirty.

    The dirty check is not bookkeeping. Hand-edited configs and stale scripts
    are exactly how the 2026-07-27 round went wrong, and the person running it
    will not necessarily think to mention it. Recording it automatically means
    the reader can see it without asking.

    exchange/ is excluded. The question this answers is "was the code that ran
    different from the repo", and deliverables are not code — including them
    would bury the signal under the collector's own freshly staged output.
    """
    head = _run(["git", "log", "--oneline", "-1"], cwd=repo)
    status = _run(["git", "status", "--short"], cwd=repo)
    lines = []
    for ln in status.splitlines():
        if not ln.strip():
            continue
        path = ln[3:].strip().strip('"')
        if path.startswith("exchange/"):
            continue
        lines.append(ln)
    return {
        "head": head or "(unknown)",
        "dirty": bool(lines),
        "status_lines": lines,
    }


def gpu_info() -> str:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else "(nvidia-smi 不可用)"


# ── 时间戳 ────────────────────────────────────────────────────────────

def timestamp_from_start_time(start_time: str) -> str:
    """`2026-07-27T11:43:57.653582` -> `20260727T114357`.

    Read from status.json rather than taken with `date` at collection time.
    Removes the "note the start time before you run" manual step, which is
    wrong the moment someone collects the next day, and guarantees the
    directory name lines up with the timestamps inside meta/.

    status.json writes datetime.now().isoformat() with no zone, i.e. the
    runner's local time. The team convention for exchange/ is Beijing time, so
    this is used as-is — see exchange/README.md on why a single stated zone.
    """
    dt = datetime.fromisoformat(start_time)
    return dt.strftime("%Y%m%dT%H%M%S")


# ── 交付目录 ──────────────────────────────────────────────────────────

def prepare_round_dir(exchange_root: Path, experiment: str, ts: str,
                      force: bool = False) -> Path:
    """Copy _TEMPLATE into <experiment>/<ts>/, refusing to clobber."""
    dest = exchange_root / experiment / ts
    if dest.exists():
        if not force:
            raise CollectError(
                f"{dest} 已存在。\n"
                "  同一趟收集两次？想覆盖加 --force。\n"
                "  如果是新的一趟，检查 status.json 的 start_time 是不是没更新"
                "（实验可能没重跑）。"
            )
        import shutil
        shutil.rmtree(dest)

    template = exchange_root / "_TEMPLATE"
    if not template.is_dir():
        raise CollectError(f"模板不存在: {template}")

    import shutil
    shutil.copytree(template, dest)
    for p in dest.rglob(".gitkeep"):
        p.unlink()
    return dest


def append_round_row(experiment_readme: Path, ts: str, who: str) -> bool:
    """Add a row to the experiment-level round table.

    Timestamp and who are mechanical, so they are filled. Conclusion and
    verdict are left blank rather than templated — they are decided in
    conversation after reading the reports, not by whoever ran the job.
    """
    if not experiment_readme.exists():
        return False
    text = experiment_readme.read_text()
    if f"[`{ts}`]" in text:
        return False

    row = f"| [`{ts}`]({ts}/) | {who} | | |\n"

    lines = text.splitlines(keepends=True)
    out, inserted = [], False
    for i, ln in enumerate(lines):
        out.append(ln)
        if inserted or not ln.startswith("|---"):
            continue
        # 表头分隔行之后，跳过已有数据行，插在最后一行之后
        j = i + 1
        while j < len(lines) and lines[j].startswith("|"):
            out.append(lines[j])
            j += 1
        out.append(row)
        out.extend(lines[j:])
        inserted = True
        break

    if not inserted:
        return False
    experiment_readme.write_text("".join(out))
    return True


# ── 杂项 ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise CollectError(f"文件不存在: {path}")
    except json.JSONDecodeError as e:
        raise CollectError(f"{path} 不是合法 JSON: {e}\n"
                           "  通常意味着进程被中途杀掉，写了一半。")


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}分{s}秒" if m else f"{s}秒"
