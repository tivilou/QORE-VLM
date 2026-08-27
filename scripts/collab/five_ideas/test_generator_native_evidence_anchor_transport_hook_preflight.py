from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).with_name("generator_native_evidence_anchor_transport_hook_preflight.py")
SPEC = importlib.util.spec_from_file_location("generator_native_hook_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_synthetic_reader_interval_maps_and_control_is_disjoint():
    prompt, intervals = MODULE._build_synthetic_prompt()
    offsets = []
    reader_start, reader_end = intervals[0]
    reader_text = prompt[reader_start:reader_end]
    assert reader_text == "alpha beta gamma"
    offsets.extend(
        [
            (0, 0),
            (reader_start, reader_start + 5),
            (reader_start + 6, reader_start + 10),
            (reader_start + 11, reader_end),
            (reader_end + 1, reader_end + 2),
            (reader_end + 2, reader_end + 3),
            (reader_end + 3, reader_end + 4),
        ]
    )
    reader = MODULE._map_char_span(offsets, reader_start, reader_end)
    control = MODULE._choose_control_tokens(offsets, reader)
    assert reader == (1, 2, 3)
    assert control == (4, 5, 6)


def test_invalid_interval_is_rejected():
    with pytest.raises(MODULE.HookPreflightError):
        MODULE._map_char_span([(0, 0)], 4, 4)


def test_zero_length_bpe_offsets_are_recovered_and_special_tokens_skipped():
    encoded = {
        "offset_mapping": [[[0, 0], [0, 0], [6, 6], [12, 12]]],
        "special_tokens_mask": [[1, 0, 0, 0]],
    }
    assert MODULE._offset_rows(encoded, 16) == [(0, 0), (0, 6), (6, 12), (12, 16)]


def test_forbidden_report_keys_are_detected():
    assert MODULE._forbidden({"checks": {"prediction": 1}}) == ["$root.checks.prediction"]


def test_model_resolution_never_downloads(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert MODULE._resolve_model_path(str(snapshot)) == snapshot.resolve()
