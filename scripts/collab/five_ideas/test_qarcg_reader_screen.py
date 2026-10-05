#!/usr/bin/env python3
"""No-model contract tests for the Q-ARCG real-data screen runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collab.five_ideas.run_qarcg_reader_screen_100 import (  # noqa: E402
    FORBIDDEN_COMPACT_FIELDS,
    ScreenError,
    _contains_forbidden,
    _normalize_answer,
    _validate_config,
    _weak_positive_mask,
)


class QARCGReaderScreenContractTests(unittest.TestCase):
    def test_config_is_approved_and_reader_boundary_is_frozen(self) -> None:
        config = ROOT / "configs/experiments/qarcg_reader_screen_100.json"
        result = _validate_config(config)
        self.assertEqual(result["phase"]["candidate_id"], "q_arcg_reader_integrated_residual")
        self.assertTrue(result["phase"]["boundary"]["silver_used_for_training_or_selection"] is False)
        self.assertEqual(result["phase"]["dataset"]["evaluation_cases"], 100)

    def test_weak_labels_are_explicit_and_deterministic(self) -> None:
        passages = ["The answer is Ada Lovelace.", "A biography of Charles Babbage.", ""]
        mask, reason = _weak_positive_mask(passages, ["Ada Lovelace"])
        self.assertEqual(mask, [True, False, False])
        self.assertEqual(reason, "ok")
        self.assertEqual(_normalize_answer("  Ada-Lovelace! "), "ada lovelace")
        empty, empty_reason = _weak_positive_mask(passages, [])
        self.assertEqual(empty, [False, False, False])
        self.assertEqual(empty_reason, "no_nonempty_answers")

    def test_compact_payload_rejects_raw_content_fields(self) -> None:
        self.assertTrue(_contains_forbidden({"question": "private"}))
        self.assertTrue(_contains_forbidden({"nested": [{"gold_answers": ["x"]}]}))
        self.assertFalse(_contains_forbidden({"mean_overlap": 2.5, "case_count": 100}))
        self.assertIn("selected_ids", FORBIDDEN_COMPACT_FIELDS)

    def test_bad_config_fails_closed(self) -> None:
        path = ROOT / "configs/experiments/qarcg_reader_screen_100.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["phase"]["dataset"]["evaluation_cases"] = 99
        broken = ROOT / ".tmp_qarcg_screen_bad_config.json"
        broken.write_text(json.dumps(document), encoding="utf-8")
        try:
            with self.assertRaises(ScreenError):
                _validate_config(broken)
        finally:
            broken.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
