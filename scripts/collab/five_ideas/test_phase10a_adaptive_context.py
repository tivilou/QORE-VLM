from __future__ import annotations

import unittest

from applications.rag.adaptive_context import build_wide_context, prefix_digest, route_extra_context, risk_features
from scripts.collab.five_ideas.phase10a_metrics import summarize_phase10a
from scripts.collab.five_ideas.run_phase10a_adaptive_context import _prefix_parity


class AdaptiveContextTests(unittest.TestCase):
    def test_routing_is_deterministic_and_gold_free(self):
        values = risk_features([0.2, 0.4, 0.3, 0.1], [0, 1], ["a b", "a c"])
        self.assertTrue(0.0 <= values["risk_score"] <= 1.0)
        self.assertEqual(route_extra_context(values["risk_score"], 0.55), values["risk_score"] >= 0.55)

    def test_wide_context_preserves_selected_prefix(self):
        wide, extras = build_wide_context(["p0", "p1", "p2", "p3", "p4"], [3, 1], extra_count=2)
        self.assertEqual(wide[:2], ("p3", "p1"))
        self.assertEqual(extras, (0, 2))

    def test_prefix_parity_checks_local_and_global_ids(self):
        retrieved = [101, 102, 103, 104, 105, 106, 107, 108]
        selected = [3, 1, 0, 2, 4]
        digest = prefix_digest(retrieved, selected)
        self.assertTrue(_prefix_parity(retrieved, selected, [5, 6, 7], digest))
        self.assertFalse(_prefix_parity(retrieved, selected, [5, 3, 7], digest))

    def test_metrics_accepts_matched_compact_arms(self):
        def arm(name):
            samples = []
            for i in range(10):
                high_risk = i < 2
                applied = name == "always_wide" or (name == "adaptive" and high_risk)
                delta = 0.20 if applied else 0.0
                samples.append({"question_id": f"q{i}", "em": 0.4 + delta, "f1": 0.5 + delta, "generation_time_ms": 140.0 if applied else 100.0, "prefix_parity": True, "applied": applied, "risk_score": 0.7 if high_risk else 0.2})
            return {"samples": samples}
        result = {"config": {"risk_threshold": 0.55}, "arms": {"baseline_k5": arm("baseline"), "always_wide": arm("always_wide"), "adaptive": arm("adaptive")}}
        summary = summarize_phase10a(result, gate={"bootstrap_repetitions": 100, "bootstrap_seed": 7, "em_delta_min": 0.03})
        self.assertEqual(summary["decision"], "pass_adaptive_context_screen")


if __name__ == "__main__":
    unittest.main()
