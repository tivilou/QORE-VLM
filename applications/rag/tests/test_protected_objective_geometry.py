import unittest
from scripts.collab.five_ideas.audit_protected_objective_geometry import pair_geometry


class ObjectiveGeometryTests(unittest.TestCase):
    def test_pair_reachable_but_protection_box_impossible(self):
        r = pair_geometry([0., 1., -2.], 0, 1, [2], 1.)
        self.assertTrue(r["pair_box_reachable"])
        self.assertFalse(r["pair_and_zero_protection_box_reachable"])

    def test_initial_order_conflict_can_be_resolved_by_joint_score_moves(self):
        r = pair_geometry([0., 1., .5], 0, 1, [2], 1.)
        self.assertTrue(r["baseline_protection_order_conflict"])
        self.assertTrue(r["pair_and_zero_protection_box_reachable"])

    def test_no_kept_does_not_add_a_protection_constraint(self):
        r = pair_geometry([0., 1.], 0, 1, [], .6)
        self.assertTrue(r["pair_and_zero_protection_box_reachable"])
        self.assertFalse(r["baseline_protection_order_conflict"])

    def test_exact_pair_tie_is_not_certified_crossing(self):
        r = pair_geometry([0., 1.], 0, 1, [], .5)
        self.assertFalse(r["pair_box_reachable"])


if __name__ == "__main__":
    unittest.main()
