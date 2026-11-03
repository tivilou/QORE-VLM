"""Synthetic routing/ranking tests; no scientific runtime, data or GPU imports."""
import unittest
from applications.rag.top8_matrix_plan_v001 import matrix_plan, rank_metrics


class BackboneScopeTests(unittest.TestCase):
    def test_dpr_only_is_never_scheduled_on_bge(self):
        plan = matrix_plan(['dpr', 'bge'], ['COMMON', 'DPR_SPECIAL'],
                           dpr_only=['DPR_SPECIAL'])
        self.assertEqual(plan['backbone_ideas'],
                         {'dpr': ['COMMON', 'DPR_SPECIAL'], 'bge': ['COMMON']})
        self.assertEqual(plan['not_scheduled'][0]['status'], 'NOT_SCHEDULED')

    def test_missing_required_model_duplicates_and_unimplemented_adapter_rejected(self):
        for args in [(['bge'], ['A'], None, ['A']),
                     (['dpr'], ['A'], ['A'], ['A']),
                     (['dpr'], ['A'], ['UNIMPLEMENTED'], None)]:
            with self.assertRaises(ValueError):
                matrix_plan(*args)

    def test_single_backbone_rank_does_not_require_or_impute_bge(self):
        seal = dict(ideas=['SPECIAL'], seeds=[7, 8],
                    backbone_ideas={'dpr': ['SPECIAL'], 'bge': []})
        metrics = {'predictions-dpr-NATIVE.json': dict(useful_lower=10, useful_upper=10)}
        for seed in seal['seeds']:
            metrics[f'dpr-SPECIAL-QUANTUM-s{seed}/predictions.json'] = dict(
                useful_lower=13, useful_upper=14)
        rows, groups = rank_metrics(seal, metrics)
        self.assertTrue(rows[0]['rankable'])
        self.assertEqual(rows[0]['comparison_group'], 'dpr_only')
        self.assertEqual(rows[0]['scope_native_delta_lower'], 3)
        self.assertIsNone(rows[0]['equal_weight_native_delta_lower'])
        self.assertEqual(len(groups['dpr_only']), 1)
        self.assertEqual(groups['dual'], [])
        del metrics['dpr-SPECIAL-QUANTUM-s8/predictions.json']
        self.assertFalse(rank_metrics(seal, metrics)[0][0]['rankable'])

    def test_missing_requested_backbone_does_not_become_dpr_only(self):
        seal = dict(ideas=['A'], seeds=[7], backbone_ideas={'dpr': ['A'], 'bge': ['A']})
        metrics = {'predictions-dpr-NATIVE.json': dict(useful_lower=10, useful_upper=10),
                   'dpr-A-QUANTUM-s7/predictions.json': dict(useful_lower=13, useful_upper=13)}
        row = rank_metrics(seal, metrics)[0][0]
        self.assertFalse(row['rankable'])
        self.assertEqual(row['comparison_group'], 'dual')

    def test_dpr_exclusive_adapter_cannot_enter_bge_loop(self):
        plan = matrix_plan(['dpr', 'bge'], ['BOTH', 'SPAN'], ['BOTH'], ['SPAN'])
        executed = [(kind, idea) for kind, ideas in plan['backbone_ideas'].items()
                    for idea in ideas]
        self.assertIn(('dpr', 'SPAN'), executed)
        self.assertNotIn(('bge', 'SPAN'), executed)


if __name__ == '__main__':
    unittest.main()
