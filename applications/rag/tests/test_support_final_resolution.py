"""Final decisions must preserve source identity, coverage and protection."""
import copy
import unittest
from applications.rag import support_supervision_axes as axes
from scripts.collab.five_ideas.support_boundary_engineering_fixture import synthetic_trace
from scripts.collab.five_ideas.finalize_support_boundary_review import apply_decisions


class FinalResolutionTests(unittest.TestCase):
    def setUp(self):
        self.trace = synthetic_trace()
        self.annotations = axes.pending_template(self.trace)
        self.row = self.annotations['cases'][0]
        self.item = self.row['items'][0]
        self.item.update(question_support='uncertain', reference_coverage='unresolved',
                         inference_type='unresolved', source_grounding='requires_external_context',
                         rationale='Synthetic truncated relation, not a real annotation')
        self.inputs = {'cases': [{'usable_case_index': 1, 'question_sha256': self.row['question_sha256'],
                                  'items': [copy.deepcopy(self.item)]}]}
        self.manual = {'cases': [{'case_index': 1, 'close_scope': True, 'items': [{
            'support': 'partial', 'coverage': 'not_supported', 'inference': 'none',
            'grounding': 'requires_external_context', 'disposition': 'excluded_insufficient_passage',
            'reason': 'Explicit fixture exclusion, not an irrelevant negative.', 'quote': 'fixture relation.'}]}]}

    def test_exclusion_closes_item_without_negative_or_source_rewrite(self):
        original = copy.deepcopy(self.annotations)
        result, decisions, excluded = apply_decisions(self.trace, self.annotations, self.inputs, self.manual)
        self.assertEqual(self.annotations, original)
        self.assertEqual(result['cases'][0]['items'][0]['question_support'], 'partial')
        self.assertEqual(result['cases'][0]['items'][0]['qualification']['status'], 'held')
        self.assertEqual(excluded, {self.row['question_sha256']})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(axes.compile_draft(self.trace, result)['draft_pair_count'], 0)

    def test_missing_decision_rejected(self):
        self.manual['cases'] = []
        with self.assertRaisesRegex(ValueError, 'coverage'):
            apply_decisions(self.trace, self.annotations, self.inputs, self.manual)

    def test_stale_opinion_or_fabricated_quote_rejected(self):
        self.inputs['cases'][0]['items'][0]['text'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'stale'):
            apply_decisions(self.trace, self.annotations, self.inputs, self.manual)
        self.inputs['cases'][0]['items'][0] = copy.deepcopy(self.item)
        self.manual['cases'][0]['items'][0]['quote'] = 'fabricated support'
        with self.assertRaisesRegex(ValueError, 'quote'):
            apply_decisions(self.trace, self.annotations, self.inputs, self.manual)

    def test_validation_and_unresolved_final_axes_rejected(self):
        self.trace['cases'][0]['future_role'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'split'):
            apply_decisions(self.trace, self.annotations, self.inputs, self.manual)
        self.trace['cases'][0]['future_role'] = 'training'
        self.manual['cases'][0]['items'][0]['coverage'] = 'unresolved'
        with self.assertRaisesRegex(ValueError, 'resolved'):
            apply_decisions(self.trace, self.annotations, self.inputs, self.manual)


if __name__ == '__main__':
    unittest.main()
