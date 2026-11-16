"""Question-preservation regressions; no model/tokenizer library or data."""
import unittest
from applications.rag.model_input_guards import require_complete_question


class NativeTokenizerFixture:
    def __init__(self, cls=101, sep=102):
        self.cls_token_id, self.sep_token_id = cls, sep

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [{'a': 11, 'b': 12, '[SEP]': self.sep_token_id}[t] for t in tokens]


class QuestionIntegrityTests(unittest.TestCase):
    def test_complete_dpr_and_bge_questions_with_shortened_passages(self):
        for tokenizer in [NativeTokenizerFixture(), NativeTokenizerFixture(0, 2)]:
            c, s = tokenizer.cls_token_id, tokenizer.sep_token_id
            require_complete_question(tokenizer, 'a b', [[c, 11, 12, s, 90],
                                                       [c, 11, 12, s, s, 90, s, 1]])

    def test_one_bad_candidate_stops_the_whole_batch(self):
        with self.assertRaisesRegex(ValueError, 'QUESTION_TOKENS_TRUNCATED_OR_CHANGED at batch row 1'):
            require_complete_question(NativeTokenizerFixture(), 'a b',
                                      [[101, 11, 12, 102, 90], [101, 11, 102, 90]])

    def test_retained_length_does_not_hide_changed_question(self):
        with self.assertRaisesRegex(ValueError, 'QUESTION_TOKENS_TRUNCATED_OR_CHANGED'):
            require_complete_question(NativeTokenizerFixture(), 'a b', [[101, 11, 90, 102, 12]])

    def test_literal_separator_in_question_is_part_of_the_prefix(self):
        require_complete_question(NativeTokenizerFixture(), 'a [SEP] b',
                                  [[101, 11, 102, 12, 102, 90]])

    def test_empty_or_missing_question_prefix_is_not_accepted(self):
        for question, pairs in [('a', []), ('', [[101, 102, 90]])]:
            with self.assertRaises(ValueError):
                require_complete_question(NativeTokenizerFixture(), question, pairs)


if __name__ == '__main__':
    unittest.main()
