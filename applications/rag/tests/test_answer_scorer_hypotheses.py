"""DPR scorer adapter tests without downloading a model."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from applications.rag.answer_scorer import DPRAnswerScorer


class FakeTokenizer:
    sep_token_id = 2
    all_special_ids = [0, 1, 2]

    def __call__(self, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 10, 2, 20, 21, 2]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs):
        vocabulary = {10: "question", 20: "New", 21: "York"}
        return " ".join(vocabulary.get(int(value), "") for value in token_ids).strip()


class FakeReader:
    def __call__(self, **_kwargs):
        start = torch.tensor([[0.0, 99.0, 0.0, 8.0, 1.0, 0.0]])
        end = torch.tensor([[0.0, 99.0, 0.0, 1.0, 9.0, 0.0]])
        return SimpleNamespace(
            relevance_logits=torch.tensor([1.25]),
            start_logits=start,
            end_logits=end,
        )


def _scorer() -> DPRAnswerScorer:
    scorer = DPRAnswerScorer.__new__(DPRAnswerScorer)
    scorer.device = "cpu"
    scorer.batch_size = 4
    scorer.tokenizer = FakeTokenizer()
    scorer.reader = FakeReader()
    return scorer


def test_diagnostic_scores_are_baseline_equivalent():
    scorer = _scorer()
    baseline = scorer.score_passages("question", ["New York"])
    diagnostic, hypotheses = scorer.score_passages_with_hypotheses(
        "question", ["New York"], top_m=2
    )
    assert np.array_equal(baseline, diagnostic)
    assert hypotheses[0][0]["normalized"] == "new york"


def test_empty_diagnostic_batch_is_supported():
    scores, hypotheses = _scorer().score_passages_with_hypotheses("question", [])
    assert scores.shape == (0,)
    assert hypotheses == []
