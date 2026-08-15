"""Synthetic contracts for Phase 7A answer-identity diagnostics."""

from __future__ import annotations

import numpy as np

from applications.rag.answer_evidence import (
    answer_agreement_matrix,
    build_answer_evidence_matrices,
    extract_top_answer_spans,
    normalize_answer_text,
    select_counterfactual_swap,
    summarize_answer_evidence,
)


class FakeTokenizer:
    sep_token_id = 2
    all_special_ids = [0, 1, 2]

    vocabulary = {
        10: "question",
        20: "New",
        21: "York",
        22: "City",
        30: "Boston",
        31: "Massachusetts",
    }

    def decode(self, token_ids, **_kwargs):
        return " ".join(self.vocabulary.get(int(value), "") for value in token_ids).strip()


def _hypothesis(text: str, probability: float = 1.0):
    return {"text": text, "normalized": normalize_answer_text(text), "probability": probability}


def test_normalization_collapses_articles_punctuation_and_case():
    assert normalize_answer_text("The New-York!") == "new york"


def test_top_span_extraction_masks_question_and_is_deterministic():
    tokenizer = FakeTokenizer()
    input_ids = np.array([[1, 10, 2, 20, 21, 22, 2, 0]])
    attention = np.array([[1, 1, 1, 1, 1, 1, 1, 0]])
    starts = np.zeros_like(input_ids, dtype=float)
    ends = np.zeros_like(input_ids, dtype=float)
    starts[0, 1] = 100.0  # Must be ignored because it belongs to the question.
    ends[0, 1] = 100.0
    starts[0, 3] = 8.0
    ends[0, 4] = 9.0

    first = extract_top_answer_spans(
        input_ids, starts, ends, tokenizer, attention_mask=attention, top_m=2
    )
    second = extract_top_answer_spans(
        input_ids, starts, ends, tokenizer, attention_mask=attention, top_m=2
    )
    assert first == second
    assert first[0][0]["normalized"] == "new york"
    assert all(item["normalized"] != "question" for item in first[0])
    assert np.isclose(sum(item["probability"] for item in first[0]), 1.0)


def test_empty_or_special_only_sequence_has_no_hypothesis():
    tokenizer = FakeTokenizer()
    ids = np.array([[1, 10, 2, 2, 0]])
    mask = np.array([[1, 1, 1, 1, 0]])
    logits = np.zeros_like(ids, dtype=float)
    assert extract_top_answer_spans(ids, logits, logits, tokenizer, attention_mask=mask) == [[]]


def test_aliases_produce_agreement_and_different_answers_conflict():
    hypotheses = [[_hypothesis("NYC")], [_hypothesis("New York City")], [_hypothesis("Boston")]]
    agreement = answer_agreement_matrix(
        hypotheses, alias_groups=[["NYC", "New York City"]]
    )
    assert agreement[0, 1] == 1.0
    assert agreement[0, 2] == 0.0

    matrices = build_answer_evidence_matrices(
        ["NYC facts", "New York City facts", "Boston facts"],
        hypotheses,
        passage_confidence=[0.9, 0.9, 0.9],
        alias_groups=[["NYC", "New York City"]],
    )
    assert matrices["conflict"][0, 1] == 0.0
    assert matrices["conflict"][0, 2] > 0.0


def test_duplicate_support_is_discounted_and_matrices_obey_contract():
    passages = [
        "New York City is in New York State",
        "New York City is in New York State",
        "Independent census says New York City",
    ]
    hypotheses = [[_hypothesis("New York City")]] * 3
    matrices = build_answer_evidence_matrices(passages, hypotheses)
    assert matrices["corroboration"][0, 1] == 0.0
    assert matrices["corroboration"][0, 2] > 0.0
    for matrix in matrices.values():
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)


def test_summary_and_counterfactual_swap_are_compact_and_deterministic():
    hypotheses = [[_hypothesis("Alpha")], [_hypothesis("Alpha")], [_hypothesis("Beta")]]
    matrices = build_answer_evidence_matrices(
        ["first independent source", "second independent source", "conflicting source"],
        hypotheses,
        passage_confidence=[0.9, 0.9, 0.9],
    )
    first = select_counterfactual_swap(matrices["conflict"], [0, 1])
    second = select_counterfactual_swap(matrices["conflict"], [0, 1])
    assert first == second
    assert first is not None and first[1] == 2
    summary = summarize_answer_evidence(matrices, hypotheses, [0, 1])
    assert summary["candidate_count"] == 3
    assert "passages" not in summary


def test_invalid_matrix_inputs_are_rejected():
    hypotheses = [[_hypothesis("A")]]
    try:
        summarize_answer_evidence(
            {
                "agreement": np.ones((1, 1)),
                "conflict": np.zeros((1, 1)),
                "duplication": np.zeros((1, 1)),
                "corroboration": np.zeros((1, 1)),
            },
            hypotheses,
            [0],
        )
    except ValueError as exc:
        assert "symmetric with zero diagonal" in str(exc)
    else:
        raise AssertionError("non-zero diagnostic diagonal was accepted")
