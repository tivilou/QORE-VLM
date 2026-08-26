import pytest

from applications.rag.answer_hypothesis_bridge import (
    BRIDGE_PLUGIN_ID,
    bridge_spec_hash,
    build_bridge_query,
    consensus_hypothesis,
    merge_candidate_indices,
)


def _item(text, probability=0.8, score=2.0):
    return {"text": text, "normalized": text, "probability": probability, "score": score}


def test_consensus_requires_distinct_passages():
    hypotheses = [[_item("Ada Lovelace"), _item("Ada Lovelace", 0.9)], []]
    assert consensus_hypothesis(hypotheses, min_support=2) is None


def test_consensus_selects_strongest_supported_answer_deterministically():
    hypotheses = [
        [_item("Ada Lovelace", 0.8, 1.0), _item("London", 0.7, 3.0)],
        [_item("Ada Lovelace", 0.9, 2.0)],
        [_item("London", 0.95, 4.0)],
    ]
    result = consensus_hypothesis(hypotheses, min_support=2, min_probability=0.5)
    assert result is not None
    assert result["normalized"] == "ada lovelace"
    assert result["support"] == 2
    assert result["passage_indices"] == [0, 1]


def test_bridge_query_and_merge_are_stable():
    hypothesis = {"text": "Ada Lovelace"}
    assert build_bridge_query("Who was the programmer?", hypothesis) == (
        "Who was the programmer? Answer hypothesis: Ada Lovelace"
    )
    assert merge_candidate_indices([5, 2, 5], [7, 2, 9]) == [5, 2, 7, 9]


def test_bridge_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        build_bridge_query("", {"text": "answer"})
    with pytest.raises(ValueError):
        merge_candidate_indices([1], [-1])
    with pytest.raises(ValueError):
        consensus_hypothesis([[_item("x", probability=-0.1)]])


def test_spec_hash_is_stable_and_plugin_named():
    first = bridge_spec_hash(min_support=2, min_probability=0.5)
    second = bridge_spec_hash(min_support=2, min_probability=0.5)
    assert first == second
    assert len(first) == 64
    assert BRIDGE_PLUGIN_ID == "answer_hypothesis_evidence_bridge"
