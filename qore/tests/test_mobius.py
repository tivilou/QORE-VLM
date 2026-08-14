import numpy as np

from applications.rag.selector import select_passages
from qore.enhancers import create_pipeline, list_enhancers


class UtilityScorer:
    def score_passages(self, question: str, passages: list[str]) -> np.ndarray:
        del question
        return np.asarray([len(text.split()) / 10.0 for text in passages])


def test_mobius_plugin_is_discovered():
    assert "mobius" in list_enhancers()


def test_pairwise_mobius_uses_second_order_surplus_and_is_additive():
    b = np.array([[0.0, 0.4], [0.4, 0.0]])
    passages = ["one two", "three four five"]
    scorer = UtilityScorer()
    pipeline = create_pipeline(
        ["baseline", "mobius"],
        {"baseline": {"gamma": 0.5}, "mobius": {"beta": 0.2}},
        strict_composition=True,
    )

    w, trace = pipeline.enhance_with_diagnostics(
        np.array([0.8, 0.3]),
        b,
        {"question": "q", "passages": passages, "answer_scorer": scorer},
    )

    single = scorer.score_passages("q", passages)
    pair = scorer.score_passages("q", ["one two three four five"])[0]
    surplus = pair - single[0] - single[1]
    expected = 0.5 * b[0, 1] - 0.2 * surplus
    np.testing.assert_allclose(w[0, 1], expected)
    np.testing.assert_allclose(w, w.T)
    assert trace[-1]["mode"] == "add"
    assert trace[-1]["plugin_diagnostics"]["n_pairs"] == 1


def test_disabled_or_zero_beta_is_baseline_equivalent_without_context():
    a = np.array([0.9, 0.4, 0.1])
    b = np.array([[0.0, 0.2, 0.6], [0.2, 0.0, 0.3], [0.6, 0.3, 0.0]])
    baseline = create_pipeline(["baseline"], {"baseline": {"gamma": 0.5}})
    disabled = create_pipeline(
        ["baseline", "mobius"],
        {"baseline": {"gamma": 0.5}, "mobius": {"mode": "disabled", "beta": 0.8}},
        strict_composition=True,
    )
    zero_beta = create_pipeline(
        ["baseline", "mobius"],
        {"baseline": {"gamma": 0.5}, "mobius": {"beta": 0.0}},
        strict_composition=True,
    )
    expected = baseline.enhance(a, b, {})
    np.testing.assert_allclose(disabled.enhance(a, b, {}), expected)
    np.testing.assert_allclose(zero_beta.enhance(a, b, {}), expected)


def test_selector_accepts_mobius_pipeline_on_small_pool():
    rng = np.random.default_rng(22)
    query = rng.normal(size=8)
    embeddings = rng.normal(size=(5, 8))
    selected = select_passages(
        query,
        embeddings,
        K=2,
        method="qore",
        seed=3,
        direct_solve_max_n=20,
        relevance_scores=np.array([0.2, 0.8, 0.4, 0.6, 0.3]),
        enhancers=["baseline", "mobius"],
        enhancer_configs={"baseline": {"gamma": 0.5}, "mobius": {"beta": 0.0}},
    )
    assert len(selected) == 2
    assert len(set(selected.tolist())) == 2


def test_selector_passes_answer_context_to_nonzero_mobius():
    rng = np.random.default_rng(23)
    query = rng.normal(size=6)
    embeddings = rng.normal(size=(4, 6))
    scorer = UtilityScorer()
    diagnostics = {}
    selected = select_passages(
        query,
        embeddings,
        K=2,
        method="qore",
        seed=4,
        direct_solve_max_n=20,
        relevance_scores=np.array([0.2, 0.8, 0.4, 0.6]),
        enhancers=["baseline", "mobius"],
        enhancer_configs={"baseline": {"gamma": 0.5}, "mobius": {"beta": 0.1}},
        answer_scorer=scorer,
        passage_texts=["one", "two three", "four five six", "seven eight"],
        question="q",
        diagnostics=diagnostics,
    )
    assert len(selected) == 2
    mobius_trace = diagnostics["enhancer_trace"][1]
    assert mobius_trace["plugin_diagnostics"]["n_pairs"] == 6
