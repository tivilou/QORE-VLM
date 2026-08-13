import numpy as np

from qore.enhancers import create_pipeline, list_enhancers
from applications.rag.selector import select_passages


def test_cohesion_plugin_is_discovered():
    assert "cohesion" in list_enhancers()


def test_calibrated_weight_follows_scale_law_and_is_additive():
    a = np.array([1.0, 0.8, 0.7, 0.2])
    b = np.array(
        [[0.0, 0.4, 0.2, 0.1], [0.4, 0.0, 0.3, 0.2],
         [0.2, 0.3, 0.0, 0.5], [0.1, 0.2, 0.5, 0.0]]
    )
    pipeline = create_pipeline(
        ["baseline", "cohesion"],
        {"baseline": {"gamma": 0.5}, "cohesion": {"eta": 0.2}},
        strict_composition=True,
    )
    w, trace = pipeline.enhance_with_diagnostics(
        a, b, {"selection_K": 2}
    )
    upper = b[np.triu_indices(4, k=1)]
    cohesion = float(np.mean(upper))
    margin = 0.8 - 0.7
    expected_delta = 0.2 * margin / ((2 - 1) * (cohesion + 1e-6))
    np.testing.assert_allclose(w, (0.5 + expected_delta) * b)
    assert trace[-1]["plugin_diagnostics"]["delta_q"] == expected_delta


def test_disabled_cohesion_is_baseline_equivalent():
    a = np.array([0.9, 0.4, 0.1])
    b = np.array([[0.0, 0.2, 0.6], [0.2, 0.0, 0.3], [0.6, 0.3, 0.0]])
    baseline = create_pipeline(["baseline"], {"baseline": {"gamma": 0.5}})
    disabled = create_pipeline(
        ["baseline", "cohesion"],
        {"baseline": {"gamma": 0.5}, "cohesion": {"mode": "disabled"}},
        strict_composition=True,
    )
    np.testing.assert_allclose(
        disabled.enhance(a, b, {"selection_K": 2}),
        baseline.enhance(a, b, {}),
    )


def test_selector_accepts_cohesion_pipeline_on_small_pool():
    rng = np.random.default_rng(12)
    query = rng.normal(size=8)
    passages = rng.normal(size=(8, 8))
    selected = select_passages(
        query,
        passages,
        K=3,
        method="qore",
        seed=7,
        direct_solve_max_n=20,
        enhancers=["baseline", "cohesion"],
        enhancer_configs={
            "baseline": {"gamma": 0.5},
            "cohesion": {"mode": "calibrated", "eta": 0.1},
        },
    )
    assert len(selected) == 3
    assert len(set(selected.tolist())) == 3
