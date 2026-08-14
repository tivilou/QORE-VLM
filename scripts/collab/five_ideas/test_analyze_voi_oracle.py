import numpy as np

from scripts.collab.five_ideas.analyze_voi_oracle import (
    gold_flags_for_pool,
    replay_selection,
)


def test_gold_flags_map_retrieved_ranks_to_pool_positions():
    item = {
        "pool_ranks": [4, 1, 3],
        "candidate_flags": [
            {"retrieved_rank": 1, "is_gold": True},
            {"retrieved_rank": 3, "is_gold": False},
            {"retrieved_rank": 4, "is_gold": True},
        ],
    }
    np.testing.assert_array_equal(gold_flags_for_pool(item), [True, True, False])


def test_replay_selection_respects_recorded_pairwise_objective():
    a = np.array([1.0, 0.2, 0.8])
    w = np.zeros((3, 3), dtype=float)
    selected = replay_selection(a, w, K=1, lam=2.0)
    np.testing.assert_array_equal(selected, [0])
