import numpy as np
import pandas as pd
import pytest

from coachmodel.coach_signal import per_coach_z
from coachmodel.forecast_check import top_ranked_hit_rate


def test_top_ranked_hit_rate_counts_only_stoppages_with_a_sub():
    test = pd.DataFrame({
        "game_id": ["g"] * 6, "stoppage_action_id": [1, 1, 1, 2, 2, 2], "team_id": [7] * 6,
        "subbed_out": [0, 1, 0, 0, 0, 0],  # stoppage 2 had no sub and is left out
    })
    hit = top_ranked_hit_rate(test, np.array([0.1, 0.6, 0.3, 0.9, 0.1, 0.1]), "subbed_out")
    miss = top_ranked_hit_rate(test, np.array([0.7, 0.2, 0.1, 0.9, 0.1, 0.1]), "subbed_out")
    assert (hit, miss) == (1.0, 0.0)


def test_per_coach_z_is_near_one_when_only_chance_differs():
    rng = np.random.default_rng(0)
    coaches = np.repeat([f"coach {i}" for i in range(200)], 2000)
    probability = rng.uniform(0.01, 0.08, len(coaches))
    observed = (rng.random(len(coaches)) < probability).astype(int)
    z = per_coach_z(pd.DataFrame({"coach_name": coaches, "subbed_out": observed}), probability, "subbed_out")
    assert z.std() == pytest.approx(1.0, abs=0.15)
