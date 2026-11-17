"""The drift-by-coach test: back-to-back pairing, and that it flags uneven drift but not even drift."""

import numpy as np
import pandas as pd

from coachmodel.drift_by_coach import consecutive_pairs, drift_test


def test_only_back_to_back_seasons_of_the_same_coach_are_paired():
    pairs = pd.DataFrame({"coach": ["A", "A", "A", "B", "B"],
                          "season": ["2022-23", "2023-24", "2025-26", "2022-23", "2024-25"],
                          "z": [1.0, 2.0, 3.0, 4.0, 5.0]})
    paired = consecutive_pairs(pairs)
    assert list(zip(paired["coach"], paired["z"], paired["z_next"])) == [("A", 1.0, 2.0)]


def simulated_pairs(rng, variance_by_coach: np.ndarray, seasons=("2022-23", "2023-24", "2024-25", "2025-26")):
    rows = [{"coach": f"coach {n}", "season": season, "z": rng.normal(0, np.sqrt(variance))}
            for n, variance in enumerate(variance_by_coach) for season in seasons]
    return pd.DataFrame(rows)


def test_even_drift_is_not_flagged_and_strongly_uneven_drift_is():
    rng = np.random.default_rng(7)
    even = simulated_pairs(rng, np.ones(200))
    # Half the coaches with 20 times the other half's variance: a gap no test should miss.
    uneven = simulated_pairs(rng, np.repeat([0.1, 2.0], 100))
    assert not drift_test(even, np.random.default_rng(0))["differs"]
    assert drift_test(uneven, np.random.default_rng(0))["differs"]
