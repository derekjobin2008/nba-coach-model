import numpy as np
import pandas as pd
import pytest

from coachmodel.timeout_value import clustered_least_squares


def test_least_squares_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    design = pd.DataFrame({"intercept": 1.0, "x": x})
    outcome = 2.0 - 0.5 * x + rng.normal(size=n)
    fit = clustered_least_squares(design, outcome, np.arange(n))
    assert fit.loc["x", "estimate"] == pytest.approx(-0.5, abs=0.03)
    assert fit.loc["x", "low"] < -0.5 < fit.loc["x", "high"]


def test_clustering_widens_errors_when_games_share_shocks():
    # 500 games of 40 rows; the treatment is constant within a game and every game has
    # its own random shock, so 20,000 rows carry only about 500 rows' worth of evidence.
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(500), 40)
    treated = np.repeat(rng.random(500) < 0.5, 40).astype(float)
    outcome = np.repeat(rng.normal(0, 2, 500), 40) + rng.normal(0, 1, len(games))
    design = pd.DataFrame({"intercept": 1.0, "treated": treated})
    clustered = clustered_least_squares(design, outcome, games).loc["treated", "se"]
    as_if_independent = clustered_least_squares(design, outcome, np.arange(len(games))).loc["treated", "se"]
    assert clustered > 4 * as_if_independent
    # And the clustered range is honest: about 95% of such ranges contain the truth (0).
    assert abs(clustered_least_squares(design, outcome, games).loc["treated", "estimate"]) < 3 * clustered
