import numpy as np
import pandas as pd
import pytest

from coachmodel.coach_effects import GroupEffects, coach_standard_errors
from coachmodel.range_check import drift_variance, one_season_effect


def fitted(game_shock_sd, rng, levels=("coach", "stint")):
    """Two coaches at two teams each; 200 games per stint with 60 decisions per game.
    game_shock_sd > 0 adds a shared random shift to every decision in a game."""
    rows = []
    for coach, teams in [("A", ["X", "Y"]), ("B", ["Z", "W"])]:
        for team in teams:
            for game in range(200):
                rows.append(pd.DataFrame({"coach_name": coach, "stint": f"{coach} ({team})",
                                          "game_id": f"{team}{game}", "shock": rng.normal(0, game_shock_sd)},
                                         index=range(60)))
    rows = pd.concat(rows, ignore_index=True)
    offset = rng.normal(-3, 0.8, len(rows))
    labels = (rng.random(len(rows)) < 1 / (1 + np.exp(-(offset + rows["shock"])))).astype(int)
    taus = {level: 0.3 for level in levels}
    columns = {"coach": "coach_name", "stint": "stint"}
    model = GroupEffects(rows, {level: columns[level] for level in levels}).fit(offset, labels, rows, taus)
    return {method: coach_standard_errors(model, rows, offset, labels, taus, method)
            for method in ["conditional", "joint", "clustered"]}


def test_joint_errors_count_coach_versus_team_uncertainty():
    # Each team has only one coach, so the data cannot split a coach from his teams: the
    # conditional error ignores that, the joint error is far larger.
    errors = fitted(0.0, np.random.default_rng(0))
    assert (errors["joint"] > 3 * errors["conditional"]).all()


def test_clustering_only_matters_when_games_share_shocks():
    # Coach effects only, so the data (not the prior) sets the uncertainty.
    independent = fitted(0.0, np.random.default_rng(1), levels=("coach",))
    assert np.allclose(independent["clustered"], independent["joint"], rtol=0.15)
    shared = fitted(0.6, np.random.default_rng(1), levels=("coach",))
    assert (shared["clustered"] > 1.3 * shared["joint"]).all()


def test_one_season_effect_recovers_a_known_shift():
    rng = np.random.default_rng(2)
    offset = rng.normal(-3, 1, 50_000)
    labels = (rng.random(50_000) < 1 / (1 + np.exp(-(offset + 0.4)))).astype(int)
    effect, se_independent, se_clustered = one_season_effect(offset, labels, np.arange(50_000) // 50)
    assert effect == pytest.approx(0.4, abs=3 * se_independent)
    assert se_clustered == pytest.approx(se_independent, rel=0.2)


def test_drift_estimate_recovers_known_season_to_season_change():
    rng = np.random.default_rng(3)
    n = 4000
    se_train, se_test, drift_sd = 0.05, 0.08, 0.1
    b_train = rng.normal(0, 0.3, n)
    b_test = b_train + rng.normal(0, drift_sd, n) + rng.normal(0, se_train, n) + rng.normal(0, se_test, n)
    pairs = pd.DataFrame({"b_train": b_train, "b_test": b_test,
                          "se_train_clustered": se_train, "se_test_clustered": se_test})
    assert np.sqrt(drift_variance(pairs)) == pytest.approx(drift_sd, rel=0.15)


def test_a_coach_with_one_game_in_the_test_season_is_not_checked():
    # Regression: Chauncey Billups coached one 2025-26 game. Fitting his effect to one game
    # makes that game's residuals sum to zero, so its game-clustered standard error is 0,
    # and his z came out 6.2 and inflated the drift estimate.
    from coachmodel.range_check import enough_to_check
    rng = np.random.default_rng(4)
    offset = rng.normal(-2, 1, 200)
    labels = (rng.random(200) < 1 / (1 + np.exp(-offset))).astype(int)
    one_game = np.zeros(200, dtype=int)
    _, se_independent, se_clustered = one_season_effect(offset, labels, one_game)
    assert se_clustered < 1e-6 < se_independent  # why the check must skip him
    assert labels.sum() >= 30 and not enough_to_check(labels, one_game)
    assert enough_to_check(labels, np.arange(200) // 10)  # 20 games
