import numpy as np
import pandas as pd
import pytest

from coachmodel.timeout_audit import audit, shortfall


def test_only_the_last_two_timeouts_count():
    # At most 2 timeouts may be used in the last 3 minutes, so 3+ left is the same as 2.
    assert shortfall(pd.Series([0, 1, 2, 3, 7])).tolist() == [2, 1, 0, 0, 0]


def close_games(coach, timeouts_left, seasons):
    return pd.DataFrame({"coach_name": coach, "timeouts_left": timeouts_left,
                         "season": np.resize(seasons, len(timeouts_left))})


def test_cost_scales_with_how_often_a_coach_runs_short():
    games = pd.concat([
        close_games("Saver", [2, 3, 2, 4] * 15, ["2023-24", "2024-25"]),  # never short
        close_games("Spender", [1, 2] * 30, ["2023-24", "2024-25"]),  # short by one in half
    ])
    table = audit(games, value=0.02, value_se=1e-9, run_timeouts=pd.Series(dtype=float),
                  rng=np.random.default_rng(0))
    assert table.loc["Saver", "wins per season"] == 0
    # Half a timeout short per close game x 0.02 x 30 close games a season = 0.3 wins.
    assert table.loc["Spender", "wins per season"] == pytest.approx(0.3)
    assert table.loc["Spender", "low"] < 0.3 < table.loc["Spender", "high"]


def test_uncertain_timeout_value_widens_the_range():
    games = close_games("Spender", [1, 2] * 30, ["2024-25"])
    narrow = audit(games, 0.02, 1e-9, pd.Series(dtype=float), np.random.default_rng(1)).iloc[0]
    wide = audit(games, 0.02, 0.02, pd.Series(dtype=float), np.random.default_rng(1)).iloc[0]
    assert (wide["high"] - wide["low"]) > 2 * (narrow["high"] - narrow["low"])
    assert wide["low"] < 0  # a value whose range includes 0 gives a cost range that does too
