import numpy as np
import pandas as pd
import pytest

from coachmodel.player_value import design, fit_player_values, starter_value_over_bench


def test_design_marks_home_plus_and_away_minus():
    segments = pd.DataFrame({"game_id": ["g"], "minutes": [2.0], "home_points": [6], "away_points": [2],
                             "home_players": [[1, 2, 3, 4, 5]], "away_players": [[6, 7, 8, 9, 10]]})
    matrix, margin, weights = design(segments, list(range(1, 11)))
    assert matrix.toarray().tolist() == [[1] * 5 + [-1] * 5]
    assert margin[0] == pytest.approx(48 * 4 / 2)
    assert weights[0] == pytest.approx(2.0)  # long enough: weight = minutes


def test_short_segments_are_downweighted():
    segments = pd.DataFrame({"game_id": ["g"], "minutes": [0.1], "home_points": [3], "away_points": [0],
                             "home_players": [[1, 2, 3, 4, 5]], "away_players": [[6, 7, 8, 9, 10]]})
    _, _, weights = design(segments, list(range(1, 11)))
    assert weights[0] == pytest.approx(0.1 ** 2 / 0.5)


def test_recovers_who_is_good():
    # 20 players, one star (+10 per 48) and one weak player (-10); random lineups.
    rng = np.random.default_rng(0)
    true = np.zeros(20)
    true[0], true[1] = 10.0, -10.0
    rows = []
    for game in range(1500):
        for _ in range(10):
            players = rng.permutation(20)[:10]
            home, away = sorted(players[:5]), sorted(players[5:])
            minutes = 2.0
            margin = (true[home].sum() - true[away].sum()) * minutes / 48 + rng.normal(0, 2.6 * np.sqrt(minutes))
            rows.append({"game_id": game, "minutes": minutes, "home_points": max(margin, 0),
                         "away_points": max(-margin, 0), "home_players": home, "away_players": away})
    values, _, improvements = fit_player_values(pd.DataFrame(rows))
    assert values.idxmax() == 0 and values.idxmin() == 1
    assert values[0] > 5 and values[1] < -5
    assert max(improvements.values()) > 0


def test_starter_value_over_bench():
    usage = pd.DataFrame({
        "coach_name": ["C"] * 4, "team_id": [1] * 4, "person_id": [1, 2, 3, 4],
        "starts": [80, 70, 2, 0], "minutes": [2800, 2400, 1500, 900]})
    values = pd.Series({1: 6.0, 2: 2.0, 3: -1.0, 4: 1.0})
    row = starter_value_over_bench(usage, values).iloc[0]
    assert row["starter_value"] == pytest.approx((6 * 2800 + 2 * 2400) / 5200)
    assert row["bench_value"] == pytest.approx((-1 * 1500 + 1 * 900) / 2400)
