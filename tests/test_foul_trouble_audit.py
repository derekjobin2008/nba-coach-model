import numpy as np
import pandas as pd

from coachmodel.foul_trouble_audit import LeaguePolicy, cells_by_quarter, coach_policy, foul_handling_cost


def league_rows():
    """Per-game rows where starters play 70% of the time, except with foul trouble
    (2+ fouls in Q1, 3+ in Q2, 4+ in Q3), when they play 10%."""
    rows = []
    for minute in range(48):
        quarter = minute // 12 + 1
        for fouls in range(6):
            trouble = quarter < 4 and fouls > quarter
            rows.append({"game_id": "g", "team_id": 1, "game_minute": minute, "fouls_now": fouls,
                         "seconds": 600.0, "seconds_on": 600.0 * (0.1 if trouble else 0.7)})
    return pd.DataFrame(rows)


def test_coach_without_data_follows_the_league():
    league = LeaguePolicy(league_rows())
    empty = pd.DataFrame(columns=["seconds", "seconds_on"],
                         index=pd.MultiIndex.from_tuples([], names=["quarter", "fouls_now"]))
    policy = coach_policy(league, empty)
    assert np.isclose(policy(40, 3), league.by_minute.iloc[8, 3])
    assert np.isclose(policy(10, 1), league.by_minute.iloc[38, 1])


def test_pooling_limits_how_far_a_little_data_moves_a_coach():
    league = LeaguePolicy(league_rows())
    # 60 seconds of a starter in Q1 with 2 fouls, always on the court: pooled with
    # 600 seconds of league behavior (10%), his share moves only part way to 100%.
    cells = pd.DataFrame({"seconds": [60.0], "seconds_on": [60.0]},
                         index=pd.MultiIndex.from_tuples([(1, 2)], names=["quarter", "fouls_now"]))
    share = coach_policy(league, cells)(44, 2)
    assert 0.1 < share < 0.25


def test_benching_less_for_foul_trouble_costs_less():
    rows = league_rows()
    league = LeaguePolicy(rows)
    keeps_them_in = rows.assign(seconds_on=rows["seconds"] * 0.7)
    args = (0.076, (1.0, 1.0, 0.99, 0.95, 0.83, 0.51), 2.7, 32, 6.0)
    benching_cost, _ = foul_handling_cost(league, rows, *args)
    keeping_cost, _ = foul_handling_cost(league, keeps_them_in, *args)
    assert benching_cost < keeping_cost <= 1e-9
    assert len(cells_by_quarter(rows)) == 24
