import pandas as pd
import pytest

from coachmodel.foul_trouble import Player, simulate, solve
from coachmodel.foul_trouble_effects import compare_with_baseline, in_trouble, ratios


def rows(person_id, quarter, minutes_by_fouls, foul_rate, trouble_multiplier=1.0, points_rate=2.4):
    """Rows for one player: exact (not random) fouls and points from the given rates."""
    return [{"person_id": person_id, "quarter": quarter, "fouls_now": f, "minutes": minutes,
             "fouls": minutes * foul_rate * (trouble_multiplier if f >= 2 else 1.0),
             "team_points": minutes * points_rate, "opponent_points": minutes * points_rate}
            for f, minutes in minutes_by_fouls.items()]


def test_within_player_comparison_removes_selection():
    # Neither player changes when in foul trouble, but the foul-prone big (0.15/min)
    # spends far more time at high foul counts than the guard (0.04/min). Pooled raw
    # rates would say high foul counts come with more fouling; within-player says 1.00.
    data = pd.DataFrame(rows(1, 4, {0: 100, 1: 100, 4: 200}, foul_rate=0.15)
                        + rows(2, 4, {0: 400, 1: 100, 4: 10}, foul_rate=0.04))
    raw_high = data[data.fouls_now == 4]
    raw_low = data[data.fouls_now <= 1]
    assert raw_high.fouls.sum() / raw_high.minutes.sum() > 1.5 * raw_low.fouls.sum() / raw_low.minutes.sum()
    compared = compare_with_baseline(data)
    assert ratios(compared[compared.fouls_now == 4])["fouls"] == pytest.approx(1.0)


def test_detects_caution():
    data = pd.DataFrame(rows(1, 4, {0: 200, 1: 100, 5: 50}, foul_rate=0.08, trouble_multiplier=0.5))
    compared = compare_with_baseline(data)
    assert ratios(compared[compared.fouls_now == 5])["fouls"] == pytest.approx(0.5)


def test_in_trouble_follows_the_coaches_rule():
    assert in_trouble(1, 2) and not in_trouble(1, 1)
    assert in_trouble(3, 4) and not in_trouble(3, 3)
    assert in_trouble(4, 5) and not in_trouble(4, 4)


def test_caution_in_foul_trouble_means_fewer_foul_outs():
    import numpy as np
    constant = Player(0.076, 6.0, 32)
    cautious = Player(0.076, 6.0, 32, foul_multipliers=(1.0, 1.0, 0.99, 0.95, 0.83, 0.51))
    rng = np.random.default_rng(0)
    always_play = lambda t, f, k, m, rng: np.ones(len(f), dtype=bool)  # noqa: E731
    fouled_out = [simulate(p, 2.7, always_play, 100_000, rng)["fouled_out"] for p in (constant, cautious)]
    assert fouled_out[1] < fouled_out[0]
    assert solve(cautious, 2.7).tip_off() >= solve(constant, 2.7).tip_off()
