import numpy as np
import pandas as pd

from coachmodel.coach_effects import GroupEffects, fit_group_effects, per_coach_count_error


def simulate(coach_effects, stint_effects, stint_coach, rows_per_stint, rng):
    """Rows for each stint, with true log-odds = situation + coach + stint."""
    stint_index = np.repeat(np.arange(len(stint_effects)), rows_per_stint)
    coach_index = np.asarray(stint_coach)[stint_index]
    offset = rng.normal(-3, 1, len(stint_index))  # situations of varying difficulty
    true_log_odds = offset + np.asarray(coach_effects)[coach_index] + np.asarray(stint_effects)[stint_index]
    labels = (rng.random(len(stint_index)) < 1 / (1 + np.exp(-true_log_odds))).astype(int)
    return offset, labels, coach_index, stint_index


def test_recovers_known_coach_and_stint_effects():
    # Three coaches; coach 0 has two stints that differ (+0.3 and -0.3 around his 0.5).
    rng = np.random.default_rng(1)
    coach_effects = [0.5, -0.4, 0.0]
    stint_effects = [0.3, -0.3, 0.0, 0.0]
    stint_coach = [0, 0, 1, 2]
    offset, labels, coach_index, stint_index = simulate(coach_effects, stint_effects, stint_coach, 40_000, rng)
    coach, stint = fit_group_effects(offset, labels, [(coach_index, 3, 1.0), (stint_index, 4, 1.0)])
    # Only coach + stint is identified for each stint, so compare the totals.
    fitted_totals = coach[stint_coach] + stint
    true_totals = np.add(np.asarray(coach_effects)[stint_coach], stint_effects)
    assert np.allclose(fitted_totals, true_totals, atol=0.1)


def test_player_effects_separate_roster_from_coach():
    # Two coaches with identical true style (0). Coach 0's roster is mostly quick-hook bench
    # players (+0.8); coach 1's mostly starters (-0.8). Players move: one of each type
    # also plays for the other coach. Without player effects, coach 0 looks like a
    # quick-hook coach; with them, both coaches come out near 0.
    rng = np.random.default_rng(4)
    player_effects = np.array([0.8, 0.8, -0.8, -0.8])
    rows = 30_000
    player_index = np.repeat([0, 1, 2, 3, 0, 2], rows)  # players 0 and 2 play for both
    coach_index = np.repeat([0, 0, 1, 1, 1, 0], rows)
    offset = rng.normal(-3, 1, len(player_index))
    labels = (rng.random(len(player_index)) < 1 / (1 + np.exp(-(offset + player_effects[player_index])))).astype(int)

    coach_only, = fit_group_effects(offset, labels, [(coach_index, 2, 1.0)])
    assert coach_only[0] - coach_only[1] > 0.3  # a false "coach style" gap
    coach, player = fit_group_effects(offset, labels, [(coach_index, 2, 1.0), (player_index, 4, 1.0)])
    assert abs(coach[0] - coach[1]) < 0.15
    assert np.allclose(player - player.mean(), player_effects, atol=0.15)


def test_tiny_stint_spread_means_style_travels():
    # With tau_stint near 0, stint effects vanish and each coach gets one tendency
    # that is the average of his stints.
    rng = np.random.default_rng(2)
    offset, labels, coach_index, stint_index = simulate([0.5], [0.3, -0.3], [0, 0], 40_000, rng)
    coach, stint = fit_group_effects(offset, labels, [(coach_index, 1, 1.0), (stint_index, 2, 0.001)])
    assert np.abs(stint).max() < 0.01
    assert abs(coach[0] - 0.5) < 0.1


def test_prior_pulls_small_samples_harder():
    # Partial pooling: each estimate is pulled from what the coach's own data says
    # (a nearly flat prior) toward 0, and the coach with less data is pulled harder.
    rng = np.random.default_rng(3)
    big = simulate([0.7], [0.0], [0], 50_000, rng)
    small = simulate([0.7], [0.0], [0], 500, rng)
    offset = np.concatenate([big[0], small[0]])
    labels = np.concatenate([big[1], small[1]])
    coach_index = np.concatenate([np.zeros(50_000, int), np.ones(500, int)])

    def coach_effects(tau):
        return fit_group_effects(offset, labels, [(coach_index, 2, tau)])[0]

    own_data_only, pooled, tight = coach_effects(1000), coach_effects(0.3), coach_effects(0.01)
    kept_share = pooled / own_data_only
    assert 0 < kept_share[1] < kept_share[0] < 1
    # A very tight prior pulls the small-sample coach to the league average, while
    # 50,000 rows of evidence still move the other one off it.
    assert abs(tight[1]) < 0.01
    assert 0 < tight[0] < own_data_only[0] / 2


def test_unseen_members_get_zero():
    train = pd.DataFrame({"coach_name": ["A", "A", "B"], "stint": ["A (X)", "A (X)", "B (Y)"]})
    model = GroupEffects(train, {"coach": "coach_name", "stint": "stint"})
    model.effects = {"coach": {"A": 0.4, "B": -0.2}, "stint": {"A (X)": 0.1, "B (Y)": 0.0}}
    test = pd.DataFrame({"coach_name": ["A", "C"], "stint": ["A (Z)", "C (Z)"]})
    # A at a new team keeps his coach tendency; an unknown coach gets the league average.
    assert np.allclose(model.effects_for(test), [0.4, 0.0])


def test_per_coach_count_error():
    coaches = np.array(["a", "a", "b", "b"])
    labels = np.array([1, 1, 1, 0])
    predicted = np.array([0.5, 0.5, 0.75, 0.75])  # a: 1 vs 2 (50% off), b: 1.5 vs 1 (50% off)
    assert np.isclose(per_coach_count_error(coaches, labels, predicted), 0.5)


def test_rate_ratio_ranges_reflect_effect_and_sample_size():
    from coachmodel.coach_effects import coach_rate_ratios

    # Coach A calls more (true effect +0.5) with lots of data; coach B is average with little.
    rng = np.random.default_rng(5)
    coaches = np.array(["A"] * 40_000 + ["B"] * 2_000)
    offset = rng.normal(-3, 1, len(coaches))
    labels = (rng.random(len(coaches)) < 1 / (1 + np.exp(-(offset + np.where(coaches == "A", 0.5, 0.0))))).astype(int)
    rows = pd.DataFrame({"coach_name": coaches, "called_timeout": labels})
    model = GroupEffects(rows, {"coach": "coach_name"}).fit(offset, labels, rows, {"coach": 1.0})
    ratios = coach_rate_ratios(model, rows, offset, 1.0, "called_timeout")
    a, b = ratios.loc["A"], ratios.loc["B"]
    assert a["low"] < a["ratio"] < a["high"] and a["low"] > 1.3  # clearly above average
    assert b["low"] < 1 < b["high"]  # not distinguishable from average
    assert (b["high"] - b["low"]) > (a["high"] - a["low"])  # less data, wider range
