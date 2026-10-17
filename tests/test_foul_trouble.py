import numpy as np
from scipy.stats import norm

from coachmodel.foul_trouble import (
    MARGINS, Player, calibrate_sigma, margin_transition, optimal_chooser, simulate, solve,
)

SIGMA = 2.7


def test_margin_transition_is_a_probability_matrix_with_the_right_drift():
    move = margin_transition(drift=0.5, sigma=SIGMA)
    assert np.allclose(move.sum(axis=1), 1)
    middle = np.flatnonzero(MARGINS == 0)[0]
    assert abs(move[middle] @ MARGINS - 0.5) < 0.01


def test_without_foul_risk_fouls_never_change_the_decision():
    solution = solve(Player(foul_rate=0.0, value_per_48=6.0, minutes_cap=33), SIGMA)
    gain = solution.play_gain[30, :, 15, :]
    assert np.allclose(gain, gain[0], equal_nan=True)


def test_a_better_starter_wins_more():
    weak, strong = (solve(Player(0.076, value, 31), SIGMA).tip_off() for value in (3.0, 10.0))
    assert 0.5 < weak < strong


def test_optimal_beats_any_fixed_rotation():
    player = Player(0.076, 6.0, 31)
    optimal = solve(player, SIGMA).tip_off()
    for chance in (0.3, 0.65, 1.0):
        assert optimal >= solve(player, SIGMA, policy=lambda t, f, c=chance: c).tip_off() - 1e-12


def test_simulation_agrees_with_backward_induction():
    player = Player(0.076, 10.0, 31)
    solution = solve(player, SIGMA)
    simulated = simulate(player, SIGMA, optimal_chooser(solution), 100_000, np.random.default_rng(0))
    assert abs(simulated["win_probability"] - solution.tip_off()) < 0.01


def test_sigma_calibration_recovers_a_random_walk():
    # If the win probability model were exactly a random walk with sd 2.4 per minute
    # and no home edge, calibration should find 2.4.
    def random_walk(margin, seconds_left, ball):
        return norm.cdf(np.asarray(margin) / (2.4 * np.sqrt(np.asarray(seconds_left) / 60)))
    assert abs(calibrate_sigma(random_walk) - 2.4) < 0.06
