"""Phase 3, first decision: when should a starter in foul trouble sit?

A finite-horizon Markov decision process, solved by backward induction.

One starter, minute by minute. State (t, f, k, m):
    t  minutes left in regulation (48 -> 0)
    f  his fouls (0-5; a 6th foul disqualifies him for the rest of the game)
    k  minutes he has played (at most his cap: his typical minutes in close games)
    m  score margin for his team
Each minute the coach chooses:
    play: margin drifts by value_per_48 / 48 in his team's favor (his edge over the
          backup), he fouls with probability foul_rate, and k goes up by one
    sit:  the backup plays (no drift), he cannot foul
Plus random margin noise each minute (normal, sd sigma). At the buzzer the team wins
if m > 0, and a tie counts as half a win (overtime is a coin flip).

Backward induction: the value at t = 0 is known (win or not), and the value of every
state one minute earlier is the better of play and sit, averaged over what can happen
in that minute. Working back to tip-off gives the win probability under the best
policy, and the policy itself: play or sit in every state.

Simplifications, stated so they can be relaxed later: one player in isolation, a
constant edge over the backup, a constant foul rate, no overtime, and a decision every
minute rather than at each stoppage.

Usage:
    python -m coachmodel.foul_trouble
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

GAME_MINUTES = 48
MAX_FOULS = 6
MARGINS = np.arange(-30, 31)  # margins beyond +-30 are treated as +-30
MODELS_SQL_DIR = REPO_ROOT / "sql" / "models"
SIMULATED_GAMES = 200_000


@dataclass(frozen=True)
class Player:
    foul_rate: float  # fouls per minute on the court, with 0-1 fouls
    value_per_48: float  # points per 48 minutes better than his backup
    minutes_cap: int  # minutes he can play (typical close-game minutes)
    # Foul rate at each foul count relative to foul_rate: players foul less once in
    # trouble (measured in coachmodel.foul_trouble_effects). All 1.0 = constant rate.
    foul_multipliers: tuple = (1.0,) * 6

    def foul_chance(self, fouls: int) -> float:
        return self.foul_rate * self.foul_multipliers[fouls]


def margin_transition(drift: float, sigma: float) -> np.ndarray:
    """Matrix P[i, j]: chance the margin moves from MARGINS[i] to MARGINS[j] in one
    minute. The normal change is rounded to whole points; the ends absorb the tails."""
    edges = np.concatenate([[-np.inf], (MARGINS[:-1] + MARGINS[1:]) / 2, [np.inf]])
    after = edges[None, :] - MARGINS[:, None] - drift
    return np.diff(norm.cdf(after / sigma), axis=1)


def final_value() -> np.ndarray:
    """Win probability at the buzzer for each margin: win, tie (coin-flip overtime), loss."""
    return np.where(MARGINS > 0, 1.0, np.where(MARGINS == 0, 0.5, 0.0))


@dataclass
class Solution:
    # value[t, f, k, m]: win probability from state (t minutes left, f fouls,
    # k minutes played, margin m) when playing optimally from then on.
    value: np.ndarray
    # play_gain[t, f, k, m]: win probability of playing him this minute minus sitting
    # him (NaN once he has reached his cap).
    play_gain: np.ndarray

    def tip_off(self) -> float:
        return float(self.value[GAME_MINUTES, 0, 0, MARGINS == 0][0])


def solve(player: Player, sigma: float, policy=None) -> Solution:
    """Backward induction. With policy=None, choose the best action in every state. A
    policy(t, f) returning the chance he plays evaluates that policy instead (playing
    is still impossible past his cap)."""
    cap = player.minutes_cap
    sit_move = margin_transition(0.0, sigma)
    play_move = margin_transition(player.value_per_48 / GAME_MINUTES, sigma)

    value = np.full((GAME_MINUTES + 1, MAX_FOULS, cap + 1, len(MARGINS)), np.nan)
    play_gain = np.full_like(value, np.nan)
    value[0] = final_value()
    # Once he fouls out the backup plays every remaining minute.
    fouled_out = np.zeros((GAME_MINUTES + 1, len(MARGINS)))
    fouled_out[0] = final_value()

    for t in range(1, GAME_MINUTES + 1):
        later = value[t - 1]
        fouled_out[t] = sit_move @ fouled_out[t - 1]
        sit = later @ sit_move.T  # (f, k, m): average over the margin's move
        play = np.full_like(sit, np.nan)
        for f in range(MAX_FOULS):
            # Fouling again moves him to f + 1 fouls, or out of the game at the 6th.
            if f + 1 < MAX_FOULS:
                after_foul = later[f + 1, 1:]
            else:
                after_foul = np.broadcast_to(fouled_out[t - 1], later[f, 1:].shape)
            chance = player.foul_chance(f)
            next_state = (1 - chance) * later[f, 1:] + chance * after_foul
            play[f, :cap] = next_state @ play_move.T
        play_gain[t] = play - sit
        if policy is None:
            value[t] = np.where(np.isnan(play), sit, np.fmax(play, sit))
        else:
            for f in range(MAX_FOULS):
                chance = policy(t, f)
                mixed = chance * play[f] + (1 - chance) * sit[f]
                value[t, f] = np.where(np.isnan(play[f]), sit[f], mixed)
    return Solution(value=value, play_gain=play_gain)


def calibrate_sigma(home_win_probability) -> float:
    """Pick the per-minute margin noise that makes a pure random walk agree with the
    fitted win probability model (no home edge, no possession) across close states."""
    from coachmodel.win_probability import REGULATION_SECONDS  # noqa: F401 (same clock)

    margins = np.arange(-12, 13)
    minutes_left = np.array([2, 6, 12, 24, 36])
    grid_margin, grid_minutes = np.meshgrid(margins, minutes_left)
    # Average over possession so the model's ball term cancels out.
    target = 0.5 * (home_win_probability(grid_margin.ravel(), grid_minutes.ravel() * 60, 1)
                    + home_win_probability(grid_margin.ravel(), grid_minutes.ravel() * 60, -1))
    # Remove the home edge: average the home and away views of the same margin.
    away_view = 0.5 * (home_win_probability(-grid_margin.ravel(), grid_minutes.ravel() * 60, 1)
                       + home_win_probability(-grid_margin.ravel(), grid_minutes.ravel() * 60, -1))
    target = 0.5 * (target + 1 - away_view)

    def random_walk(sigma):
        spread = sigma * np.sqrt(grid_minutes.ravel())
        return norm.cdf(grid_margin.ravel() / spread)

    candidates = np.arange(1.5, 4.01, 0.05)
    errors = [np.mean((random_walk(sigma) - target) ** 2) for sigma in candidates]
    return float(candidates[int(np.argmin(errors))])


def on_court_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Sum per-game rows (foul_trouble_coach_policy.sql) into on-court shares by game
    minute and foul count: the revealed policy of whichever games are passed in."""
    totals = rows.groupby(["game_minute", "fouls_now"], as_index=False)[["seconds", "seconds_on"]].sum()
    totals["on_court_share"] = totals["seconds_on"] / totals["seconds"]
    return totals


def coach_policy_from_data(shares: pd.DataFrame):
    """Turn the observed on-court shares (by game minute and fouls) into a policy(t, f)."""
    table = shares.pivot(index="game_minute", columns="fouls_now", values="on_court_share")
    table = table.reindex(index=range(GAME_MINUTES), columns=range(MAX_FOULS)).ffill(axis=1).fillna(0.0)

    def policy(minutes_left: int, fouls: int) -> float:
        return float(table.iloc[GAME_MINUTES - minutes_left, fouls])

    return policy


def ignore_fouls(policy):
    """The same rotation, but a starter with foul trouble plays as if he had one foul."""
    return lambda minutes_left, fouls: policy(minutes_left, min(fouls, 1))


def simulate(player: Player, sigma: float, choose_play, games: int, rng) -> dict:
    """Play many games forward under a policy. choose_play(t, f, k, m_index, rng) says
    whether he plays this minute. Returns averages: minutes, fouls, foul-outs, wins."""
    play_drift = player.value_per_48 / GAME_MINUTES
    minutes = np.zeros(games)
    fouls = np.zeros(games, dtype=int)
    margin = np.zeros(games)
    for t in range(GAME_MINUTES, 0, -1):
        index = np.clip(np.round(margin).astype(int) - MARGINS[0], 0, len(MARGINS) - 1)
        available = (fouls < MAX_FOULS) & (minutes < player.minutes_cap)
        plays = available & choose_play(t, np.minimum(fouls, MAX_FOULS - 1), minutes.astype(int), index, rng)
        chances = np.array([player.foul_chance(f) for f in range(MAX_FOULS)])[np.minimum(fouls, MAX_FOULS - 1)]
        fouls += plays & (rng.random(games) < chances)
        minutes += plays
        margin += plays * play_drift + rng.normal(0, sigma, games)
    margin = np.round(margin)
    return {
        "minutes": minutes.mean(),
        "fouled_out": (fouls >= MAX_FOULS).mean(),
        "win_probability": np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0)).mean(),
    }


def optimal_chooser(solution: Solution):
    def choose(t, f, k, m_index, rng):
        gain = solution.play_gain[t, f, k, m_index]
        return np.nan_to_num(gain, nan=-1.0) > 0
    return choose


def policy_chooser(policy):
    def choose(t, f, k, m_index, rng):
        chance = np.array([policy(t, foul) for foul in range(MAX_FOULS)])[f]
        return rng.random(len(f)) < chance
    return choose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--value", type=float, action="append",
                        help="starter's edge over his backup, points per 48 (repeatable; default 3, 6, 10)")
    parser.add_argument("--constant-foul-rate", action="store_true",
                        help="ignore the measured drop in foul rate once a player is in foul trouble")
    args = parser.parse_args()
    values = args.value or [3.0, 6.0, 10.0]

    from coachmodel.baseline import split_by_date
    from coachmodel.foul_trouble_effects import EFFECTS_SQL, fourth_quarter_foul_multipliers
    from coachmodel.win_probability import WinProbability, load_rows

    with duckdb.connect(str(args.database), read_only=True) as connection:
        players = connection.sql((MODELS_SQL_DIR / "foul_trouble_players.sql").read_text()).df()
        shares = on_court_shares(connection.sql((MODELS_SQL_DIR / "foul_trouble_coach_policy.sql").read_text()).df())
        win_rows = load_rows(connection)
        effects = connection.sql(EFFECTS_SQL.read_text()).df()
    multipliers = (1.0,) * MAX_FOULS if args.constant_foul_rate else tuple(fourth_quarter_foul_multipliers(effects))
    win_rows = win_rows[win_rows["season_type"] == "Regular Season"]
    win_model = WinProbability().fit(split_by_date(win_rows, "2100-01-01")[0])
    sigma = calibrate_sigma(win_model.home_win_probability)

    regulars = players[players["close_starts"] >= 40]
    # Per-minute foul rate, pooled toward the league rate (adding 30 minutes' worth of
    # league-average fouling) so players with fewer minutes are not over-read.
    league_rate = regulars["fouls"].sum() / regulars["minutes"].sum()
    typical = Player(
        foul_rate=float(((regulars["fouls"] + 30 * league_rate) / (regulars["minutes"] + 30)).median()),
        value_per_48=0.0,
        minutes_cap=int(round(regulars["typical_close_minutes"].median())),
    )
    coach = coach_policy_from_data(shares)
    rng = np.random.default_rng(124)
    # Give every policy the same minutes budget: what the coaches' own rotation gives a
    # starter who never gets into foul trouble. Then differences come from when he plays
    # and how foul trouble is handled, not from one policy simply playing him more.
    uncapped = Player(typical.foul_rate, 0.0, GAME_MINUTES, multipliers)
    rotation_minutes = simulate(uncapped, 1.0, policy_chooser(ignore_fouls(coach)), SIMULATED_GAMES, rng)["minutes"]
    typical = Player(typical.foul_rate, 0.0, int(round(rotation_minutes)), multipliers)

    print(f"margin noise calibrated to the win probability model: sigma = {sigma:.2f} points per minute")
    print(f"typical starter: {typical.foul_rate:.3f} fouls per minute ({typical.foul_rate * 36:.1f} per 36); "
          f"median close-game minutes {regulars['typical_close_minutes'].median():.1f}; "
          f"cap {typical.minutes_cap} minutes = coaches' rotation without foul trouble")
    print("foul rate by foul count, relative to 0-1 fouls: "
          + ", ".join(f"{f}: {m:.2f}" for f, m in enumerate(typical.foul_multipliers)) + "\n")

    for value_per_48 in values:
        player = Player(typical.foul_rate, value_per_48, typical.minutes_cap, typical.foul_multipliers)
        policies = {
            "optimal (MDP)": (solve(player, sigma), None),
            "coaches (from the data)": (solve(player, sigma, policy=coach), coach),
            "same rotation, ignoring fouls": (solve(player, sigma, policy=ignore_fouls(coach)), ignore_fouls(coach)),
        }
        rows = []
        for name, (solution, policy) in policies.items():
            chooser = optimal_chooser(solution) if policy is None else policy_chooser(policy)
            simulated = simulate(player, sigma, chooser, SIMULATED_GAMES, rng)
            rows.append({"policy": name, "win probability": solution.tip_off(),
                         "simulated wins": simulated["win_probability"],
                         "minutes": simulated["minutes"], "fouled out": simulated["fouled_out"]})
        table = pd.DataFrame(rows).set_index("policy")
        print(f"=== starter worth {value_per_48:g} points per 48 over his backup, "
              f"{SIMULATED_GAMES:,} simulated games per policy ===")
        print(table.to_string(formatters={"win probability": "{:.2%}".format, "simulated wins": "{:.2%}".format,
                                          "minutes": "{:.1f}".format, "fouled out": "{:.1%}".format}))
        print()


if __name__ == "__main__":
    main()
