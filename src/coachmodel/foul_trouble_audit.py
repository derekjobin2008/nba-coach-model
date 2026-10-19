"""Per-coach foul-trouble audit: what does each coach's handling of foul trouble cost?

For each head coach, his revealed policy is his starters' on-court share by quarter and
foul count in his close games. Cells with little data are pulled toward the league
(partial pooling: his seconds in a cell are combined with PRIOR_SECONDS of league-average
behavior), and the league's minute-by-minute shape is kept within each quarter.

Each policy goes through the foul-trouble MDP (coachmodel.foul_trouble) for a typical
starter, worth what this coach's starters are worth over his bench (RAPM, from
coachmodel.player_value; floored at 0, since "his starters are worse than his bench"
is noise, not a reason to bench them). The coach is compared with himself:
    foul-handling cost = win probability with his actual policy
                         - win probability with his own rotation, ignoring fouls
so how many minutes he gives starters, or when, does not count for or against him;
only what he does about foul trouble does. A 95% range comes from resampling his games.

Usage:
    python -m coachmodel.foul_trouble_audit
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coachmodel.build_db import DATABASE_PATH
from coachmodel.foul_trouble import (
    GAME_MINUTES, MAX_FOULS, MODELS_SQL_DIR, Player, calibrate_sigma, ignore_fouls, on_court_shares,
    policy_chooser, simulate, solve,
)

PRIOR_SECONDS = 600  # each coach cell starts with 10 minutes of league-average behavior
MIN_CLOSE_GAMES = 100
BOOTSTRAP_SAMPLES = 200
FIXED_VALUE_PER_48 = 6.0  # the earlier assumption, kept for comparison
STARTERS = 5
SEASON_GAMES = 82


def quarter_of(game_minute):
    return np.minimum(np.asarray(game_minute) // 12 + 1, 4)


class LeaguePolicy:
    """League on-court shares by game minute and foul count, and by quarter."""

    def __init__(self, rows: pd.DataFrame):
        by_minute = on_court_shares(rows)
        self.by_minute = (by_minute.pivot(index="game_minute", columns="fouls_now", values="on_court_share")
                          .reindex(index=range(GAME_MINUTES), columns=range(MAX_FOULS)).ffill(axis=1).fillna(0.0))
        by_quarter = rows.assign(quarter=quarter_of(rows["game_minute"])).groupby(
            ["quarter", "fouls_now"])[["seconds", "seconds_on"]].sum()
        self.by_quarter = (by_quarter["seconds_on"] / by_quarter["seconds"]).unstack().reindex(
            index=range(1, 5), columns=range(MAX_FOULS)).ffill(axis=1).fillna(0.0)


def coach_policy(league: LeaguePolicy, coach_cells: pd.DataFrame):
    """policy(t, f) for one coach: the league's minute-by-minute shape, shifted in each
    (quarter, fouls) cell by how far his pooled share sits from the league's."""
    cells = coach_cells.reindex(pd.MultiIndex.from_product([range(1, 5), range(MAX_FOULS)]), fill_value=0.0)
    league_share = league.by_quarter.stack().reindex(cells.index)
    pooled = (cells["seconds_on"] + PRIOR_SECONDS * league_share) / (cells["seconds"] + PRIOR_SECONDS)
    shift = (pooled - league_share).unstack()

    def policy(minutes_left: int, fouls: int) -> float:
        minute = GAME_MINUTES - minutes_left
        share = league.by_minute.iloc[minute, fouls] + shift.loc[int(quarter_of(minute)), fouls]
        return float(np.clip(share, 0.0, 1.0))

    return policy


def cells_by_quarter(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.assign(quarter=quarter_of(rows["game_minute"])).groupby(
        ["quarter", "fouls_now"])[["seconds", "seconds_on"]].sum()


def foul_handling_cost(league, rows, foul_rate, multipliers, sigma, cap, value_per_48) -> tuple[float, float]:
    """(win probability of his policy - his rotation ignoring fouls, optimal - his policy)."""
    policy = coach_policy(league, cells_by_quarter(rows))
    player = Player(foul_rate, value_per_48, cap, multipliers)
    actual = solve(player, sigma, policy=policy).tip_off()
    ignoring = solve(player, sigma, policy=ignore_fouls(policy)).tip_off()
    return actual - ignoring, solve(player, sigma).tip_off() - actual


def audit_coaches(connection: duckdb.DuckDBPyConnection, rng, bootstrap_samples: int = BOOTSTRAP_SAMPLES) -> pd.DataFrame:
    """One row per coach with MIN_CLOSE_GAMES+ close games: his revealed foul-trouble
    shares, his starters' value over his bench, and the win probability his foul handling
    costs per starter-game, with a 95% range from resampling his games."""
    from coachmodel.foul_trouble_effects import EFFECTS_SQL, fourth_quarter_foul_multipliers
    from coachmodel.player_value import SEGMENTS_SQL, STARTER_SQL, coach_value_over_bench, fit_player_values
    from coachmodel.win_probability import WinProbability, load_rows

    rows = connection.sql((MODELS_SQL_DIR / "foul_trouble_coach_policy.sql").read_text()).df()
    coaches = connection.sql("SELECT game_id, team_id, coach_name FROM game_coaches").df()
    players = connection.sql((MODELS_SQL_DIR / "foul_trouble_players.sql").read_text()).df()
    multipliers = tuple(fourth_quarter_foul_multipliers(connection.sql(EFFECTS_SQL.read_text()).df()))
    win_rows = load_rows(connection)
    player_values, _, _ = fit_player_values(connection.sql(SEGMENTS_SQL.read_text()).df())
    coach_values = coach_value_over_bench(connection.sql(STARTER_SQL).df(), player_values)
    sigma = calibrate_sigma(WinProbability().fit(win_rows[win_rows["season_type"] == "Regular Season"])
                            .home_win_probability)
    regulars = players[players["close_starts"] >= 40]
    league_rate = regulars["fouls"].sum() / regulars["minutes"].sum()
    foul_rate = float(((regulars["fouls"] + 30 * league_rate) / (regulars["minutes"] + 30)).median())

    rows = rows.merge(coaches, on=["game_id", "team_id"])
    league = LeaguePolicy(rows)
    results = []
    for coach_name, coach_rows in rows.groupby("coach_name"):
        games = coach_rows["game_id"].unique()
        if len(games) < MIN_CLOSE_GAMES:
            continue
        # His cap: the minutes his own rotation gives a starter with no foul trouble.
        policy = coach_policy(league, cells_by_quarter(coach_rows))
        uncapped = Player(foul_rate, 0.0, GAME_MINUTES, multipliers)
        cap = int(round(simulate(uncapped, 1.0, policy_chooser(ignore_fouls(policy)), 20_000, rng)["minutes"]))
        value = max(float(coach_values.get(coach_name, 0.0)), 0.0)
        fixed_cost, _ = foul_handling_cost(league, coach_rows, foul_rate, multipliers, sigma, cap, FIXED_VALUE_PER_48)
        cost, optimal_gap = foul_handling_cost(league, coach_rows, foul_rate, multipliers, sigma, cap, value)

        by_game = {game: part for game, part in coach_rows.groupby("game_id")}
        draws = []
        for _ in range(bootstrap_samples):
            sample = pd.concat([by_game[game] for game in rng.choice(games, len(games))])
            draws.append(foul_handling_cost(league, sample, foul_rate, multipliers, sigma, cap, value)[0])
        low, high = np.percentile(draws, [2.5, 97.5])

        cells = cells_by_quarter(coach_rows)
        share = lambda q, f: cells.loc[(q, f), "seconds_on"] / cells.loc[(q, f), "seconds"] if (q, f) in cells.index else np.nan  # noqa: E731
        results.append({
            "coach": coach_name, "close games": len(games), "starter cap": cap, "value over bench": value,
            "Q1 2 fouls": share(1, 2), "Q3 4 fouls": share(3, 4), "Q4 5 fouls": share(4, 5),
            "cost per starter-game": cost, "low": low, "high": high,
            "season wins": cost * STARTERS * SEASON_GAMES, "optimal gain": optimal_gap,
            "cost if worth 6": fixed_cost,
        })
    return pd.DataFrame(results).sort_values("cost per starter-game").set_index("coach")


def league_foul_trouble_shares(connection: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """League-wide starter on-court shares in the three foul-trouble cells the page shows."""
    rows = connection.sql((MODELS_SQL_DIR / "foul_trouble_coach_policy.sql").read_text()).df()
    cells = cells_by_quarter(rows)
    return {name: float(cells.loc[cell, "seconds_on"] / cells.loc[cell, "seconds"])
            for name, cell in {"Q1 2 fouls": (1, 2), "Q3 4 fouls": (3, 4), "Q4 5 fouls": (4, 5)}.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    with duckdb.connect(str(args.database), read_only=True) as connection:
        table = audit_coaches(connection, np.random.default_rng(124))

    print("foul-handling cost for a starter worth what this coach's starters are worth over his bench "
          "(RAPM, points per 48)\n(win probability points per starter-game; negative = his foul handling costs wins)")
    print(f"season wins = cost x {STARTERS} starters x {SEASON_GAMES} games (assumes similar starters)\n")
    percent = "{:+.3f}".format
    print(table.to_string(formatters={
        "value over bench": "{:.1f}".format,
        "Q1 2 fouls": "{:.0%}".format, "Q3 4 fouls": "{:.0%}".format, "Q4 5 fouls": "{:.0%}".format,
        "cost if worth 6": lambda v: percent(v * 100),
        "cost per starter-game": lambda v: percent(v * 100), "low": lambda v: percent(v * 100),
        "high": lambda v: percent(v * 100), "season wins": "{:+.2f}".format,
        "optimal gain": lambda v: percent(v * 100)}))
    clear = table[(table["high"] < 0) | (table["low"] > 0)]
    print(f"\n{len(clear)} of {len(table)} coaches have a 95% range that excludes zero.")
    print("rank agreement with the fixed 6-point assumption (Spearman): "
          f"{table['cost per starter-game'].rank().corr(table['cost if worth 6'].rank()):.2f}")


if __name__ == "__main__":
    main()
