"""Do starters benched for early foul trouble get their minutes back later in the game?

The foul-trouble MDP assumes not: a coach's policy depends only on the clock and the
player's fouls, so minutes lost to benching are gone. If coaches make them up in the
second half, benching costs less than the audit says.

Each starter is compared with himself: his minutes in close games with early foul
trouble (2+ fouls in Q1, or 3+ by halftime) against his own close games without it.
Second-half minutes above his usual level are minutes made up. Games where he fouled out
are shown separately, since those minutes cannot be made up. Ranges come from resampling
players.

Usage:
    python -m coachmodel.foul_trouble_minutes
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

MINUTES_SQL = REPO_ROOT / "sql" / "models" / "foul_trouble_minutes.sql"
BOOTSTRAP_SAMPLES = 2000
HALVES = ["first_half_minutes", "second_half_minutes"]


def early_trouble(rows: pd.DataFrame) -> pd.Series:
    """The coaches' rule would bench him: 2+ fouls in Q1, or 3+ by halftime."""
    return (rows["fouls_after_q1"] >= 2) | (rows["fouls_at_half"] >= 3)


def minutes_change(rows: pd.DataFrame) -> pd.DataFrame:
    """Per player: his average minutes in early-trouble games minus his average in his
    other close games, by half. Only players with both kinds of games."""
    rows = rows.assign(total_minutes=rows[HALVES].sum(axis=1), trouble=early_trouble(rows))
    columns = HALVES + ["total_minutes"]
    usual = rows[~rows["trouble"]].groupby("person_id")[columns].mean()
    troubled = rows[rows["trouble"]].groupby("person_id")[columns]
    change = troubled.mean() - usual.reindex(troubled.mean().index)
    change["games"] = troubled.size()
    return change.dropna()


def weighted_change(change: pd.DataFrame) -> pd.Series:
    columns = HALVES + ["total_minutes"]
    return change[columns].mul(change["games"], axis=0).sum() / change["games"].sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    rng = np.random.default_rng(124)

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = connection.sql(MINUTES_SQL.read_text()).df()
    trouble = early_trouble(rows)
    print(f"{len(rows):,} starter-games in close games; early foul trouble in {trouble.mean():.1%}; "
          f"fouled out in {rows.loc[trouble, 'fouled_out'].mean():.1%} of those "
          f"vs {rows.loc[~trouble, 'fouled_out'].mean():.1%} of the others\n")

    for label, subset in [("did not foul out", rows[~rows["fouled_out"]]), ("all games", rows)]:
        change = minutes_change(subset)
        estimate = weighted_change(change)
        draws = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sample = change.iloc[rng.integers(0, len(change), len(change))]
            draws.append(weighted_change(sample))
        draws = pd.DataFrame(draws)
        low, high = draws.quantile(0.025), draws.quantile(0.975)
        print(f"early-trouble games vs. the same starter's other close games ({label}; "
              f"{len(change)} starters, {int(change['games'].sum()):,} trouble games):")
        for column, name in zip(HALVES + ["total_minutes"], ["first half", "second half", "whole game"]):
            print(f"  {name:12s} {estimate[column]:+.2f} minutes   ({low[column]:+.2f} to {high[column]:+.2f})")
        made_up = estimate["second_half_minutes"] / -estimate["first_half_minutes"]
        print(f"  share of first-half minutes made up after halftime: {made_up:.0%}\n")


if __name__ == "__main__":
    main()
