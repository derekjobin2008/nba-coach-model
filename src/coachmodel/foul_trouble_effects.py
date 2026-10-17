"""Does foul trouble change how a player plays? Each player is compared with himself.

For every player and quarter, his low-foul minutes (0 or 1 fouls) set a baseline: fouls
per minute, and points his team scores and allows per minute. His foul-trouble minutes
in the same quarter are then compared with what that baseline predicts for them:
    ratio = what happened / what his own baseline predicts
Comparing within player removes the selection problem (foul-prone big men are the ones
who pile up fouls); comparing within quarter removes changes in pace and play across the
game. Ranges come from resampling players (bootstrap).

Three readings:
    fouls ratio < 1      he plays cautiously
    fouls ratio > 1      opponents target him to draw the next foul (Derek's observation)
    points allowed > 1   his team defends worse with him in foul trouble (value drops)

Usage:
    python -m coachmodel.foul_trouble_effects
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

EFFECTS_SQL = REPO_ROOT / "sql" / "models" / "foul_trouble_effects.sql"
BASELINE_FOULS = 1  # 0 or 1 fouls: not in foul trouble in any quarter
BOOTSTRAP_SAMPLES = 2000
MEASURES = {"fouls": "fouls", "points allowed": "opponent_points", "points scored": "team_points"}


def in_trouble(quarter: int, fouls_now: int) -> bool:
    """Coaches' own rule: more fouls than the quarter number (in the 4th, 5 fouls)."""
    return fouls_now >= 5 if quarter == 4 else fouls_now > quarter


def compare_with_baseline(rows: pd.DataFrame) -> pd.DataFrame:
    """Add each row's expected count for every measure, from the same player's
    low-foul rate in the same quarter. Rows without a baseline are dropped."""
    baseline = (rows[rows["fouls_now"] <= BASELINE_FOULS]
                .groupby(["person_id", "quarter"])[["minutes", *MEASURES.values()]].sum())
    rates = baseline[list(MEASURES.values())].div(baseline["minutes"], axis=0)
    rates = rates[baseline["minutes"] >= 30]  # need some low-foul minutes to set a baseline
    compared = rows.join(rates, on=["person_id", "quarter"], rsuffix="_rate", how="inner")
    for column in MEASURES.values():
        compared[f"{column}_expected"] = compared[f"{column}_rate"] * compared["minutes"]
    return compared


def ratios(compared: pd.DataFrame) -> pd.Series:
    totals = compared.sum(numeric_only=True)
    return pd.Series({name: totals[column] / totals[f"{column}_expected"] for name, column in MEASURES.items()})


def bootstrap_ratios(compared: pd.DataFrame, rng) -> pd.DataFrame:
    """Ratios recomputed on players resampled with replacement: a 95% range."""
    per_player = compared.groupby("person_id")[
        [*MEASURES.values(), *[f"{c}_expected" for c in MEASURES.values()]]].sum()
    samples = rng.integers(0, len(per_player), (BOOTSTRAP_SAMPLES, len(per_player)))
    values = per_player.to_numpy()
    sums = values[samples].sum(axis=1)
    count = len(MEASURES)
    draws = sums[:, :count] / sums[:, count:]
    return pd.DataFrame(np.percentile(draws, [2.5, 97.5], axis=0), index=["low", "high"], columns=list(MEASURES))


def fourth_quarter_foul_multipliers(rows: pd.DataFrame) -> list[float]:
    """Foul rate at each foul count relative to the player's low-foul rate, from the 4th
    quarter, where coaches rarely bench for fouls (so the least selection). Used by the
    foul-trouble MDP."""
    compared = compare_with_baseline(rows[rows["quarter"] == 4])
    multipliers = [1.0] * 6
    for fouls_now in range(BASELINE_FOULS + 1, 6):
        cell = compared[compared["fouls_now"] == fouls_now]
        multipliers[fouls_now] = float(cell["fouls"].sum() / cell["fouls_expected"].sum())
    return multipliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    rng = np.random.default_rng(124)

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = connection.sql(EFFECTS_SQL.read_text()).df()
    compared = compare_with_baseline(rows)
    compared["in_trouble"] = [in_trouble(q, f) for q, f in zip(compared["quarter"], compared["fouls_now"])]

    print("what happened / what each player's own low-foul baseline predicts (1.00 = no change)\n")
    table = []
    for (quarter, fouls_now), cell in compared[compared["fouls_now"] > BASELINE_FOULS].groupby(["quarter", "fouls_now"]):
        if cell["minutes"].sum() < 500:
            continue
        table.append({"quarter": quarter, "foul count": fouls_now, "minutes": cell["minutes"].sum(),
                      "in trouble": "yes" if in_trouble(quarter, fouls_now) else "", **ratios(cell)})
    print(pd.DataFrame(table).to_string(index=False, formatters={
        "minutes": "{:,.0f}".format, **{name: "{:.2f}".format for name in MEASURES}}))

    trouble = compared[compared["in_trouble"]]
    overall = ratios(trouble)
    ranges = bootstrap_ratios(trouble, rng)
    print(f"\nall foul-trouble minutes ({trouble['minutes'].sum():,.0f} minutes, "
          f"{trouble['person_id'].nunique()} players), with 95% ranges over players:")
    for name in MEASURES:
        print(f"  {name:15s} {overall[name]:.3f}   ({ranges.loc['low', name]:.3f} to {ranges.loc['high', name]:.3f})")
    expected_minutes = trouble["minutes"].sum()
    net_change = ((trouble["team_points"].sum() - trouble["team_points_expected"].sum())
                  - (trouble["opponent_points"].sum() - trouble["opponent_points_expected"].sum()))
    print(f"  net points per 48 minutes vs. his baseline: {net_change / expected_minutes * 48:+.2f}")


if __name__ == "__main__":
    main()
