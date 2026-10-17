"""Player value: regularized adjusted plus-minus (RAPM), points per 48 minutes.

Each lineup segment (the same ten players on the court) gives a margin per 48 minutes:
    margin = home edge + sum(home players' values) - sum(away players' values) + noise
fit by ridge regression. The ridge penalty pulls every value toward 0 (an average
player), so a player with few minutes stays near average: partial pooling again, with
its strength chosen by cross-validation over games.

Weights: a segment's noise grows with its length, but points come in lumps (a 5-second
segment can contain a 3-point play), so segments under MIN_SEGMENT_MINUTES are weighted
as if they were that long. With y = margin per 48 that gives weight minutes^2 /
max(minutes, MIN_SEGMENT_MINUTES).

Value over backup (the foul-trouble MDP's input) is a starter's value minus the
minutes-weighted value of his team's non-starters under the same coach.

Usage:
    python -m coachmodel.player_value
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

SEGMENTS_SQL = REPO_ROOT / "sql" / "models" / "lineup_segments.sql"
MIN_SEGMENT_MINUTES = 0.5
ALPHAS = [300, 1000, 3000, 10000, 30000]
FOLDS = 5


def design(segments: pd.DataFrame, player_ids: list[int]):
    """Sparse matrix with +1 for each home player and -1 for each away player on the
    court, the margin per 48, and the weights."""
    column = {player: index for index, player in enumerate(player_ids)}
    rows, columns, values = [], [], []
    for row, (home, away) in enumerate(zip(segments["home_players"], segments["away_players"])):
        for player in home:
            rows.append(row), columns.append(column[player]), values.append(1.0)
        for player in away:
            rows.append(row), columns.append(column[player]), values.append(-1.0)
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(segments), len(player_ids)))
    minutes = segments["minutes"].to_numpy()
    margin_per_48 = 48 * (segments["home_points"] - segments["away_points"]).to_numpy() / np.maximum(minutes, 1e-9)
    weights = minutes ** 2 / np.maximum(minutes, MIN_SEGMENT_MINUTES)
    return matrix, margin_per_48, weights


def weighted_squared_error(truth, predicted, weights) -> float:
    return float(np.sum(weights * (truth - predicted) ** 2) / np.sum(weights))


def choose_alpha(matrix, margin, weights, games) -> tuple[float, dict]:
    """Ridge strength by cross-validation over games. Also returns, per alpha, the
    held-out error relative to a model with no players (home edge only)."""
    errors = {alpha: 0.0 for alpha in ALPHAS}
    baseline = 0.0
    for train, test in GroupKFold(FOLDS).split(matrix, margin, games):
        home_edge = np.average(margin[train], weights=weights[train])
        baseline += weighted_squared_error(margin[test], home_edge, weights[test])
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(matrix[train], margin[train], sample_weight=weights[train])
            errors[alpha] += weighted_squared_error(margin[test], model.predict(matrix[test]), weights[test])
    best = min(errors, key=errors.get)
    return best, {alpha: 1 - error / baseline for alpha, error in errors.items()}


def fit_player_values(segments: pd.DataFrame) -> tuple[pd.Series, float, dict]:
    """Values (points per 48 vs an average player) by person_id, the chosen alpha, and
    the cross-validated improvement over the no-player model for each alpha."""
    player_ids = sorted({p for lineup in pd.concat([segments["home_players"], segments["away_players"]]) for p in lineup})
    matrix, margin, weights = design(segments, player_ids)
    alpha, improvements = choose_alpha(matrix, margin, weights, segments["game_id"].to_numpy())
    model = Ridge(alpha=alpha).fit(matrix, margin, sample_weight=weights)
    return pd.Series(model.coef_, index=player_ids), alpha, improvements


STARTER_SQL = """
    -- Starts and minutes per player under each coach at each team (regular season).
    SELECT c.coach_name, c.team_id, s.person_id,
           count(DISTINCT s.game_id) FILTER (WHERE s.period = 1 AND s.start_action_id = 0) AS starts,
           sum(s.end_seconds - s.start_seconds) / 60 AS minutes
    FROM stints AS s
    JOIN game_coaches AS c USING (game_id, team_id)
    JOIN games AS g USING (game_id)
    WHERE g.season_type = 'Regular Season' AND c.coach_name IS NOT NULL
    GROUP BY ALL
"""


def starter_value_over_bench(usage: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Per coach and team: minutes-weighted value of the regular starters (starting at
    least half of the games anyone started) minus that of everyone else."""
    usage = usage.assign(value=usage["person_id"].map(values).fillna(0.0))
    results = []
    for (coach, team), group in usage.groupby(["coach_name", "team_id"]):
        is_starter = group["starts"] >= 0.5 * group["starts"].max()
        starters, bench = group[is_starter], group[~is_starter]
        if bench["minutes"].sum() == 0:
            continue
        results.append({
            "coach_name": coach, "team_id": team, "minutes": group["minutes"].sum(),
            "starter_value": np.average(starters["value"], weights=starters["minutes"]),
            "bench_value": np.average(bench["value"], weights=bench["minutes"]),
        })
    table = pd.DataFrame(results)
    table["value_over_bench"] = table["starter_value"] - table["bench_value"]
    return table


def coach_value_over_bench(usage: pd.DataFrame, values: pd.Series) -> pd.Series:
    """One number per coach: his starters' value over his bench, across his teams."""
    by_team = starter_value_over_bench(usage, values)
    return by_team.groupby("coach_name").apply(
        lambda group: np.average(group["value_over_bench"], weights=group["minutes"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()

    with duckdb.connect(str(args.database), read_only=True) as connection:
        segments = connection.sql(SEGMENTS_SQL.read_text()).df()
        usage = connection.sql(STARTER_SQL).df()
        names = dict(connection.sql(f"""
            SELECT PERSON_ID, DISPLAY_FIRST_LAST FROM read_json('{REPO_ROOT / "data" / "raw" / "players.json"}')
        """).fetchall())
    values, alpha, improvements = fit_player_values(segments)

    print(f"{len(segments):,} lineup segments, {segments['game_id'].nunique():,} games, {len(values):,} players")
    print("cross-validated improvement over a no-player model (held-out weighted squared error):")
    for candidate, improvement in improvements.items():
        print(f"  alpha {candidate:>6}: {improvement:+.2%}" + ("   <- chosen" if candidate == alpha else ""))

    minutes = usage.groupby("person_id")["minutes"].sum()
    table = pd.DataFrame({"player": values.index.map(names), "value_per_48": values, "minutes": minutes.reindex(values.index)})
    regulars = table[table["minutes"] >= 3000].sort_values("value_per_48", ascending=False)
    print("\nhighest values (3,000+ minutes), points per 48 vs an average player:")
    print(regulars.head(12).set_index("player").round(1).to_string())
    print("\nlowest:")
    print(regulars.tail(5).set_index("player").round(1).to_string())

    by_coach = coach_value_over_bench(usage, values)
    print(f"\nstarters' value over their bench, per coach: median {by_coach.median():.1f} per 48, "
          f"range {by_coach.min():.1f} to {by_coach.max():.1f}")


if __name__ == "__main__":
    main()
