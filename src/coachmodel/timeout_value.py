"""What is a timeout worth? Two measurements the timeout decision depends on.

(a) Does a timeout stop a run? Stoppages while the opponent is on a run (6+ unanswered
    points) are compared over the next 3 minutes: coach timeout, TV timeout (forced by
    the clock rules, so a break no coach chose), or no timeout. The raw comparison is
    misleading: runs end on their own (regression to the mean) and coaches call timeouts
    when they have the ball. So the 3-minute point differential is regressed on the kind
    of stoppage plus the game state (run, score, time, possession, what stopped play,
    fouls, timeouts left, team strength), with standard errors clustered by game.

(b) What is a timeout saved for the end worth? For each team in a game within 6 points
    at 2:00 of the 4th, whether it won is regressed (linear probability, clustered by
    game) on the win probability model's prediction at that moment (margin, possession,
    team strength) and both teams' usable timeouts (at most 2 may be used in the last 3
    minutes, so 3+ left counts as 2). The timeout coefficient is win probability per
    extra usable timeout.

Usage:
    python -m coachmodel.timeout_value
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from coachmodel.build_db import DATABASE_PATH, REPO_ROOT
from coachmodel.win_probability import team_rating

RUN_SQL = REPO_ROOT / "sql" / "models" / "timeout_run_effects.sql"
END_GAME_SQL = REPO_ROOT / "sql" / "models" / "timeout_endgame.sql"
LAST_THREE_MINUTES_LIMIT = 2  # NBA rule: at most 2 timeouts in the last 3 minutes
RATING_GAMES, RATING_PRIOR_WEIGHT = 10, 0.5  # as fit for win probability


def clustered_least_squares(design: pd.DataFrame, outcome: np.ndarray, clusters: np.ndarray) -> pd.DataFrame:
    """Ordinary least squares with cluster-robust (sandwich) standard errors:
    (X'X)^-1 [sum over clusters of X_g' e_g e_g' X_g] (X'X)^-1."""
    matrix = design.to_numpy(dtype=float)
    bread = np.linalg.inv(matrix.T @ matrix)
    coefficients = bread @ matrix.T @ outcome
    residuals = outcome - matrix @ coefficients
    scores = pd.DataFrame(matrix * residuals[:, None]).groupby(clusters).sum().to_numpy()
    covariance = bread @ (scores.T @ scores) @ bread
    errors = np.sqrt(np.diag(covariance))
    return pd.DataFrame({"estimate": coefficients, "se": errors,
                         "low": coefficients - 1.96 * errors, "high": coefficients + 1.96 * errors},
                        index=design.columns)


def run_design(rows: pd.DataFrame) -> pd.DataFrame:
    """Kind of stoppage (vs no timeout) plus the game-state controls."""
    gap = (team_rating(rows["team_games_before"], rows["team_margin_before"], rows["team_previous_season_margin"],
                       RATING_GAMES, RATING_PRIOR_WEIGHT)
           - team_rating(rows["opponent_games_before"], rows["opponent_margin_before"],
                         rows["opponent_previous_season_margin"], RATING_GAMES, RATING_PRIOR_WEIGHT))
    run = rows["opponent_run"].clip(upper=20).astype(float)
    margin = rows["score_margin"].clip(-25, 25).astype(float)
    design = pd.DataFrame({
        "intercept": 1.0,
        "coach timeout": (rows["stoppage_kind"] == "coach timeout").astype(float),
        "TV timeout": (rows["stoppage_kind"] == "TV timeout").astype(float),
        "opponent run": run, "opponent run squared": run ** 2,
        "score margin": margin, "absolute margin": margin.abs(),
        "minutes left in period": rows["period_seconds_left"] / 60,
        "has ball": rows["has_ball"].astype(float),
        "is home": rows["is_home"].astype(float),
        "team fouls": rows["team_fouls_in_period"].astype(float),
        "opponent fouls": rows["opponent_fouls_in_period"].astype(float),
        "timeouts remaining": rows["timeouts_remaining"].astype(float),
        "rating gap": np.nan_to_num(gap),
    }, index=rows.index)
    for period in (2, 3, 4):
        design[f"quarter {period}"] = (rows["period"] == period).astype(float)
    causes = pd.get_dummies(rows["stoppage_cause"], prefix="cause", dtype=float)
    return pd.concat([design, causes.iloc[:, 1:]], axis=1)  # first cause is the baseline


def run_effects(rows: pd.DataFrame) -> pd.DataFrame:
    fit = clustered_least_squares(run_design(rows), rows["next_3_minutes"].to_numpy(dtype=float),
                                  rows["game_id"].to_numpy())
    return fit.loc[["coach timeout", "TV timeout", "has ball", "opponent run"]]


def end_game_value(rows: pd.DataFrame, win_model) -> pd.DataFrame:
    """Win probability per extra usable timeout at 2:00 of a close game."""
    home_margin = np.where(rows["is_home"], rows["score_margin"], -rows["score_margin"])
    home_ball = np.where(rows["is_home"] == rows["has_ball"], 1, -1)
    gap = (team_rating(rows["team_games_before"], rows["team_margin_before"], rows["team_previous_season_margin"],
                       RATING_GAMES, RATING_PRIOR_WEIGHT)
           - team_rating(rows["opponent_games_before"], rows["opponent_margin_before"],
                         rows["opponent_previous_season_margin"], RATING_GAMES, RATING_PRIOR_WEIGHT))
    home_gap = np.nan_to_num(np.where(rows["is_home"], gap, -gap))
    home_win = win_model.home_win_probability(home_margin, np.full(len(rows), 120), home_ball, home_gap)
    predicted = np.where(rows["is_home"], home_win, 1 - home_win)
    design = pd.DataFrame({
        "intercept": 1.0,
        "predicted win probability": predicted,
        "usable timeouts": rows["timeouts_left"].clip(upper=LAST_THREE_MINUTES_LIMIT).astype(float),
        "opponent usable timeouts": rows["opponent_timeouts_left"].clip(upper=LAST_THREE_MINUTES_LIMIT).astype(float),
    }, index=rows.index)
    fit = clustered_least_squares(design, rows["won"].to_numpy(dtype=float), rows["game_id"].to_numpy())
    return fit.loc[["usable timeouts", "opponent usable timeouts", "predicted win probability"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()

    from coachmodel.win_probability import WinProbability, load_rows

    with duckdb.connect(str(args.database), read_only=True) as connection:
        rows = connection.sql(RUN_SQL.read_text()).df()
        end_game = connection.sql(END_GAME_SQL.read_text()).df()
        win_rows = load_rows(connection)

    print(f"(a) stoppages during an opponent run of 6+: {len(rows):,} in {rows['game_id'].nunique():,} games")
    raw = rows.groupby("stoppage_kind")["next_3_minutes"].agg(["size", "mean"])
    print("raw next-3-minute point differential (team minus opponent):")
    print(raw.round(2).to_string())
    effects = run_effects(rows)
    print("\nadjusted for game state, points over the next 3 minutes vs. no timeout (95% range, clustered by game):")
    print(effects.round(2).to_string())
    print("\nsame, for bigger runs only:")
    for minimum_run in (8, 10):
        bigger = rows[rows["opponent_run"] >= minimum_run]
        effect = run_effects(bigger).loc["coach timeout"]
        print(f"  runs of {minimum_run}+ ({len(bigger):,} stoppages): coach timeout {effect['estimate']:+.2f} "
              f"({effect['low']:+.2f} to {effect['high']:+.2f})")

    win_model = WinProbability().fit(win_rows[win_rows["season_type"] == "Regular Season"])
    usable = end_game["timeouts_left"].clip(upper=LAST_THREE_MINUTES_LIMIT)
    print(f"\n(b) teams in games within 6 points at 2:00 of the 4th: {len(end_game):,} "
          f"({end_game['game_id'].nunique():,} games); usable timeouts 2: {(usable == 2).sum():,}, "
          f"1: {(usable == 1).sum():,}, 0: {(usable == 0).sum():,}")
    value = end_game_value(end_game, win_model)
    print("win probability per extra usable timeout, adjusted for the state at 2:00 (95% range, clustered by game):")
    print(value.round(3).to_string())


if __name__ == "__main__":
    main()
