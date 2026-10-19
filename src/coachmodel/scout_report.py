"""Scout report: a one-page website where you pick an NBA head coach and see how he
manages timeouts, substitutions, and foul trouble compared with an average coach.

Two steps, so the page can be restyled without refitting the models:
    compute  fit the models on every game and run the foul-trouble audit (about 15
             minutes); results are cached in data/scout_data.json
    render   write docs/index.html from the cached results (instant)

The page is a single self-contained file (data embedded), so it works from GitHub Pages
or opened straight from disk.

Usage:
    python -m coachmodel.scout_report            # compute, then render
    python -m coachmodel.scout_report --render   # render from the cache only
"""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from coachmodel.baseline import TASKS
from coachmodel.build_db import DATABASE_PATH, REPO_ROOT
from coachmodel.coach_effects import coach_rate_ratios, fit_on_all_games, tendency_and_season_uncertainty
from coachmodel.foul_trouble_audit import audit_coaches, league_foul_trouble_shares

CACHE_PATH = REPO_ROOT / "data" / "scout_data.json"
TEMPLATE_PATH = Path(__file__).with_name("scout_template.html")
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
MIN_GAMES = 40  # coaches with fewer regular-season games are left off the page

COACH_GAMES_SQL = """
    -- Regular-season games per coach and team, with the seasons they span.
    SELECT c.coach_name, c.team, count(*) AS games,
           min(g.season) AS first_season, max(g.season) AS last_season
    FROM game_coaches AS c
    JOIN games AS g USING (game_id)
    JOIN (SELECT DISTINCT game_id FROM pbp) USING (game_id)
    WHERE g.season_type = 'Regular Season' AND c.coach_name IS NOT NULL
    GROUP BY ALL
"""


def stint_rate_ratios(model, rows: pd.DataFrame, offset: np.ndarray) -> pd.Series:
    """Per coach-team stint: predicted decisions with his coach and stint effects divided
    by what an average coach would make in the same situations."""
    total = offset + model.effects_for(rows)
    without = total - model.effects_for(rows, ["coach", "stint"])
    stints = rows["stint"].to_numpy()
    return pd.Series(expit(total)).groupby(stints).sum() / pd.Series(expit(without)).groupby(stints).sum()


def task_summary(connection, task_name: str) -> tuple[pd.DataFrame, pd.Series]:
    model, rows, offset = fit_on_all_games(connection, task_name)
    tendency, season = tendency_and_season_uncertainty(model, rows, offset, task_name)
    label = TASKS[task_name].label
    ratios = coach_rate_ratios(model, rows, offset, season, label)
    tendency_ratios = coach_rate_ratios(model, rows, offset, tendency, label)
    ratios["tendency_low"], ratios["tendency_high"] = tendency_ratios["low"], tendency_ratios["high"]
    return ratios, stint_rate_ratios(model, rows, offset)


def compute(connection: duckdb.DuckDBPyConnection) -> dict:
    rng = np.random.default_rng(124)
    print("fitting the timeout model on every game...", flush=True)
    timeouts, timeout_stints = task_summary(connection, "timeout")
    print("fitting the substitution model on every game (the slow one)...", flush=True)
    substitutions, substitution_stints = task_summary(connection, "substitution")
    print("running the foul-trouble audit...", flush=True)
    audit = audit_coaches(connection, rng)
    audit["cost rank"] = audit["cost per starter-game"].rank().astype(int)  # 1 = costs the most
    games = connection.sql(COACH_GAMES_SQL).df()
    data_range = connection.sql(
        "SELECT min(season), max(season), count(DISTINCT game_id) FROM games JOIN pbp USING (game_id) "
        "WHERE season_type = 'Regular Season'").fetchone()

    coaches = []
    for coach_name, stints in games.groupby("coach_name"):
        if stints["games"].sum() < MIN_GAMES or coach_name not in timeouts.index:
            continue
        teams = []
        for _, stint in stints.sort_values("first_season").iterrows():
            key = f"{coach_name} ({stint['team']})"
            teams.append({
                "team": stint["team"], "games": int(stint["games"]),
                "seasons": stint["first_season"] if stint["first_season"] == stint["last_season"]
                else f"{stint['first_season']} to {stint['last_season']}",
                "timeout_ratio": float(timeout_stints.get(key, np.nan)),
                "substitution_ratio": float(substitution_stints.get(key, np.nan)),
            })
        coach = {
            "name": coach_name,
            "games": int(stints["games"].sum()),
            "teams": teams,
            "timeouts": {key: float(value) for key, value in timeouts.loc[coach_name].items()},
            "substitutions": {key: float(value) for key, value in substitutions.loc[coach_name].items()},
            "foul_trouble": None,
        }
        if coach_name in audit.index:
            row = audit.loc[coach_name]
            coach["foul_trouble"] = {
                "close_games": int(row["close games"]),
                "q1_2_fouls": float(row["Q1 2 fouls"]), "q3_4_fouls": float(row["Q3 4 fouls"]),
                "q4_5_fouls": float(row["Q4 5 fouls"]),
                "cost_per_starter_game": float(row["cost per starter-game"]),
                "low": float(row["low"]), "high": float(row["high"]),
                "season_wins": float(row["season wins"]),
                "value_over_bench": float(row["value over bench"]),
                "cost_rank": int(row["cost rank"]),
            }
        coaches.append(coach)

    return {
        "seasons": f"{data_range[0]} to {data_range[1]}",
        "games": int(data_range[2]),
        "audited_coaches": int(len(audit)),
        "league_foul_trouble": league_foul_trouble_shares(connection),
        "coaches": coaches,
    }


def without_missing(value):
    """Replace NaN with None (null in the page), so missing numbers show as a dash."""
    if isinstance(value, dict):
        return {key: without_missing(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_missing(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def render(data: dict) -> str:
    # Escape "</" so no value in the data can end the embedding <script> tag early.
    embedded = json.dumps(without_missing(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text().replace("__SCOUT_DATA__", embedded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--render", action="store_true", help="only re-render the page from the cached results")
    args = parser.parse_args()

    if not args.render:
        with duckdb.connect(str(args.database), read_only=True) as connection:
            data = compute(connection)
        CACHE_PATH.write_text(json.dumps(data, indent=1))
        print(f"saved {CACHE_PATH.relative_to(REPO_ROOT)}")
    data = json.loads(CACHE_PATH.read_text())
    PAGE_PATH.parent.mkdir(exist_ok=True)
    PAGE_PATH.write_text(render(data))
    print(f"wrote {PAGE_PATH.relative_to(REPO_ROOT)} ({len(data['coaches'])} coaches)")


if __name__ == "__main__":
    main()
