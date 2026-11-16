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
from coachmodel.coach_effects import (
    coach_rate_ratios, fit_on_all_games, season_weighted_taus, tendency_and_season_uncertainty,
)
from coachmodel.foul_trouble_audit import audit_coaches, league_foul_trouble_shares

CACHE_PATH = REPO_ROOT / "data" / "scout_data.json"
TEMPLATE_PATH = Path(__file__).with_name("scout_template.html")
PAGE_PATH = REPO_ROOT / "docs" / "index.html"
HEAD_COACHES_PATH = REPO_ROOT / "data" / "head_coaches.csv"
MIN_GAMES = 40  # coaches with fewer regular-season games are left off the page
# A close game, for the "most likely next moves" card: within this many points with at
# most this many seconds left in the 4th quarter.
CLOSE_GAME_MAX_MARGIN = 5
CLOSE_GAME_MAX_SECONDS_LEFT = 360
MIN_PLAYER_STOPPAGES = 30  # a player needs this many close-game stoppages on the floor to be ranked
TOP_MOVES = 3

PLAYERS_PATH = REPO_ROOT / "data" / "raw" / "players.json"
PLAYER_NAMES_SQL = """
    SELECT PERSON_ID AS person_id, DISPLAY_FIRST_LAST AS player_name
    FROM read_json(?)
"""

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


def task_summary(connection, task_name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Per-coach rate ratios, per-stint rate ratios, and each coach's one-season
    uncertainty (log-odds)."""
    model, rows, offset = fit_on_all_games(connection, task_name)
    tendency, season = tendency_and_season_uncertainty(model, rows, offset, task_name)
    label = TASKS[task_name].label
    ratios = coach_rate_ratios(model, rows, offset, season, label)
    tendency_ratios = coach_rate_ratios(model, rows, offset, tendency, label)
    ratios["tendency_low"], ratios["tendency_high"] = tendency_ratios["low"], tendency_ratios["high"]
    return ratios, stint_rate_ratios(model, rows, offset), season


def latest_close_game(rows: pd.DataFrame) -> np.ndarray:
    """Stoppages in a close 4th quarter, from each coach's latest season."""
    close = (rows["is_fourth_or_overtime"].astype(bool) & ~rows["is_overtime"].astype(bool)
             & (rows["period_seconds_left"] <= CLOSE_GAME_MAX_SECONDS_LEFT)
             & (rows["abs_score_margin"] <= CLOSE_GAME_MAX_MARGIN))
    return (close & (rows["season"] == rows.groupby("coach_name")["season"].transform("max"))).to_numpy()


def move_probabilities(connection, task_name: str, season_sd: pd.Series) -> pd.DataFrame:
    """Each coach's chance of each move at a close-game stoppage in his latest season,
    averaged over his real stoppages, from the season-weighted fit. Also what an average
    coach would do in the same stoppages (his coach, team, and season effects removed)
    and a 95% range from shifting those effects by his one-season uncertainty."""
    model, rows, offset = fit_on_all_games(connection, task_name, season_weighted_taus(task_name))
    chosen = latest_close_game(rows)
    close_rows = rows[chosen]
    total = offset[chosen] + model.effects_for(close_rows)
    coach_part = model.effects_for(close_rows, ["coach", "stint", "season"])
    spread = 1.96 * close_rows["coach_name"].map(season_sd).fillna(season_sd.max()).to_numpy()
    moves = pd.DataFrame({
        "coach_name": close_rows["coach_name"].to_numpy(),
        "season": close_rows["season"].to_numpy(),
        "move": "timeout" if task_name == "timeout" else close_rows["person_id"].to_numpy(),
        "probability": expit(total), "league": expit(total - coach_part),
        "low": expit(total - spread), "high": expit(total + spread),
    })
    grouped = moves.groupby(["coach_name", "season", "move"])
    summary = grouped[["probability", "league", "low", "high"]].mean()
    summary["stoppages"] = grouped.size()
    if task_name == "substitution":
        summary = summary[summary["stoppages"] >= MIN_PLAYER_STOPPAGES]
    return summary.reset_index()


def likely_moves(timeout_moves: pd.DataFrame, substitution_moves: pd.DataFrame, player_names: dict,
                 clearly_differs: dict) -> dict[str, dict]:
    """Per coach: his top moves at a close-game stoppage, ranked by probability."""
    timeout_moves = timeout_moves.assign(kind="timeout", label="Call a timeout")
    substitution_moves = substitution_moves.assign(
        kind="substitution", label="Take out " + substitution_moves["move"].map(player_names).fillna("a player"))
    every_move = pd.concat([timeout_moves, substitution_moves])
    result = {}
    for coach_name, moves in every_move.groupby("coach_name"):
        top = moves.sort_values("probability", ascending=False).head(TOP_MOVES)
        result[coach_name] = {
            "season": str(moves["season"].iloc[0]),
            "stoppages": int(timeout_moves.loc[timeout_moves["coach_name"] == coach_name, "stoppages"].sum()),
            "moves": [{
                "label": move.label, "kind": move.kind,
                "probability": float(move.probability), "low": float(move.low), "high": float(move.high),
                "league": float(move.league), "differs": bool(clearly_differs[move.kind].get(coach_name, False)),
            } for move in top.itertuples()],
        }
    return result


def compute(connection: duckdb.DuckDBPyConnection, min_games: int = MIN_GAMES, players_path: Path = PLAYERS_PATH) -> dict:
    """Everything the page shows, for coaches with at least min_games regular-season games
    (player names for the likely-moves card from players_path)."""
    rng = np.random.default_rng(124)
    print("fitting the timeout model on every game...", flush=True)
    timeouts, timeout_stints, timeout_season_sd = task_summary(connection, "timeout")
    print("fitting the substitution model on every game (the slow one)...", flush=True)
    substitutions, substitution_stints, substitution_season_sd = task_summary(connection, "substitution")
    print("fitting the season-weighted models for the likely-moves card...", flush=True)
    clearly_differs = {kind: (table["tendency_low"] > 1) | (table["tendency_high"] < 1)
                       for kind, table in (("timeout", timeouts), ("substitution", substitutions))}
    moves = likely_moves(move_probabilities(connection, "timeout", timeout_season_sd),
                         move_probabilities(connection, "substitution", substitution_season_sd),
                         dict(connection.execute(PLAYER_NAMES_SQL, [str(players_path)]).fetchall()), clearly_differs)
    print("running the foul-trouble audit...", flush=True)
    audit = audit_coaches(connection, rng)
    audit["cost rank"] = audit["cost per starter-game"].rank().astype(int)  # 1 = costs the most
    games = connection.sql(COACH_GAMES_SQL).df()
    data_range = connection.sql(
        "SELECT min(season), max(season), count(DISTINCT game_id) FROM games JOIN pbp USING (game_id) "
        "WHERE season_type = 'Regular Season'").fetchone()

    coaches = []
    for coach_name, stints in games.groupby("coach_name"):
        if stints["games"].sum() < min_games or coach_name not in timeouts.index:
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
            "likely_moves": moves.get(coach_name),
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
        "close_game": {"max_margin": CLOSE_GAME_MAX_MARGIN, "max_seconds_left": CLOSE_GAME_MAX_SECONDS_LEFT},
        "league_foul_trouble": league_foul_trouble_shares(connection),
        "coaches": coaches,
    }


def current_coaches(path: Path = HEAD_COACHES_PATH) -> dict[str, str]:
    """Each team's head coach at the end of the latest season: the one hired last, so an
    interim coach replaces the coach he took over from."""
    table = pd.read_csv(path)
    latest = table[table["season"] == table["season"].max()].sort_values("from_date")
    return latest.groupby("team")["coach_name"].last().to_dict()


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
    data["current_coaches"] = current_coaches()  # from the hand-kept table, so no refit is needed
    PAGE_PATH.parent.mkdir(exist_ok=True)
    PAGE_PATH.write_text(render(data))
    print(f"wrote {PAGE_PATH.relative_to(REPO_ROOT)} ({len(data['coaches'])} coaches)")


if __name__ == "__main__":
    main()
