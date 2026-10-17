"""Download NBA game lists and play-by-play from stats.nba.com and cache the raw JSON.

Python only fetches and saves; all parsing and reshaping happens in SQL (see sql/).
Every response is written to data/raw/ so a crash, a block, or a schema mistake never
costs a re-download: reruns skip files that already exist.

Usage:
    python -m coachmodel.ingest --season 2024-25
    python -m coachmodel.ingest --season 2023-24 --season-type Playoffs --game-id 0042300217
"""

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonallplayers, commonteamroster, leaguegamefinder, playbyplayv3

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
# stats.nba.com stopped answering after ~600 requests at 0.6 s apart (2026-10-11): it
# silently times out rather than returning an error. So pace requests with jitter, and
# when blocked, cool down for longer and longer instead of skipping games.
SECONDS_BETWEEN_REQUESTS = 1.5
RETRY_WAITS_SECONDS = [10, 60, 300, 900]


def fetch_with_retries(fetch):
    """Call fetch(); on failure wait and retry with growing cool-downs."""
    for wait in RETRY_WAITS_SECONDS:
        try:
            return fetch()
        except Exception as error:  # nba_api raises plain requests/JSON errors
            print(f"  request failed ({type(error).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    return fetch()


def season_file_stem(season: str, season_type: str) -> str:
    return f"{season}_{season_type.replace(' ', '_')}"


def download_game_list(season: str, season_type: str) -> list[dict]:
    """Save one row per team per game (two per game) as JSON records; return them."""
    path = RAW_DIR / "games" / f"{season_file_stem(season, season_type)}.json"
    if path.exists():
        return json.loads(path.read_text())
    frame = fetch_with_retries(
        lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",
            timeout=30,
        ).get_data_frames()[0]
    )
    frame["SEASON"] = season
    frame["SEASON_TYPE"] = season_type
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_json(orient="records"))  # to_json writes null, not NaN
    return json.loads(path.read_text())


def download_player_list() -> None:
    """Save every NBA player ever (id, full name) once; used to resolve short names."""
    path = RAW_DIR / "players.json"
    if path.exists():
        return
    frame = fetch_with_retries(
        lambda: commonallplayers.CommonAllPlayers(
            is_only_current_season=0, league_id="00", timeout=30
        ).get_data_frames()[0]
    )
    path.write_text(frame.to_json(orient="records"))  # to_json writes null, not NaN


def download_head_coaches(season: str, team_ids: list[int]) -> None:
    """Save each team's head coach for the season (one request per team).

    The roster endpoint lists only the coach at the END of the season, so in-season
    changes are recorded by hand in data/coach_changes.csv.
    """
    path = RAW_DIR / "coaches" / f"{season}.json"
    if path.exists():
        return
    head_coach_frames = []
    for team_id in team_ids:
        coaches = fetch_with_retries(
            lambda: commonteamroster.CommonTeamRoster(
                team_id=team_id, season=season, timeout=30
            ).get_data_frames()[1]
        )
        head_coach_frames.append(coaches[coaches.COACH_TYPE == "Head Coach"])
        time.sleep(SECONDS_BETWEEN_REQUESTS + random.uniform(0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    head_coaches = pd.concat(head_coach_frames)
    head_coaches["SEASON_LABEL"] = season  # the endpoint's SEASON is "2024" for 2024-25
    path.write_text(head_coaches.to_json(orient="records"))


def download_play_by_play(game_id: str) -> bool:
    """Save the raw PlayByPlayV3 response for one game. Returns True if newly downloaded."""
    path = RAW_DIR / "pbp" / f"{game_id}.json"
    if path.exists():
        return False
    response = fetch_with_retries(
        lambda: playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30).get_dict()
    )
    if not response["game"]["actions"]:
        raise ValueError(f"game {game_id} returned no play-by-play actions")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name and rename, so the database build never reads a half-written file.
    partial_path = path.with_suffix(".partial")
    partial_path.write_text(json.dumps(response))
    partial_path.rename(path)
    time.sleep(SECONDS_BETWEEN_REQUESTS + random.uniform(0, 1))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", required=True, help='e.g. "2024-25"')
    parser.add_argument("--season-type", default="Regular Season", choices=["Regular Season", "Playoffs"])
    parser.add_argument("--game-id", action="append", help="only these games (repeatable)")
    args = parser.parse_args()

    download_player_list()
    games = download_game_list(args.season, args.season_type)
    game_ids = sorted({row["GAME_ID"] for row in games})
    if args.game_id:
        missing = set(args.game_id) - set(game_ids)
        if missing:
            parser.error(f"not in {args.season} {args.season_type}: {sorted(missing)}")
        game_ids = sorted(args.game_id)

    failed = []
    for index, game_id in enumerate(game_ids, start=1):
        try:
            if download_play_by_play(game_id):
                print(f"[{index}/{len(game_ids)}] {game_id} downloaded", flush=True)
        except Exception as error:
            failed.append(game_id)
            print(f"[{index}/{len(game_ids)}] {game_id} FAILED: {error}")
    print(f"done: {len(game_ids) - len(failed)} of {len(game_ids)} games cached in {RAW_DIR / 'pbp'}")
    download_head_coaches(args.season, sorted({row["TEAM_ID"] for row in games}))
    if failed:
        print("failed games (rerun to retry):", " ".join(failed))


if __name__ == "__main__":
    main()
