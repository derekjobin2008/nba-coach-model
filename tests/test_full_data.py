"""Checks against the full downloaded database (data/coach.duckdb), using facts
anyone can look up. Skipped on a fresh clone until the data is downloaded and built."""

from pathlib import Path

import duckdb
import pytest

DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "coach.duckdb"

pytestmark = pytest.mark.skipif(not DATABASE_PATH.exists(), reason="full database not built")


@pytest.fixture(scope="module")
def database():
    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    yield connection
    connection.close()


def coach_record(database, season, coach_name):
    """Regular-season wins and losses for a coach with a given season's games."""
    return database.sql(f"""
        SELECT
            count(*) FILTER (WHERE (c.team_id = g.home_team_id) = (g.home_points > g.away_points)),
            count(*) FILTER (WHERE (c.team_id = g.home_team_id) <> (g.home_points > g.away_points))
        FROM game_coaches AS c JOIN games AS g USING (game_id)
        WHERE g.season = '{season}' AND g.season_type = 'Regular Season'
          AND c.coach_name = '{coach_name}'
    """).fetchone()


def season_is_complete(database, season):
    # Allow a few games missing (a download interrupted by the API quota can skip some).
    return database.sql(f"""
        SELECT count(DISTINCT game_id) >= 1225 FROM games JOIN pbp USING (game_id)
        WHERE season = '{season}' AND season_type = 'Regular Season'
    """).fetchone()[0]


@pytest.mark.parametrize(
    ("coach_name", "record_when_replaced"),
    [
        ("Mike Brown", (13, 18)),  # Kings, fired 2024-12-27
        ("Taylor Jenkins", (44, 29)),  # Grizzlies, fired 2025-03-28
        ("Michael Malone", (47, 32)),  # Nuggets, fired 2025-04-08
    ],
)
def test_fired_coaches_2024_25_records(database, coach_name, record_when_replaced):
    if not season_is_complete(database, "2024-25"):
        pytest.skip("2024-25 not fully downloaded")
    assert coach_record(database, "2024-25", coach_name) == record_when_replaced


def test_bucks_interim_record_2023_24(database):
    # Joe Prunty went 2-1 between Adrian Griffin's firing and Doc Rivers' debut.
    if not season_is_complete(database, "2023-24"):
        pytest.skip("2023-24 not fully downloaded")
    bucks_prunty = database.sql("""
        SELECT count(*) FROM game_coaches JOIN games USING (game_id)
        WHERE season = '2023-24' AND team = 'MIL' AND coach_name = 'Joe Prunty'
    """).fetchone()[0]
    assert bucks_prunty == 3


def test_roster_endpoint_agrees_with_head_coach_table(database):
    # The roster endpoint's snapshot is taken after the season, so for each team-season
    # it should name either a coach from our table for that season or the coach the
    # team had at the start of the next season (an off-season hire). Anything else
    # points to a mistake in data/head_coaches.csv. For the last season in the table
    # the off-season explanation cannot be checked (e.g. the 2025-26 snapshot already
    # shows the 2026 hires Taylor Jenkins at MIL and Jamahl Mosley at NOP), so only
    # seasons followed by another season in the table are checked.
    disagreements = database.sql("""
        WITH table_coaches AS (
            SELECT season, team, replace(coach_name, ' Jr.', '') AS coach
            FROM read_csv('data/head_coaches.csv')
        ),
        next_season_openers AS (
            SELECT season, team, replace(coach_name, ' Jr.', '') AS coach,
                   lead(coach_name) OVER (PARTITION BY team ORDER BY season) AS next_coach
            FROM read_csv('data/head_coaches.csv') WHERE note = 'opening coach'
        ),
        roster AS (
            SELECT r.SEASON_LABEL AS season, teams.team, replace(r.COACH_NAME, ' Jr.', '') AS coach
            FROM read_json('data/raw/coaches/*.json') AS r
            JOIN (SELECT DISTINCT home_team_id AS team_id, home_team AS team FROM games) AS teams
              ON teams.team_id = r.TEAM_ID
        )
        SELECT roster.season, roster.team, roster.coach
        FROM roster
        WHERE roster.season < (SELECT max(season) FROM table_coaches)
          AND NOT EXISTS (SELECT 1 FROM table_coaches AS t
                          WHERE t.season = roster.season AND t.team = roster.team AND t.coach = roster.coach)
          AND NOT EXISTS (SELECT 1 FROM next_season_openers AS n
                          WHERE n.season = roster.season AND n.team = roster.team
                            AND replace(n.next_coach, ' Jr.', '') = roster.coach)
    """).fetchall()
    assert disagreements == []
