-- The head coach on each bench for each game.
--
-- The play-by-play never names coaches, and the roster endpoint is not reliable for past
-- seasons: it reports the staff as of a later snapshot, so a team that changed coaches
-- over the summer shows the new coach for the season before (e.g. the 2022-23 Bucks as
-- Adrian Griffin instead of Mike Budenholzer). So data/head_coaches.csv is the source:
-- every team's opening coach per season plus every in-season change. A game takes the
-- row whose from_date is the latest on or before the game date (an ASOF JOIN on dates).
-- The roster endpoint is kept as a cross-check (tests/test_full_data.py). Assistants and
-- one-off absences (illness, ejection) are not modeled: results describe the bench under
-- that head coach.
CREATE OR REPLACE TABLE game_coaches AS
WITH team_games AS (
    SELECT game_id, season, game_date, home_team_id AS team_id, home_team AS team FROM games
    UNION ALL
    SELECT game_id, season, game_date, away_team_id, away_team FROM games
),
head_coaches AS (
    SELECT season, team, coach_name, from_date
    FROM read_csv('data/head_coaches.csv', types = {'from_date': 'DATE'})
)
SELECT t.game_id, t.team_id, t.team, coaches.coach_name
FROM team_games AS t
ASOF LEFT JOIN head_coaches AS coaches
    ON coaches.season = t.season AND coaches.team = t.team AND t.game_date >= coaches.from_date;
