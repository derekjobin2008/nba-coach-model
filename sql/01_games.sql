-- One row per game, with home and away teams side by side.
--
-- The game-list endpoint returns two rows per game (one per team). MATCHUP reads
-- "IND vs. NYK" for the home team and "NYK @ IND" for the away team, so we pivot the
-- pair into one row with conditional aggregation: max(CASE ...) picks the value from
-- whichever of the two rows matches the condition. Neutral-site games (Paris, Mexico
-- City, NBA Cup in Las Vegas) read "X @ Y" on both rows; Y is the designated home team.
CREATE OR REPLACE TABLE games AS
WITH team_rows AS (
    SELECT
        GAME_ID AS game_id,
        SEASON AS season,
        SEASON_TYPE AS season_type,
        CAST(GAME_DATE AS DATE) AS game_date,
        TEAM_ID AS team_id,
        TEAM_ABBREVIATION AS team_abbr,
        MATCHUP LIKE '%vs.%' OR MATCHUP LIKE '%@ ' || TEAM_ABBREVIATION AS is_home,
        MATCHUP LIKE '%vs.%' AS listed_as_home,
        PTS AS points
    FROM read_json('data/raw/games/*.json')
)
SELECT
    game_id,
    any_value(season) AS season,
    any_value(season_type) AS season_type,
    any_value(game_date) AS game_date,
    max(CASE WHEN is_home THEN team_id END) AS home_team_id,
    max(CASE WHEN is_home THEN team_abbr END) AS home_team,
    max(CASE WHEN NOT is_home THEN team_id END) AS away_team_id,
    max(CASE WHEN NOT is_home THEN team_abbr END) AS away_team,
    max(CASE WHEN is_home THEN points END) AS home_points,
    max(CASE WHEN NOT is_home THEN points END) AS away_points,
    NOT bool_or(listed_as_home) AS is_neutral_site
FROM team_rows
GROUP BY game_id;
