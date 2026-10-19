-- What is known about each team's strength before each game: its point differential in
-- the season's earlier games, how many games that is, and its average margin over the
-- whole previous season. Ratings are built from these in Python (win_probability.py),
-- where the weights are fit; this table never looks at the game itself or later games.
CREATE OR REPLACE TABLE team_rating_inputs AS
WITH team_games AS (
    SELECT game_id, season, game_date, home_team_id AS team_id, home_points - away_points AS margin
    FROM games WHERE season_type = 'Regular Season'
    UNION ALL
    SELECT game_id, season, game_date, away_team_id, away_points - home_points
    FROM games WHERE season_type = 'Regular Season'
),
season_averages AS (
    SELECT team_id, season, avg(margin) AS margin_per_game
    FROM team_games GROUP BY ALL
),
previous_seasons AS (
    -- Each season paired with the one before it ("2024-25" -> "2023-24").
    SELECT season,
           (CAST(left(season, 4) AS INTEGER) - 1) || '-' || right(CAST(CAST(left(season, 4) AS INTEGER) AS VARCHAR), 2)
               AS previous_season
    FROM (SELECT DISTINCT season FROM team_games)
)
SELECT
    t.game_id, t.team_id, t.season, t.game_date,
    -- Earlier games only: the frame ends one row before the current game.
    count(*) OVER earlier AS games_before,
    coalesce(sum(t.margin) OVER earlier, 0) AS margin_before,
    previous.margin_per_game AS previous_season_margin
FROM team_games AS t
JOIN previous_seasons AS p USING (season)
LEFT JOIN season_averages AS previous
  ON previous.team_id = t.team_id AND previous.season = p.previous_season
WINDOW earlier AS (PARTITION BY t.team_id, t.season ORDER BY t.game_date, t.game_id
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

