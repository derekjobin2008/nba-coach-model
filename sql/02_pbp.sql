-- Typed, cleaned play-by-play: one row per action, in game order.
--
-- Each raw file is {"meta": ..., "game": {"gameId": ..., "actions": [...]}}.
-- unnest(..., recursive := true) turns the actions array into rows and its fields
-- into columns, so the JSON is parsed entirely in SQL.
CREATE OR REPLACE TABLE pbp AS
WITH raw_actions AS (
    SELECT
        raw.game.gameId AS game_id,
        unnest(raw.game.actions, recursive := true)
    FROM read_json('data/raw/pbp/*.json', maximum_object_size = 64000000) AS raw
),
typed AS (
    SELECT
        game_id,
        -- actionId is unique and sequential within a game; actionNumber is not
        -- (a steal shares the turnover's actionNumber), so actionId defines order.
        actionId AS action_id,
        actionNumber AS action_number,  -- shared by linked rows, e.g. a turnover and its steal
        period,
        -- clock looks like 'PT06M30.00S': minutes and seconds left in the period.
        CAST(regexp_extract(clock, 'PT(\d+)M', 1) AS INTEGER) * 60
            + CAST(regexp_extract(clock, 'M([\d.]+)S', 1) AS DOUBLE) AS period_seconds_left,
        -- Timeout rows put the team id in personId and leave teamId as 0.
        CASE WHEN actionType = 'Timeout' THEN personId ELSE nullif(teamId, 0) END AS team_id,
        CASE WHEN actionType = 'Timeout' THEN NULL ELSE nullif(personId, 0) END AS person_id,
        nullif(playerName, '') AS player_name,
        nullif(playerNameI, '') AS player_name_initial,  -- e.g. 'L. James'
        nullif(actionType, '') AS action_type,
        nullif(subType, '') AS sub_type,
        description,
        -- Scores are only filled on scoring plays; blank elsewhere.
        CAST(nullif(scoreHome, '') AS INTEGER) AS score_home_if_changed,
        CAST(nullif(scoreAway, '') AS INTEGER) AS score_away_if_changed
    FROM raw_actions
)
SELECT
    * EXCLUDE (score_home_if_changed, score_away_if_changed),
    -- Seconds since tip-off: 12-minute quarters, then 5-minute overtimes.
    CASE
        WHEN period <= 4 THEN (period - 1) * 720 + (720 - period_seconds_left)
        ELSE 2880 + (period - 5) * 300 + (300 - period_seconds_left)
    END AS game_seconds_elapsed,
    -- Carry the last known score forward to every row. A window function looks at
    -- other rows of the same game (PARTITION BY) in order (ORDER BY); IGNORE NULLS
    -- makes last_value skip the blank rows, so each row sees the latest score so far.
    coalesce(last_value(score_home_if_changed IGNORE NULLS) OVER game_so_far, 0) AS score_home,
    coalesce(last_value(score_away_if_changed IGNORE NULLS) OVER game_so_far, 0) AS score_away
FROM typed
WINDOW game_so_far AS (
    PARTITION BY game_id ORDER BY action_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
);
