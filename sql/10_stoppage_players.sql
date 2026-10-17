-- Substitution opportunities: one row per player on the court at each stoppage, labeled
-- with whether the coach took that player out there. This is the training table for
-- "will this player be subbed out?", with the player-level confounds alongside the
-- team state: foul trouble, minutes so far, and how long the current stint has run.
CREATE OR REPLACE TABLE stoppage_players AS
WITH stints_with_history AS (
    -- Seconds the player had already played before each stint began: a running sum
    -- over the player's earlier stints. ROWS ... 1 PRECEDING stops before this stint.
    SELECT
        *,
        coalesce(sum(end_seconds - start_seconds) OVER (
            PARTITION BY game_id, person_id ORDER BY start_seconds, start_action_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0) AS seconds_before_stint
    FROM stints
),
stoppage_moments AS (
    SELECT DISTINCT game_id, stoppage_action_id, team_id, period, period_seconds_left, game_seconds_elapsed
    FROM stoppages
),
on_court AS (
    -- Players whose stint began before the stoppage (so not anyone subbed in at it) and
    -- had not ended before it.
    SELECT
        m.*,
        s.person_id,
        s.seconds_before_stint + (m.game_seconds_elapsed - s.start_seconds) AS seconds_played,
        m.game_seconds_elapsed - s.start_seconds AS stint_seconds,
        -- The stint ends at this stoppage when its closing sub belongs to it (which may be
        -- logged a few seconds later, e.g. at a timeout after a made basket).
        coalesce(closing_sub.stoppage_action_id = m.stoppage_action_id, false) AS subbed_out
    FROM stoppage_moments AS m
    JOIN stints_with_history AS s
      ON s.game_id = m.game_id AND s.team_id = m.team_id AND s.period = m.period
     AND s.start_action_id < m.stoppage_action_id
     AND (s.end_action_id IS NULL OR s.end_action_id >= m.stoppage_action_id)
    LEFT JOIN bench_action_stoppages AS closing_sub
      ON closing_sub.game_id = s.game_id AND closing_sub.action_id = s.end_action_id
)
SELECT
    on_court.game_id, on_court.stoppage_action_id, on_court.team_id, on_court.person_id,
    round(on_court.seconds_played / 60, 2) AS minutes_played,
    round(on_court.stint_seconds / 60, 2) AS stint_minutes,
    coalesce(player_fouls.player_fouls, 0) AS player_fouls,
    on_court.subbed_out
FROM on_court
-- Fouls up to and including the play that stopped the game (the coach has seen it).
ASOF LEFT JOIN personal_fouls AS player_fouls
    ON player_fouls.game_id = on_court.game_id AND player_fouls.person_id = on_court.person_id
   AND on_court.stoppage_action_id >= player_fouls.action_id;
