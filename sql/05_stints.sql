-- Who is on the court: one row per player stint (a continuous stretch on the floor).
--
-- The feed logs substitutions but never says who starts each period (lineups also
-- change at period breaks without a logged sub). So period starters are inferred:
-- a player started the period if they did something in it before their first sub-in.
-- Each starter or sub-in opens a stint; the next sub-out or the period end closes it.
CREATE OR REPLACE TABLE stints AS
WITH periods AS (
    SELECT DISTINCT
        game_id, period,
        CASE WHEN period <= 4 THEN (period - 1) * 720 ELSE 2880 + (period - 5) * 300 END AS start_seconds,
        CASE WHEN period <= 4 THEN period * 720 ELSE 2880 + (period - 4) * 300 END AS end_seconds
    FROM pbp
),
appearances AS (
    -- Anything that puts a player on the floor. Technical fouls are excluded because
    -- they can be called on players sitting on the bench.
    SELECT game_id, period, team_id, person_id, action_id
    FROM pbp
    WHERE person_id IS NOT NULL AND team_id IS NOT NULL AND action_type IS NOT NULL
      AND action_type <> 'Substitution'
      AND NOT (action_type = 'Foul' AND (sub_type ILIKE '%technical%'
                                        OR sub_type IN ('Defense 3 Second', 'Flopping')))
    UNION ALL
    SELECT game_id, period, team_id, player_out_id, action_id
    FROM decision_events
    WHERE event_type = 'substitution'
),
first_sub_in AS (
    SELECT game_id, period, team_id, player_in_id AS person_id, min(action_id) AS action_id
    FROM decision_events
    WHERE event_type = 'substitution' AND player_in_id IS NOT NULL
    GROUP BY ALL
),
seen_starters AS (
    SELECT DISTINCT a.game_id, a.period, a.team_id, a.person_id
    FROM appearances AS a
    LEFT JOIN first_sub_in AS f USING (game_id, period, team_id, person_id)
    WHERE f.action_id IS NULL OR a.action_id < f.action_id
),
sub_moves AS (
    -- Every substitution as two moves: the outgoing player leaves, the incoming enters.
    SELECT game_id, period, team_id, action_id, game_seconds_elapsed, player_out_id AS person_id, 'out' AS move
    FROM decision_events WHERE event_type = 'substitution'
    UNION ALL
    SELECT game_id, period, team_id, action_id, game_seconds_elapsed, player_in_id, 'in'
    FROM decision_events WHERE event_type = 'substitution' AND player_in_id IS NOT NULL
),
on_court_at_period_end AS (
    -- Starters plus net subs: a player is on at the end if their last move was 'in',
    -- or they started and never moved.
    SELECT s.game_id, s.period, s.team_id, s.person_id
    FROM seen_starters AS s
    LEFT JOIN (
        SELECT game_id, period, team_id, person_id, arg_max(move, action_id) AS last_move
        FROM sub_moves GROUP BY ALL
    ) AS m USING (game_id, period, team_id, person_id)
    WHERE m.last_move IS NULL OR m.last_move = 'in'
    UNION
    SELECT game_id, period, team_id, person_id
    FROM sub_moves
    GROUP BY ALL
    HAVING arg_max(move, action_id) = 'in'
),
team_periods_short AS (
    SELECT game_id, period, team_id, 5 - count(*) AS players_missing
    FROM seen_starters
    GROUP BY ALL
    HAVING count(*) < 5
),
silent_candidates AS (
    -- In a short period (mostly 5-minute overtimes) a player can play it all without a
    -- single logged action. For teams short of five, candidates are players who ended
    -- the previous period on the floor and did nothing in this one.
    SELECT short.game_id, short.period, short.team_id, short.players_missing, previous.person_id
    FROM team_periods_short AS short
    JOIN on_court_at_period_end AS previous
      ON previous.game_id = short.game_id AND previous.team_id = short.team_id
     AND previous.period = short.period - 1
    WHERE NOT EXISTS (
        SELECT 1 FROM appearances AS a
        WHERE a.game_id = short.game_id AND a.period = short.period
          AND a.team_id = short.team_id AND a.person_id = previous.person_id)
      AND NOT EXISTS (
        SELECT 1 FROM sub_moves AS m
        WHERE m.game_id = short.game_id AND m.period = short.period
          AND m.team_id = short.team_id AND m.person_id = previous.person_id)
),
silent_starters AS (
    -- Only fill when the candidates exactly cover the gap; otherwise we cannot tell
    -- which of them stayed on, and leaving the lineup short is the honest answer.
    SELECT game_id, period, team_id, person_id
    FROM silent_candidates
    QUALIFY count(*) OVER (PARTITION BY game_id, period, team_id) = players_missing
),
period_starters AS (
    SELECT * FROM seen_starters
    UNION
    SELECT * FROM silent_starters
),
moves AS (
    -- Starters "enter" at action 0, before anything else in the period.
    SELECT st.game_id, st.period, st.team_id, st.person_id,
           0 AS action_id, p.start_seconds AS seconds, 'in' AS move
    FROM period_starters AS st JOIN periods AS p USING (game_id, period)
    UNION ALL
    SELECT game_id, period, team_id, person_id, action_id, game_seconds_elapsed, move
    FROM sub_moves
),
paired AS (
    -- lead() looks at the player's next move in the period: an 'in' followed by an
    -- 'out' is one stint; an 'in' with no later move lasts to the end of the period.
    SELECT
        *,
        lead(move) OVER player_moves AS next_move,
        lead(seconds) OVER player_moves AS next_seconds,
        lead(action_id) OVER player_moves AS next_action_id
    FROM moves
    WINDOW player_moves AS (PARTITION BY game_id, period, team_id, person_id ORDER BY action_id)
)
SELECT
    paired.game_id, paired.period, paired.team_id, paired.person_id,
    paired.action_id AS start_action_id,
    paired.seconds AS start_seconds,
    -- NULL end action: the stint ran to the end of the period.
    CASE WHEN next_move = 'out' THEN next_action_id END AS end_action_id,
    CASE WHEN next_move = 'out' THEN next_seconds ELSE periods.end_seconds END AS end_seconds
FROM paired
JOIN periods USING (game_id, period)
WHERE paired.move = 'in';
