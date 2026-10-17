-- Timeout opportunities: one row per dead-ball stoppage per team, labeled with whether
-- that team's coach called a timeout there. This is the training table for "will the
-- coach call a timeout at the next stoppage?": the rows where they did not call one
-- matter as much as the rows where they did.
--
-- A stoppage starts with a dead-ball play: made shots, fouls, free throws, violations,
-- jump balls, replays, turnovers without a steal (a steal keeps the ball live), and
-- team rebounds (ball out of bounds, or the placeholder logged after a missed first free
-- throw; the feed leaves team_id empty and puts the team's id in the person field, so
-- they are recognised by person ids in the NBA team-id range).
-- Dead-ball plays at the same game clock form one stoppage.
--
-- Bench actions are not always logged at the stoppage's clock: after a made basket the
-- clock keeps running (outside the last 2 minutes), so a timeout called in response is
-- logged a few seconds later. So each timeout or sub is assigned to the most recent
-- dead-ball play if no live play (rebound, steal, missed shot) came between them and it
-- came within 4 seconds. On 2024-25 data, 88% of timeouts come within 2 s of a dead ball
-- and the counts fall off after 4 s; a second bump at 6-20 s is live-ball timeouts (a
-- team brings the ball up and calls one). Live-ball bench actions get no stoppage and
-- are left out of this task.

-- Every play except bench actions, marked dead-ball or live.
CREATE OR REPLACE TABLE game_plays AS
WITH steals AS (
    SELECT DISTINCT game_id, action_number
    FROM pbp
    WHERE description LIKE '%STEAL%'
)
SELECT
    pbp.game_id, pbp.action_id, pbp.period, pbp.period_seconds_left, pbp.game_seconds_elapsed,
    pbp.action_type IN ('Made Shot', 'Foul', 'Free Throw', 'Violation', 'Jump Ball', 'Instant Replay')
        OR (pbp.action_type = 'Turnover' AND steals.action_number IS NULL)
        OR (pbp.action_type = 'Rebound' AND pbp.person_id BETWEEN 1610612737 AND 1610612766) AS is_dead_ball
FROM pbp
LEFT JOIN steals USING (game_id, action_number)
WHERE coalesce(pbp.action_type, '') NOT IN ('Timeout', 'Substitution', 'period');

-- The first dead-ball play at each game clock identifies a stoppage.
CREATE OR REPLACE TABLE stoppage_starts AS
SELECT game_id, period, period_seconds_left, min(action_id) AS stoppage_action_id
FROM game_plays
WHERE is_dead_ball
GROUP BY ALL;

-- Which stoppage, if any, each timeout and substitution belongs to.
CREATE OR REPLACE TABLE bench_action_stoppages AS
SELECT
    bench.game_id, bench.action_id, bench.team_id, bench.event_type,
    -- NULL for live-ball bench actions.
    CASE WHEN previous.is_dead_ball
              AND bench.game_seconds_elapsed - previous.game_seconds_elapsed <= 4
         THEN starts.stoppage_action_id END AS stoppage_action_id
FROM decision_events AS bench
ASOF LEFT JOIN game_plays AS previous
    ON previous.game_id = bench.game_id AND bench.action_id > previous.action_id
LEFT JOIN stoppage_starts AS starts
    ON starts.game_id = previous.game_id AND starts.period = previous.period
   AND starts.period_seconds_left = previous.period_seconds_left;

CREATE OR REPLACE TABLE stoppages AS
WITH timeouts_here AS (
    SELECT
        b.game_id, b.stoppage_action_id, b.team_id,
        bool_or(NOT d.likely_mandatory) AS coach_timeout,
        bool_or(d.likely_mandatory) AS mandatory_timeout
    FROM bench_action_stoppages AS b
    JOIN decision_events AS d USING (game_id, action_id)
    WHERE b.event_type = 'timeout' AND b.stoppage_action_id IS NOT NULL
    GROUP BY ALL
),
mandatory_stoppages AS (
    SELECT DISTINCT game_id, stoppage_action_id FROM timeouts_here WHERE mandatory_timeout
)
SELECT
    s.game_id,
    s.stoppage_action_id,
    -- State is read at the stoppage's first play: the event that stopped play is
    -- known to the coach, but nothing the benches do afterwards is.
    state.* EXCLUDE (game_id, action_id),
    -- What stopped play, from this team's side (e.g. 'opponent_made_shot').
    CASE
        WHEN first_play.action_type = 'Rebound' THEN 'team_rebound'
        WHEN first_play.action_type IN ('Made Shot', 'Foul', 'Turnover', 'Free Throw', 'Violation')
            THEN CASE WHEN first_play.team_id = state.team_id THEN 'own_' ELSE 'opponent_' END
                 || lower(replace(first_play.action_type, ' ', '_'))
        ELSE lower(replace(first_play.action_type, ' ', '_'))
    END AS stoppage_cause,
    -- Label: this team's coach called a timeout at this stoppage.
    coalesce(t.coach_timeout, false) AS called_timeout,
    -- A scorer-forced (TV) timeout happened here, for either team. Coaches do not need
    -- to call their own then, so models may want to leave these stoppages out.
    m.game_id IS NOT NULL AS has_mandatory_timeout
FROM stoppage_starts AS s
JOIN team_state AS state
    ON state.game_id = s.game_id AND state.action_id = s.stoppage_action_id
JOIN pbp AS first_play
    ON first_play.game_id = s.game_id AND first_play.action_id = s.stoppage_action_id
LEFT JOIN timeouts_here AS t
    ON t.game_id = s.game_id AND t.stoppage_action_id = s.stoppage_action_id AND t.team_id = state.team_id
LEFT JOIN mandatory_stoppages AS m
    ON m.game_id = s.game_id AND m.stoppage_action_id = s.stoppage_action_id;
