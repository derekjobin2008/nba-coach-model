-- Game state from each team's side after every play: one row per play per team.
--
-- Decisions (08) and stoppages (09) both read their state from here, so the features
-- are defined once. Columns come from ASOF JOINs: for each play, the latest foul,
-- score, or timeout at or before it. A row can never see a later play, which is how
-- leakage is kept out. Timeouts left in overtime are not modeled yet (different rules).

-- Personal and team fouls. Descriptions carry running counts:
--   "(P3.T1)" = the player's 3rd personal foul, the team's 1st foul this period;
--   "(P3.PN)" = the team is already in the penalty;
--   "(P2)"    = offensive foul, which is personal but not a team foul.
-- Technicals reuse the P number for technical counts, so they are left out.
CREATE OR REPLACE TABLE personal_fouls AS
SELECT
    game_id, action_id, period, team_id, person_id,
    CAST(regexp_extract(description, '\(P(\d+)', 1) AS INTEGER) AS player_fouls,
    regexp_matches(description, '\(P\d+\.(T\d+|PN)\)') AS is_team_foul,
    regexp_matches(description, '\(P\d+\.PN\)') AS team_in_penalty
FROM pbp
WHERE action_type = 'Foul'
  AND sub_type NOT ILIKE '%technical%'
  AND sub_type NOT IN ('Defense 3 Second', 'Flopping')
  AND regexp_matches(description, '\(P\d+');

CREATE OR REPLACE TABLE team_state AS
WITH team_plays AS (
    -- Every play twice, once from each team's side.
    SELECT
        pbp.game_id, pbp.action_id, pbp.period, pbp.period_seconds_left, pbp.game_seconds_elapsed,
        sides.is_home,
        CASE WHEN sides.is_home THEN games.home_team_id ELSE games.away_team_id END AS team_id,
        CASE WHEN sides.is_home THEN games.away_team_id ELSE games.home_team_id END AS opponent_team_id,
        -- Positive margin means this team leads.
        CASE WHEN sides.is_home THEN score_home - score_away ELSE score_away - score_home END AS score_margin,
        possession.team_with_ball,
        CASE WHEN sides.is_home THEN score_away ELSE score_home END AS opponent_score
    FROM pbp
    JOIN games USING (game_id)
    JOIN possession USING (game_id, action_id)
    CROSS JOIN (VALUES (true), (false)) AS sides(is_home)
),
team_fouls AS (
    -- Running team-foul count per period. Counting rows (instead of reading T#) stays
    -- correct after the feed switches from "T4" to "PN".
    SELECT
        game_id, action_id, period, team_id, team_in_penalty,
        count(*) OVER (PARTITION BY game_id, team_id, period ORDER BY action_id) AS team_fouls_in_period
    FROM personal_fouls
    WHERE is_team_foul
),
scoring_plays AS (
    -- Each play where a team's score went up, with the opponent's score at that moment.
    SELECT game_id, action_id, team_id AS scoring_team_id, opponent_score
    FROM (
        SELECT *, score_margin - lag(score_margin, 1, 0) OVER team_order AS margin_change,
               opponent_score - lag(opponent_score, 1, 0) OVER team_order AS opponent_change
        FROM team_plays
        WINDOW team_order AS (PARTITION BY game_id, team_id ORDER BY action_id)
    )
    -- This team scored when its margin rose by more than the opponent's score did.
    WHERE margin_change + opponent_change > 0
),
timeouts AS (
    SELECT game_id, action_id, team_id, team_timeouts_used
    FROM decision_events
    WHERE event_type = 'timeout'
),
team_rest AS (
    -- Days off before each game, per team: the gap to that team's previous game.
    SELECT
        game_id, team_id,
        game_date - lag(game_date) OVER (PARTITION BY team_id, season ORDER BY game_date) - 1 AS rest_days
    FROM (
        SELECT game_id, season, game_date, home_team_id AS team_id FROM games
        UNION ALL
        SELECT game_id, season, game_date, away_team_id FROM games
    )
)
SELECT
    t.game_id, t.action_id, t.team_id, t.opponent_team_id, t.is_home,
    t.period, t.period_seconds_left, t.game_seconds_elapsed, t.score_margin,
    t.team_with_ball = t.team_id AS has_ball,
    coalesce(own_fouls.team_fouls_in_period, 0) AS team_fouls_in_period,
    coalesce(own_fouls.team_in_penalty, false) AS team_in_penalty,
    coalesce(opponent_fouls.team_fouls_in_period, 0) AS opponent_fouls_in_period,
    coalesce(opponent_fouls.team_in_penalty, false) AS opponent_in_penalty,
    -- Opponent points since this team last scored ("unanswered run"). Floored at 0:
    -- a basket overturned on review can briefly make the difference negative.
    greatest(t.opponent_score - coalesce(last_own_score.opponent_score, 0), 0) AS opponent_run,
    -- Regulation allows 7 per team; overtime rules differ, so NULL there. Strictly
    -- before: a timeout's own row should not count against what it had left.
    CASE WHEN t.period <= 4
         THEN 7 - coalesce(timeouts_before.team_timeouts_used, 0) END AS timeouts_remaining,
    rest.rest_days
FROM team_plays AS t
ASOF LEFT JOIN team_fouls AS own_fouls
    ON own_fouls.game_id = t.game_id AND own_fouls.team_id = t.team_id
   AND own_fouls.period = t.period AND t.action_id >= own_fouls.action_id
ASOF LEFT JOIN team_fouls AS opponent_fouls
    ON opponent_fouls.game_id = t.game_id AND opponent_fouls.team_id = t.opponent_team_id
   AND opponent_fouls.period = t.period AND t.action_id >= opponent_fouls.action_id
ASOF LEFT JOIN scoring_plays AS last_own_score
    ON last_own_score.game_id = t.game_id AND last_own_score.scoring_team_id = t.team_id
   AND t.action_id >= last_own_score.action_id
ASOF LEFT JOIN timeouts AS timeouts_before
    ON timeouts_before.game_id = t.game_id AND timeouts_before.team_id = t.team_id
   AND t.action_id > timeouts_before.action_id
LEFT JOIN team_rest AS rest
    ON rest.game_id = t.game_id AND rest.team_id = t.team_id;
