-- Which team has the ball after each play.
--
-- The feed never says who has possession, so it is inferred from the plays that decide
-- it, using only that play and earlier ones:
--   made field goal by T          -> the other team inbounds
--   missed shot by T              -> T until someone rebounds
--   rebound by R (player or team) -> R
--   turnover by T                 -> the other team (a steal row names the stealer)
--   foul by T (not technical)     -> the other team: a defensive foul leaves the
--                                    offense with the ball, an offensive foul gives
--                                    it away
--   last free throw by T          -> made: the other team; missed: T until the rebound;
--                                    after a flagrant or clear-path foul, T keeps it
--   anything else (subs, timeouts, technicals, jump balls, violations) -> unchanged
-- Rows before the first deciding play of a period (the opening jump ball, or the
-- possession arrow) take the team of the period's first shot or turnover.
CREATE OR REPLACE TABLE possession AS
WITH plays AS (
    SELECT
        pbp.game_id, pbp.action_id, pbp.period, pbp.action_type, pbp.sub_type, pbp.description,
        -- Team plays (team rebounds and turnovers) carry the team id in person_id.
        coalesce(pbp.team_id, CASE WHEN pbp.person_id BETWEEN 1610612737 AND 1610612766
                                   THEN pbp.person_id END) AS acting_team,
        games.home_team_id, games.away_team_id
    FROM pbp JOIN games USING (game_id)
),
deciding_plays AS (
    SELECT
        game_id, action_id, period, action_type, acting_team,
        CASE WHEN acting_team = home_team_id THEN away_team_id ELSE home_team_id END AS other_team,
        CASE
            WHEN action_type = 'Made Shot' THEN 'to_other'
            WHEN action_type = 'Missed Shot' THEN 'to_actor'
            WHEN action_type = 'Rebound' THEN 'to_actor'
            WHEN action_type = 'Turnover' THEN 'to_other'
            WHEN description LIKE '%STEAL%' THEN 'to_actor'
            WHEN action_type = 'Foul' AND sub_type NOT ILIKE '%technical%'
                 AND sub_type NOT IN ('Defense 3 Second', 'Flopping') THEN 'to_other'
            WHEN action_type = 'Free Throw' AND sub_type NOT ILIKE '%technical%'
                 AND regexp_extract(sub_type, '(\d) of (\d)', 1) = regexp_extract(sub_type, '(\d) of (\d)', 2)
                THEN CASE
                    WHEN sub_type ILIKE '%flagrant%' OR sub_type ILIKE '%clear path%' THEN 'to_actor'
                    WHEN description LIKE 'MISS %' THEN 'to_actor'
                    ELSE 'to_other'
                END
        END AS rule
    FROM plays
),
known AS (
    SELECT
        game_id, action_id, period, action_type,
        CASE rule WHEN 'to_actor' THEN acting_team WHEN 'to_other' THEN other_team END AS team_with_ball,
        CASE WHEN action_type IN ('Made Shot', 'Missed Shot', 'Turnover') THEN acting_team END AS offense_team
    FROM deciding_plays
),
period_openers AS (
    -- The team of the first shot or turnover in each period had the ball to start it.
    SELECT game_id, period, arg_min(offense_team, action_id) AS opening_team
    FROM known
    WHERE offense_team IS NOT NULL
    GROUP BY ALL
)
SELECT
    k.game_id, k.action_id,
    coalesce(
        last_value(k.team_with_ball IGNORE NULLS) OVER (
            PARTITION BY k.game_id, k.period ORDER BY k.action_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
        o.opening_team
    ) AS team_with_ball
FROM known AS k
LEFT JOIN period_openers AS o USING (game_id, period);
