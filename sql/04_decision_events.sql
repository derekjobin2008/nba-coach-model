-- Bench decisions: coach timeouts and substitutions, one row per decision, plus
-- lineup_changes, which groups the substitutions a team makes at the same stoppage.
--
-- Game state is kept minimal here (clock, score margin, timeouts used); the full state
-- table (fouls, runs, minutes, rest) comes later and joins on (game_id, action_id).
CREATE OR REPLACE TABLE decision_events AS
WITH team_actions AS (
    SELECT
        pbp.*,
        pbp.team_id = games.home_team_id AS is_home,
        -- Margin from the deciding team's side: positive means that team leads.
        CASE WHEN pbp.team_id = games.home_team_id
             THEN score_home - score_away
             ELSE score_away - score_home END AS score_margin
    FROM pbp
    JOIN games USING (game_id)
    WHERE action_type IN ('Timeout', 'Substitution')
      -- Coach's challenges are logged as timeouts but are a different decision.
      AND coalesce(sub_type, '') <> 'Coach Challenge'
),
timeouts AS (
    SELECT
        *,
        -- Descriptions end like "(Full 4 Short 0)" or "(Reg.4 Short 0)": the number
        -- is how many timeouts this team has used so far in the game.
        CAST(regexp_extract(description, '(Full |Reg\.)(\d+)', 2) AS INTEGER) AS team_timeouts_used,
        -- Order of this timeout among all timeouts in the period, and who took the first.
        row_number() OVER period_timeouts AS timeout_number_in_period,
        first_value(team_id) OVER period_timeouts AS first_timeout_team_id
    FROM team_actions
    WHERE action_type = 'Timeout'
    WINDOW period_timeouts AS (PARTITION BY game_id, period ORDER BY action_id)
)
SELECT
    game_id, action_id, period, period_seconds_left, game_seconds_elapsed,
    team_id, is_home, score_margin,
    'timeout' AS event_type,
    team_timeouts_used,
    -- Heuristic, to be checked by hand: in each regulation period the scorer forces a
    -- timeout at the first dead ball under 7:00 if none has been taken (charged to the
    -- home team), and again under 3:00 if fewer than two have been taken (charged to
    -- the other team). The play-by-play cannot tell these from coach calls, so we flag
    -- timeouts that fit that pattern rather than drop them.
    period <= 4 AND (
        (timeout_number_in_period = 1 AND period_seconds_left < 420 AND is_home)
        OR (timeout_number_in_period = 2 AND period_seconds_left < 180
            AND team_id <> first_timeout_team_id)
    ) AS likely_mandatory,
    NULL::BIGINT AS player_out_id,
    NULL AS player_out_name,
    NULL AS player_in_name,
    NULL::BIGINT AS player_in_id,
    description
FROM timeouts

UNION ALL BY NAME

SELECT
    game_id, action_id, period, period_seconds_left, game_seconds_elapsed,
    team_id, is_home, score_margin,
    'substitution' AS event_type,
    NULL AS team_timeouts_used,
    NULL AS likely_mandatory,
    -- The row's person is the player leaving; the incoming player appears only as a
    -- short name in "SUB: Toppin FOR Siakam", resolved to an id via player_names.
    person_id AS player_out_id,
    player_name AS player_out_name,
    regexp_extract(description, 'SUB: (.+) FOR ', 1) AS player_in_name,
    -- NULL for the ~1% of entrants who never appear in the play-by-play again.
    (SELECT names.person_id
     FROM player_names AS names
     WHERE names.game_id = team_actions.game_id
       AND names.team_id = team_actions.team_id
       AND names.name_form = regexp_extract(team_actions.description, 'SUB: (.+) FOR ', 1)
    ) AS player_in_id,
    description
FROM team_actions
WHERE action_type = 'Substitution';

-- Substitutions a team makes at one stoppage form a single lineup decision.
CREATE OR REPLACE VIEW lineup_changes AS
SELECT
    game_id, team_id, period, period_seconds_left,
    any_value(score_margin) AS score_margin,
    count(*) AS players_changed,
    list(player_out_name ORDER BY action_id) AS players_out,
    list(player_in_name ORDER BY action_id) AS players_in
FROM decision_events
WHERE event_type = 'substitution'
GROUP BY game_id, team_id, period, period_seconds_left;
