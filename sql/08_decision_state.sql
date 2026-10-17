-- Game state at each bench decision: the team's state at that play (07_team_state),
-- plus, for substitutions, the outgoing player's foul trouble and playing time, so
-- fatigue and fouls are not mistaken for coaching style.
CREATE OR REPLACE TABLE decision_state AS
SELECT
    state.*,
    d.event_type,
    CASE WHEN d.event_type = 'substitution'
         THEN coalesce(player_fouls.player_fouls, 0) END AS player_out_fouls,
    round(player_time.seconds_played / 60, 2) AS player_out_minutes,
    round(player_time.stint_seconds / 60, 2) AS player_out_stint_minutes
FROM decision_events AS d
JOIN team_state AS state USING (game_id, action_id, team_id)
ASOF LEFT JOIN personal_fouls AS player_fouls
    ON player_fouls.game_id = d.game_id AND player_fouls.person_id = d.player_out_id
   AND d.action_id >= player_fouls.action_id
LEFT JOIN LATERAL (
    -- Time on court before this decision: each earlier stint counts up to the
    -- decision's clock, so a stint still running is cut off at "now".
    SELECT
        sum(least(stints.end_seconds, d.game_seconds_elapsed) - stints.start_seconds) AS seconds_played,
        max(CASE WHEN stints.end_action_id = d.action_id
                 THEN stints.end_seconds - stints.start_seconds END) AS stint_seconds
    FROM stints
    WHERE stints.game_id = d.game_id AND stints.person_id = d.player_out_id
      AND stints.start_seconds < d.game_seconds_elapsed
) AS player_time ON true;  -- timeouts have no outgoing player, so these stay NULL
