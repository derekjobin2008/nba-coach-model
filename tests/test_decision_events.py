"""Checks the SQL pipeline against two hand-checkable playoff games:
2025 ECF Game 6 (NYK @ IND, 108-125) and 2024 West semis Game 7 (MIN @ DEN, 98-90),
plus one 2024-25 regular-season NYK-IND game that gives the name matcher a season roster."""

PACERS_KNICKS_GAME_6 = "0042400306"
WOLVES_NUGGETS_GAME_7 = "0042300237"
REGULAR_SEASON_KNICKS_PACERS = "0022400081"


def test_carried_forward_score_matches_final_score(database):
    mismatches = database.sql("""
        SELECT game_id
        FROM games JOIN pbp USING (game_id)
        GROUP BY game_id, home_points, away_points
        HAVING max(score_home) <> home_points OR max(score_away) <> away_points
    """).fetchall()
    assert mismatches == []


def test_timeout_counts_per_team(database):
    counts = dict(database.sql(f"""
        SELECT CASE WHEN is_home THEN home_team ELSE away_team END, count(*)
        FROM decision_events JOIN games USING (game_id)
        WHERE event_type = 'timeout' AND game_id <> '{REGULAR_SEASON_KNICKS_PACERS}'
        GROUP BY ALL
    """).fetchall())
    assert counts == {"IND": 5, "NYK": 5, "DEN": 6, "MIN": 5}


def test_timeouts_used_counts_up_by_one(database):
    gaps = database.sql("""
        SELECT game_id, team_id
        FROM decision_events
        WHERE event_type = 'timeout'
        GROUP BY game_id, team_id
        HAVING list(team_timeouts_used ORDER BY action_id) <> range(1, count(*) + 1)
    """).fetchall()
    assert gaps == []


def test_first_quarter_mandatory_timeouts_flagged(database):
    flagged = database.sql(f"""
        SELECT period, period_seconds_left
        FROM decision_events
        WHERE game_id = '{WOLVES_NUGGETS_GAME_7}' AND likely_mandatory AND period = 1
        ORDER BY action_id
    """).fetchall()
    assert flagged == [(1, 379.0), (1, 120.0)]  # DEN at 6:19, MIN at 2:00


def test_every_substitution_names_both_players(database):
    incomplete = database.sql("""
        SELECT count(*) FROM decision_events
        WHERE event_type = 'substitution'
          AND (player_out_id IS NULL OR coalesce(player_in_name, '') = '')
    """).fetchone()[0]
    assert incomplete == 0


def test_pacers_substitution_count(database):
    pacers_subs = database.sql(f"""
        SELECT count(*) FROM decision_events JOIN games USING (game_id)
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND event_type = 'substitution'
          AND is_home
    """).fetchone()[0]
    assert pacers_subs == 38


def test_garbage_time_entrants_resolved_from_season_roster(database):
    # Five players checked in late in the Game 6 blowout and never touched the
    # play-by-play again. Four of them recorded stats in the regular-season fixture
    # game, so the season-roster tier resolves them; Achiuwa did not, so he stays NULL.
    # Likewise Jacob Toppin in the regular-season game: his stats are in games the
    # fixtures do not include.
    unresolved = database.sql("""
        SELECT game_id, player_in_name FROM decision_events
        WHERE event_type = 'substitution' AND player_in_id IS NULL
        ORDER BY game_id
    """).fetchall()
    assert unresolved == [(REGULAR_SEASON_KNICKS_PACERS, "Toppin"), (PACERS_KNICKS_GAME_6, "Achiuwa")]
    furphy_id = database.sql(f"""
        SELECT player_in_id FROM decision_events
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND player_in_name = 'Furphy'
    """).fetchone()[0]
    assert furphy_id == 1642277  # Johnny Furphy


def test_incoming_player_id_matches_their_later_substitution(database):
    # Siakam is subbed in at 1:18 of Q1 in Game 6 and out again later; the id we
    # resolve for his entry must be the id the feed attaches to his exit.
    siakam_ids = database.sql(f"""
        SELECT DISTINCT player_in_id FROM decision_events
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND player_in_name = 'Siakam'
        UNION
        SELECT DISTINCT player_out_id FROM decision_events
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND player_out_name = 'Siakam'
    """).fetchall()
    assert siakam_ids == [(1627783,)]


def test_one_state_row_per_decision(database):
    decisions, states, distinct_states = database.sql("""
        SELECT (SELECT count(*) FROM decision_events),
               count(*), count(DISTINCT (game_id, action_id))
        FROM decision_state
    """).fetchone()
    assert decisions == states == distinct_states


def test_counted_team_fouls_agree_with_feed_numbers(database):
    # Team-foul descriptions show "(P#.T#)" up to the 4th foul, then "(P#.PN)". When a
    # decision's most recent team foul shows T#, our running count must equal it.
    mismatches = database.sql(r"""
        WITH team_fouls AS (
            SELECT game_id, team_id, period, action_id,
                   CAST(nullif(regexp_extract(description, '\(P\d+\.T(\d+)\)', 1), '') AS INTEGER)
                       AS feed_count
            FROM pbp
            WHERE action_type = 'Foul' AND regexp_matches(description, '\(P\d+\.(T\d+|PN)\)')
              AND sub_type NOT ILIKE '%technical%'
        )
        SELECT count(*)
        FROM decision_state AS state
        ASOF JOIN team_fouls AS latest_foul
            ON latest_foul.game_id = state.game_id AND latest_foul.team_id = state.team_id
           AND latest_foul.period = state.period AND state.action_id >= latest_foul.action_id
        WHERE latest_foul.feed_count IS NOT NULL
          AND state.team_fouls_in_period <> latest_foul.feed_count
    """).fetchone()[0]
    assert mismatches == 0


def test_pacers_second_quarter_timeout_state(database):
    # Pacers timeout at 3:02 of Q2, Game 6: up 5, right after Siakam's foul, their
    # 7th of the quarter; third timeout of the game, so 5 left before it.
    state = database.sql(f"""
        SELECT score_margin, team_fouls_in_period, team_in_penalty, timeouts_remaining
        FROM decision_state
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND event_type = 'timeout'
          AND period = 2 AND period_seconds_left = 182
    """).fetchall()
    assert state == [(5, 7, True, 5)]


def test_five_players_on_court_for_whole_game(database):
    # Stints per team must add up to 5 players x game length. The exceptions are the
    # two Knicks games with an unresolved entrant (Toppin, Achiuwa; see above).
    off_by_seconds = database.sql("""
        WITH game_lengths AS (
            SELECT game_id, max(CASE WHEN period <= 4 THEN period * 720
                                     ELSE 2880 + (period - 4) * 300 END) AS seconds
            FROM pbp GROUP BY game_id
        )
        SELECT stints.game_id, stints.team_id,
               round(sum(end_seconds - start_seconds) - 5 * any_value(game_lengths.seconds)) AS off_by
        FROM stints JOIN game_lengths USING (game_id)
        GROUP BY ALL
        HAVING abs(off_by) >= 1
        ORDER BY stints.game_id
    """).fetchall()
    knicks = 1610612752
    assert [(game, team) for game, team, _ in off_by_seconds] == [
        (REGULAR_SEASON_KNICKS_PACERS, knicks),
        (PACERS_KNICKS_GAME_6, knicks),
    ]


def test_outgoing_player_minutes_add_up(database):
    # Haliburton's last exit in Game 6 (0:47 of Q4): 36.52 minutes so far, the last
    # 8.41 of them in the stint that is ending.
    minutes = database.sql(f"""
        SELECT player_out_minutes, player_out_stint_minutes
        FROM decision_state JOIN decision_events USING (game_id, action_id)
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND player_out_name = 'Haliburton'
        ORDER BY action_id DESC LIMIT 1
    """).fetchone()
    assert minutes == (36.52, 8.41)


def test_every_coach_timeout_on_a_stoppage_is_labeled(database):
    assigned, labeled = database.sql("""
        SELECT
            (SELECT count(DISTINCT (b.game_id, b.stoppage_action_id, b.team_id))
             FROM bench_action_stoppages AS b JOIN decision_events AS d USING (game_id, action_id)
             WHERE b.event_type = 'timeout' AND NOT d.likely_mandatory
               AND b.stoppage_action_id IS NOT NULL),
            (SELECT count(*) FROM stoppages WHERE called_timeout)
    """).fetchone()
    assert assigned == labeled > 0


def test_every_stoppage_starts_with_a_dead_ball_play(database):
    # A stoppage made only of bench actions would exist because the coach acted,
    # leaking the label into the rows (this happened before the 4-second rule).
    not_dead_ball = database.sql("""
        SELECT count(*) FROM stoppages AS s
        LEFT JOIN game_plays AS g
          ON g.game_id = s.game_id AND g.action_id = s.stoppage_action_id AND g.is_dead_ball
        WHERE g.action_id IS NULL
    """).fetchone()[0]
    assert not_dead_ball == 0


def test_hand_checked_wolves_nuggets_stoppages(database):
    # Derek confirmed from the broadcast: the Q1 6:19 Nuggets timeout was a TV
    # (mandatory) timeout, and at Q4 3:05 both coaches called timeouts back to back.
    # Those Q4 timeouts answer Edwards' three at 3:07: after a made basket the clock
    # runs on, so they belong to that stoppage.
    rows = database.sql(f"""
        SELECT period, period_seconds_left, is_home, called_timeout, has_mandatory_timeout
        FROM stoppages
        WHERE game_id = '{WOLVES_NUGGETS_GAME_7}'
          AND ((period = 1 AND period_seconds_left = 379) OR (period = 4 AND period_seconds_left = 187))
        ORDER BY period, is_home
    """).fetchall()
    assert rows == [
        (1, 379.0, False, False, True),  # Wolves at the TV timeout
        (1, 379.0, True, False, True),  # Nuggets: TV timeout, not a coach call
        (4, 187.0, False, True, False),  # Wolves call one...
        (4, 187.0, True, True, False),  # ...and so do the Nuggets
    ]


def test_live_ball_steals_are_not_stoppages(database):
    # Bridges' bad pass at 10:32 of Q1 in Game 6 was stolen by Nembhard: play went on.
    stoppages_at_steal = database.sql(f"""
        SELECT count(*) FROM stoppages
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND period = 1 AND period_seconds_left = 632
    """).fetchone()[0]
    assert stoppages_at_steal == 0


def test_five_players_on_court_at_every_wolves_nuggets_stoppage(database):
    lineup_sizes = database.sql(f"""
        SELECT DISTINCT count(*) FROM stoppage_players
        WHERE game_id = '{WOLVES_NUGGETS_GAME_7}'
        GROUP BY stoppage_action_id, team_id
    """).fetchall()
    assert lineup_sizes == [(5,)]


def test_wolves_nuggets_subs_are_labeled_player_rows(database):
    # All but one: Porter Jr. for Braun at 1:41 of Q4 comes 6 s after a jump-ball tip
    # with no dead ball logged in between, so the feed gives it no stoppage.
    subs, labeled = database.sql(f"""
        SELECT
            (SELECT count(*) FROM decision_events
             WHERE game_id = '{WOLVES_NUGGETS_GAME_7}' AND event_type = 'substitution'),
            (SELECT count(*) FROM stoppage_players
             WHERE game_id = '{WOLVES_NUGGETS_GAME_7}' AND subbed_out)
    """).fetchone()
    assert (subs, labeled) == (35, 34)


def test_haliburton_final_exit_row(database):
    # His last exit (0:47 of Q4, Game 6) matches the outgoing-player features in
    # decision_state: same minutes, same stint length, 2 fouls.
    row = database.sql(f"""
        SELECT minutes_played, stint_minutes, player_fouls, subbed_out
        FROM stoppage_players
        WHERE game_id = '{PACERS_KNICKS_GAME_6}' AND person_id = 1630169 AND subbed_out
        ORDER BY stoppage_action_id DESC LIMIT 1
    """).fetchone()
    assert row == (36.52, 8.41, 2, True)


def test_head_coach_on_each_bench(database):
    coaches = database.sql(f"""
        SELECT game_id, team, coach_name FROM game_coaches
        WHERE game_id IN ('{PACERS_KNICKS_GAME_6}', '{WOLVES_NUGGETS_GAME_7}')
        ORDER BY game_id, team
    """).fetchall()
    assert coaches == [
        (WOLVES_NUGGETS_GAME_7, "DEN", "Michael Malone"),
        (WOLVES_NUGGETS_GAME_7, "MIN", "Chris Finch"),
        (PACERS_KNICKS_GAME_6, "IND", "Rick Carlisle"),
        (PACERS_KNICKS_GAME_6, "NYK", "Tom Thibodeau"),
    ]
