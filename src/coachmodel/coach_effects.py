"""First claim: does knowing the coach improve on the league-average (pooled) model?

Partially pooled effects are added to the pooled model's log-odds:
    log-odds = pooled log-odds + coach[c] + stint[c, team] (+ player[p] for substitutions)
- coach: the head coach's general tendency, wherever he coaches.
- stint: how his bench differs at one team (e.g. Monty Williams in Phoenix vs Detroit).
- player (substitution task only): how readily any coach takes this player out, given
  his fouls and minutes. Stars stay in; deep-bench players come out. Players change
  teams, so this separates coach style from who is on the roster.
Each level's effects come from a normal distribution centered at 0 with its own spread
(tau). A coach, stint, or player with little data stays near 0, and only consistent
evidence moves it away. The effects maximize
    log-likelihood - sum over levels of sum(effect^2) / (2 tau^2)
(the penalties are the logs of the normal priors), and the spreads are chosen by
cross-validation over training games. Comparing tau_stint with tau_coach answers "does
a coach's style travel with him?": a tau_stint near 0 means yes. Anything not seen in
training (a coach's new team, a rookie) gets 0 for that level.

This is a random-intercept model fit by penalized maximum likelihood (the most likely
effects under the hierarchical model), not a full Bayesian fit.

Usage:
    python -m coachmodel.coach_effects --task timeout --test-from 2025-03-01
"""

import argparse
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit

from coachmodel.baseline import TASKS, build_model, load_features, split_by_date
from coachmodel.build_db import DATABASE_PATH, REPO_ROOT

# Candidate spreads per level; 0.01 effectively turns a level off. For substitutions,
# coach and stint are fixed at 0.1 (what cross-validation chose for them without player
# effects) and only the player spread is searched: the full 3-level grid took over an
# hour on 4 million rows.
TAU_GRIDS = {
    "timeout": {"coach": [0.1, 0.2, 0.3, 0.5, 0.8], "stint": [0.01, 0.1, 0.2, 0.4]},
    "substitution": {"coach": [0.1], "stint": [0.1], "player": [0.01, 0.2, 0.4, 0.8]},
}
# Spreads chosen by cross-validation in the full-season test (train through 2024-25,
# test 2025-26); reused when fitting on every game for the scout report, where running
# the cross-validation again would take about an hour for substitutions.
CHOSEN_TAUS = {
    "timeout": {"coach": 0.2, "stint": 0.1},
    "substitution": {"coach": 0.1, "stint": 0.1, "player": 0.4},
}
# Season-to-season change in a coach's true effect (sd in log-odds), estimated by
# coachmodel.range_check on 2022-23 to 2025-26.
SEASON_DRIFT_SD = {"timeout": 0.105, "substitution": 0.176}
LEVEL_COLUMNS = {"coach": "coach_name", "stint": "stint", "player": "person_id"}
CROSS_VALIDATION_FOLDS = 5
BOOTSTRAP_SAMPLES = 2000


def fit_group_effects(offset: np.ndarray, labels: np.ndarray,
                      groups: list[tuple[np.ndarray, int, float]]) -> list[np.ndarray]:
    """Maximize the penalized log-likelihood. Each group is (row index into the group's
    effects, number of effects, prior spread tau). Returns one effect array per group."""
    sizes = [count for _, count, _ in groups]
    starts = np.cumsum([0] + sizes)

    def objective(parameters):
        effects = [parameters[starts[g]:starts[g + 1]] for g in range(len(groups))]
        log_odds = offset + sum(effect[index] for effect, (index, _, _) in zip(effects, groups))
        # log(1 + e^x) - y*x is the negative log-likelihood of one yes/no outcome.
        negative_log_likelihood = np.sum(np.logaddexp(0, log_odds) - labels * log_odds)
        penalty = sum(np.sum(effect ** 2) / (2 * tau ** 2) for effect, (_, _, tau) in zip(effects, groups))
        residual = expit(log_odds) - labels
        gradient = np.concatenate([
            np.bincount(index, residual, minlength=count) + effect / tau ** 2
            for effect, (index, count, tau) in zip(effects, groups)
        ])
        return negative_log_likelihood + penalty, gradient

    result = minimize(objective, np.zeros(starts[-1]), jac=True, method="L-BFGS-B")
    return [result.x[starts[g]:starts[g + 1]] for g in range(len(groups))]


def row_log_loss(labels: np.ndarray, log_odds: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, log_odds) - labels * log_odds


class GroupEffects:
    """Partially pooled effects for named levels (e.g. coach, stint, player), fit on
    training rows and applied to any rows; names unseen in training get 0."""

    def __init__(self, train: pd.DataFrame, columns: dict[str, str]):
        self.columns = columns  # level name -> column holding each row's member of it
        self.members = {level: sorted(train[column].unique()) for level, column in columns.items()}

    def fit(self, offset, labels, rows: pd.DataFrame, taus: dict[str, float]):
        groups = []
        for level, column in self.columns.items():
            number = {name: n for n, name in enumerate(self.members[level])}
            groups.append((rows[column].map(number).to_numpy(), len(self.members[level]), taus[level]))
        fitted = fit_group_effects(offset, labels, groups)
        self.effects = {level: dict(zip(self.members[level], values))
                        for level, values in zip(self.columns, fitted)}
        return self

    def effects_for(self, rows: pd.DataFrame, levels: list[str] | None = None) -> np.ndarray:
        levels = levels or list(self.columns)
        return sum(rows[self.columns[level]].map(self.effects[level]).fillna(0).to_numpy() for level in levels)


def effect_design(model: GroupEffects, rows: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, slice], np.ndarray]:
    """Indicator matrix Z (one column per coach, stint, and player effect) so that the
    fitted log-odds are offset + Z @ parameters, with each level's column range."""
    blocks, slices, parameters, start = [], {}, [], 0
    for level, column in model.columns.items():
        members = model.members[level]
        number = {name: n for n, name in enumerate(members)}
        index = rows[column].map(number).to_numpy()
        known = ~np.isnan(index.astype(float))
        blocks.append(sparse.csr_matrix(
            (np.ones(known.sum()), (np.flatnonzero(known), index[known].astype(int))),
            shape=(len(rows), len(members))))
        slices[level] = slice(start, start + len(members))
        parameters.append([model.effects[level][name] for name in members])
        start += len(members)
    return sparse.hstack(blocks).tocsr(), slices, np.concatenate(parameters)


def coach_standard_errors(model: GroupEffects, rows: pd.DataFrame, offset: np.ndarray, labels: np.ndarray,
                          taus: dict[str, float], method: str) -> pd.Series:
    """Standard error of each coach effect, three ways (all Laplace approximations at the
    fitted peak, where H is the curvature of the penalized log-likelihood):
        conditional  1 / sqrt(H[c, c]): treats his team and player effects as known
        joint        sqrt of the diagonal of H^-1: also counts the uncertainty about how
                     much belongs to him rather than his team or players
        clustered    joint, plus the extra variance from decisions within a game moving
                     together: H^-1 + H^-1 (B - A) H^-1, where A is the information if
                     decisions were independent and B is the same measured game by game
                     (B = A when they are independent, leaving the joint answer)
    """
    design, slices, parameters = effect_design(model, rows)
    labels, offset = np.asarray(labels, dtype=float), np.asarray(offset, dtype=float)
    probability = expit(offset + design @ parameters)
    prior = np.concatenate([np.full(model_slice.stop - model_slice.start, 1 / taus[level] ** 2)
                            for level, model_slice in slices.items()])
    independent_information = (design.T @ sparse.diags(probability * (1 - probability)) @ design).toarray()
    curvature = independent_information + np.diag(prior)
    coaches = slices["coach"]
    if method == "conditional":
        variance = 1 / np.diag(curvature)[coaches]
    else:
        inverse = np.linalg.inv(curvature)
        if method == "joint":
            variance = np.diag(inverse)[coaches]
        elif method == "clustered":
            games = pd.factorize(rows["game_id"])[0]
            game_matrix = sparse.csr_matrix((np.ones(len(rows)), (games, np.arange(len(rows)))))
            scores = (game_matrix @ design.multiply((labels - probability)[:, None])).toarray()
            by_game_information = scores.T @ scores
            extra = inverse @ (by_game_information - independent_information) @ inverse
            variance = np.diag(inverse + extra)[coaches]
        else:
            raise ValueError(f"unknown method {method}")
    return pd.Series(np.sqrt(variance), index=model.members["coach"])


def choose_taus(offset, labels, rows: pd.DataFrame, columns, tau_grid, rng) -> dict[str, float]:
    """Pick the prior spreads that predict held-out games best (K-fold by game)."""
    games = rows["game_id"].unique()
    fold_of_game = dict(zip(games, rng.integers(0, CROSS_VALIDATION_FOLDS, len(games))))
    folds = rows["game_id"].map(fold_of_game).to_numpy()
    losses = {}
    for values in product(*tau_grid.values()):
        taus = dict(zip(tau_grid, values))
        total = 0.0
        for fold in range(CROSS_VALIDATION_FOLDS):
            held_out = folds == fold
            model = GroupEffects(rows[~held_out], columns).fit(
                offset[~held_out], labels[~held_out], rows[~held_out], taus)
            total += row_log_loss(labels[held_out], offset[held_out] + model.effects_for(rows[held_out])).sum()
        losses[values] = total
    return dict(zip(tau_grid, min(losses, key=losses.get)))


def load_task_rows(connection: duckdb.DuckDBPyConnection, task) -> pd.DataFrame:
    """The task's feature rows that have a head coach, with a coach-team stint label."""
    frame = load_features(connection, task)
    teams = dict(connection.sql("SELECT DISTINCT home_team_id, home_team FROM games").fetchall())
    frame = frame[frame["coach_name"].notna()].reset_index(drop=True)
    frame["stint"] = frame["coach_name"] + " (" + frame["team_id"].map(teams) + ")"
    return frame


def pooled_log_odds(model, rows: pd.DataFrame) -> np.ndarray:
    """The pooled model's prediction as log-odds (clipped so the logit stays finite)."""
    return logit(np.clip(model.predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6))


def coach_rate_ratios(model: GroupEffects, rows: pd.DataFrame, offset: np.ndarray,
                      uncertainty: pd.Series, label: str) -> pd.DataFrame:
    """Per coach: his predicted decisions divided by what an average coach would make in
    his exact situations, with a 95% range from shifting his coach effect by +-1.96 times
    his uncertainty (a standard deviation in log-odds, one per coach)."""
    total = offset + model.effects_for(rows)
    coach_effect = rows["coach_name"].map(model.effects["coach"]).to_numpy()
    without_coach = total - coach_effect
    coaches = rows["coach_name"].to_numpy()
    spread = rows["coach_name"].map(uncertainty).to_numpy()

    def ratio(shift):
        predicted = pd.Series(expit(without_coach + coach_effect + shift)).groupby(coaches).sum()
        average = pd.Series(expit(without_coach)).groupby(coaches).sum()
        return predicted / average

    return pd.DataFrame({
        "ratio": ratio(0.0),
        "low": ratio(-1.96 * spread),
        "high": ratio(1.96 * spread),
        "decisions": pd.Series(rows[label].to_numpy()).groupby(coaches).sum(),
    })


def tendency_and_season_uncertainty(model: GroupEffects, rows: pd.DataFrame, offset: np.ndarray,
                                    task_name: str) -> tuple[pd.Series, pd.Series]:
    """Two standard deviations per coach (log-odds):
        tendency  how precisely his long-run tendency is known: joint, game-clustered
                  standard error. Decides whether he is clearly different from average.
        season    where one season of his can land: the tendency's uncertainty plus real
                  season-to-season change. Passed the rolling-origin check in
                  coachmodel.range_check (about 95% of next seasons inside the 95% range)."""
    labels = rows[TASKS[task_name].label].to_numpy()
    tendency = coach_standard_errors(model, rows, offset, labels, CHOSEN_TAUS[task_name], "clustered")
    return tendency, np.sqrt(tendency ** 2 + SEASON_DRIFT_SD[task_name] ** 2)


def fit_on_all_games(connection: duckdb.DuckDBPyConnection, task_name: str):
    """Pooled model plus group effects fit on every game, with the spreads chosen in the
    full-season test. Returns (group effects model, rows, pooled log-odds)."""
    task = TASKS[task_name]
    taus = CHOSEN_TAUS[task_name]
    rows = load_task_rows(connection, task)
    offset = pooled_log_odds(build_model(task).fit(rows, rows[task.label]), rows)
    columns = {level: LEVEL_COLUMNS[level] for level in taus}
    model = GroupEffects(rows, columns).fit(offset, rows[task.label].to_numpy(), rows, taus)
    return model, rows, offset


def bootstrap_improvement(game_ids, pooled_losses, coach_losses, rng) -> np.ndarray:
    """Relative log-loss improvement, recomputed on games resampled with replacement.
    Resampling whole games keeps each game's correlated decisions together."""
    per_game = pd.DataFrame({"game": game_ids, "pooled": pooled_losses, "coach": coach_losses}).groupby("game").sum()
    pooled, coach = per_game["pooled"].to_numpy(), per_game["coach"].to_numpy()
    samples = rng.integers(0, len(per_game), (BOOTSTRAP_SAMPLES, len(per_game)))
    return 1 - coach[samples].sum(axis=1) / pooled[samples].sum(axis=1)


def per_coach_count_error(coach_names, labels, predicted) -> float:
    """Median over coaches of |actual decisions - predicted decisions| / actual: how far
    off a forecast of each bench's total would be."""
    totals = pd.DataFrame({"coach": coach_names, "actual": labels, "predicted": predicted}).groupby("coach").sum()
    totals = totals[totals["actual"] > 0]
    return float(np.median(np.abs(totals["actual"] - totals["predicted"]) / totals["actual"]))


def print_coach_table(model: GroupEffects, train: pd.DataFrame, label: str, without_players: GroupEffects | None):
    table = pd.DataFrame({"coach": model.members["coach"]})
    table["train_events"] = table["coach"].map(train.groupby("coach_name")[label].sum())
    table["coach_multiplier"] = np.exp(table["coach"].map(model.effects["coach"]))
    columns = ["train_events", "coach_multiplier"]
    if without_players is not None:
        table["without_players"] = np.exp(table["coach"].map(without_players.effects["coach"]))
        columns.append("without_players")
    table = table.sort_values("coach_multiplier").set_index("coach")
    print("\ncoach tendency (odds multiplier vs. the average coach in the same situation):")
    print(table[columns].head(5).round(2).to_string())
    print("   ...")
    print(table[columns].tail(5).round(2).to_string(header=False))
    if without_players is not None:
        moved = (np.log(table["coach_multiplier"]) - np.log(table["without_players"])).abs()
        print("coaches whose effect changed most once players are accounted for:")
        print(table.loc[moved.sort_values().index[-5:], ["without_players", "coach_multiplier"]].round(2).to_string())


def print_multi_team_coaches(model: GroupEffects):
    stints = pd.DataFrame({"stint": model.members["stint"]})
    stints["coach"] = stints["stint"].str.replace(r" \(.*\)$", "", regex=True)
    stints = stints[stints.groupby("coach")["stint"].transform("size") > 1]
    if stints.empty:
        return
    stints["total_multiplier"] = np.exp(
        stints["coach"].map(model.effects["coach"]) + stints["stint"].map(model.effects["stint"]))
    print("\ncoaches with more than one team in training (coach + stint, as an odds multiplier):")
    print(stints.set_index("stint")["total_multiplier"].round(2).to_string())


def print_player_extremes(model: GroupEffects, train: pd.DataFrame, connection: duckdb.DuckDBPyConnection):
    names = dict(connection.sql(f"""
        SELECT PERSON_ID, DISPLAY_FIRST_LAST FROM read_json('{REPO_ROOT / "data" / "raw" / "players.json"}')
    """).fetchall())
    stoppages = train.groupby("person_id").size()
    players = pd.DataFrame({"person_id": model.members["player"]})
    players["player"] = players["person_id"].map(names)
    players["stoppages_on_court"] = players["person_id"].map(stoppages)
    players["multiplier"] = np.exp(players["person_id"].map(model.effects["player"]))
    players = players[players["stoppages_on_court"] >= 1000].sort_values("multiplier").set_index("player")
    print("\nplayers (1,000+ stoppages on court), odds of being subbed out vs. an average player:")
    print(players[["stoppages_on_court", "multiplier"]].head(5).round(2).to_string())
    print("   ...")
    print(players[["stoppages_on_court", "multiplier"]].tail(5).round(2).to_string(header=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--test-from", required=True, help="hold out games on or after this date (YYYY-MM-DD)")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    task = TASKS[args.task]
    tau_grid = TAU_GRIDS[args.task]
    columns = {level: LEVEL_COLUMNS[level] for level in tau_grid}
    rng = np.random.default_rng(124)

    connection = duckdb.connect(str(args.database), read_only=True)
    train, test = split_by_date(load_task_rows(connection, task), args.test_from)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    pooled_model = build_model(task).fit(train, train[task.label])
    train_offset = pooled_log_odds(pooled_model, train)
    test_offset = pooled_log_odds(pooled_model, test)
    train_labels, test_labels = train[task.label].to_numpy(), test[task.label].to_numpy()

    taus = choose_taus(train_offset, train_labels, train, columns, tau_grid, rng)
    model = GroupEffects(train, columns).fit(train_offset, train_labels, train, taus)
    test_effects = model.effects_for(test)

    pooled_losses = row_log_loss(test_labels, test_offset)
    model_losses = row_log_loss(test_labels, test_offset + test_effects)
    improvement = 1 - model_losses.sum() / pooled_losses.sum()
    low, high = np.percentile(bootstrap_improvement(test["game_id"].to_numpy(), pooled_losses, model_losses, rng), [2.5, 97.5])
    pooled_error = per_coach_count_error(test["coach_name"], test_labels, expit(test_offset))
    model_error = per_coach_count_error(test["coach_name"], test_labels, expit(test_offset + test_effects))

    print(f"task: {args.task}   " + "   ".join(f"{level} effects: {len(model.members[level])}" for level in columns))
    print("chosen spreads: " + ", ".join(f"tau_{level} {tau}" for level, tau in taus.items()))
    print(f"test log loss, pooled model:          {pooled_losses.mean():.4f}")
    print(f"test log loss, with group effects:    {model_losses.mean():.4f}")
    print(f"improvement: {improvement:.2%}   (95% bootstrap range over games: {low:.2%} to {high:.2%})")
    print(f"median error in each coach's total {task.label} count: pooled {pooled_error:.0%}, "
          f"with group effects {model_error:.0%}")

    without_players = None
    if "player" in columns:
        # Same coach and stint spreads, players left out: how much of what looked like
        # coach style was really the roster?
        coach_columns = {level: column for level, column in columns.items() if level != "player"}
        without_players = GroupEffects(train, coach_columns).fit(train_offset, train_labels, train, taus)
        only_coach = row_log_loss(test_labels, test_offset + without_players.effects_for(test))
        print(f"improvement without player effects: {1 - only_coach.sum() / pooled_losses.sum():.2%}")
    print_coach_table(model, train, task.label, without_players)
    print_multi_team_coaches(model)
    if "player" in columns:
        print_player_extremes(model, train, connection)
    connection.close()


if __name__ == "__main__":
    main()
