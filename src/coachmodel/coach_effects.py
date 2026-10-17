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
    columns = {"coach": "coach_name", "stint": "stint", "player": "person_id"}
    columns = {level: column for level, column in columns.items() if level in tau_grid}
    rng = np.random.default_rng(124)

    connection = duckdb.connect(str(args.database), read_only=True)
    frame = load_features(connection, task)
    teams = dict(connection.sql("SELECT DISTINCT home_team_id, home_team FROM games").fetchall())
    frame = frame[frame["coach_name"].notna()].reset_index(drop=True)
    frame["stint"] = frame["coach_name"] + " (" + frame["team_id"].map(teams) + ")"
    train, test = split_by_date(frame, args.test_from)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    pooled_model = build_model(task).fit(train, train[task.label])
    clip = lambda p: np.clip(p, 1e-6, 1 - 1e-6)  # noqa: E731 (keeps logit finite)
    train_offset = logit(clip(pooled_model.predict_proba(train)[:, 1]))
    test_offset = logit(clip(pooled_model.predict_proba(test)[:, 1]))
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
