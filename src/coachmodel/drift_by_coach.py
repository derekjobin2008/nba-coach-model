"""Does season-to-season drift differ by coach? A test on the range check's coach-seasons.

The page gives every coach the same drift (SEASON_DRIFT_SD), and its ranges hold about 95%
of next seasons on average. If some coaches really are steadier than others, a coach whose
season landed far outside his range (large |z|) should tend to land far outside it again
the next season.

    statistic  correlation of z^2 in one test season with z^2 in the coach's next test
               season, over every consecutive pair
    null       every coach has the same drift. The signed z's of a coach's consecutive
               seasons can still be correlated (a season effect that carries over, and
               training data shared between checks), and that alone makes the z^2's
               correlated a little. So the null is simulated: standard bivariate normal
               pairs with the observed signed correlation, as many pairs as observed.
    verdict    observed z^2 correlation above the null's 95th percentile means drift
               differs by coach.
    power      how often the test would notice if half the coaches had three times the
               other half's variance (z spread 0.71 vs 1.22), so "no measurable
               difference" can be read as either "same drift" or "too few seasons to tell".

Reads reports/range_check_pairs_<task>.csv, written by coachmodel.range_check.

Usage:
    python -m coachmodel.drift_by_coach --task timeout
"""

import argparse

import numpy as np
import pandas as pd

from coachmodel.range_check import PAIRS_PATH

SIMULATIONS = 20000
MIN_SEASONS_TO_LIST = 3


def consecutive_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per coach and pair of back-to-back test seasons, with z then and next."""
    table = pairs.assign(start=pairs["season"].str[:4].astype(int))[["coach", "start", "z"]]
    return table.merge(table.assign(start=table["start"] - 1), on=["coach", "start"], suffixes=("", "_next"))


def squared_z_correlation(z: np.ndarray, z_next: np.ndarray) -> float:
    return float(np.corrcoef(z ** 2, z_next ** 2)[0, 1])


def simulated_squared_correlations(signed_correlation: float, count: int, rng, variance_multipliers=(1.0, 1.0),
                                   simulations=SIMULATIONS) -> np.ndarray:
    """z^2 correlations over `count` back-to-back pairs of simulated coaches. Each coach
    keeps one variance multiplier for both seasons, drawn evenly from variance_multipliers;
    (1, 1) is the same-drift null."""
    covariance = [[1, signed_correlation], [signed_correlation, 1]]
    draws = rng.multivariate_normal([0, 0], covariance, size=(simulations, count))
    scale = np.sqrt(rng.choice(variance_multipliers, size=(simulations, count, 1)))
    first, second = (draws * scale)[..., 0] ** 2, (draws * scale)[..., 1] ** 2
    first, second = first - first.mean(1, keepdims=True), second - second.mean(1, keepdims=True)
    return (first * second).sum(1) / np.sqrt((first ** 2).sum(1) * (second ** 2).sum(1))


def drift_test(pairs: pd.DataFrame, rng) -> dict:
    back_to_back = consecutive_pairs(pairs)
    z, z_next = back_to_back["z"].to_numpy(), back_to_back["z_next"].to_numpy()
    signed = float(np.corrcoef(z, z_next)[0, 1])
    observed = squared_z_correlation(z, z_next)
    null = simulated_squared_correlations(signed, len(z), rng)
    uneven = simulated_squared_correlations(signed, len(z), rng, variance_multipliers=(0.5, 1.5))
    return {"power": float(np.mean(uneven > np.percentile(null, 95))), "pairs": len(z), "signed_correlation": signed, "squared_correlation": observed,
            "null_95th": float(np.percentile(null, 95)), "p_value": float(np.mean(null >= observed)),
            "differs": bool(observed > np.percentile(null, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["timeout", "substitution"], required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(str(PAIRS_PATH).format(task=args.task))
    result = drift_test(pairs, np.random.default_rng(124))

    print(f"{args.task}: {len(pairs)} coach-seasons, {result['pairs']} back-to-back pairs")
    print(f"  signed z, one season to the next:   {result['signed_correlation']:+.2f}")
    print(f"  z^2, one season to the next:        {result['squared_correlation']:+.2f} "
          f"(same-drift null: 95th percentile {result['null_95th']:+.2f}, p = {result['p_value']:.3f})")
    print("  drift differs by coach" if result["differs"] else
          "  no measurable difference: one drift for every coach fits")
    print(f"  power: {result['power']:.0%} chance of noticing if half the coaches had 3x the other half's variance")

    per_coach = pairs.groupby("coach")["z"].agg(seasons="size", mean_z_squared=lambda z: np.mean(z ** 2))
    listed = per_coach[per_coach["seasons"] >= MIN_SEASONS_TO_LIST].sort_values("mean_z_squared")
    print(f"\n  mean z^2 by coach ({MIN_SEASONS_TO_LIST}+ seasons; about 1 if his range is the right width):")
    for name, row in pd.concat([listed.head(3), listed.tail(3)]).iterrows():
        print(f"    {name:22s} {row['mean_z_squared']:5.2f} over {int(row['seasons'])} seasons")


if __name__ == "__main__":
    main()
