"""The greedy baseline.

This is the comparison the whole project rests on. An optimizer with no
benchmark is an unfalsifiable claim: it produces a number, and nobody -- the
analyst included -- can say whether that number is good. The baseline turns the
result into a measurable improvement over the thing a competent team would
actually have done without any of this machinery.

Which is why the baseline is written to be *good*. It respects every constraint
the optimizer respects: budget, capacity, delivery radius, and the incumbent
network. Beating a deliberately crippled baseline proves nothing, and an
interviewer will go looking for exactly that weakness. The honest framing is
that greedy loses not because it is naive but because it is myopic, and myopia
is a structural limitation no amount of care in the implementation can fix.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.evaluate import NetworkResult, nearest_first_allocation


def greedy_network(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    max_new_stores: int | None = None,
    verbose: bool = False,
    value_weights: np.ndarray | None = None,
    running_cost: np.ndarray | None = None,
) -> NetworkResult:
    """Open stores one at a time, always taking the biggest immediate gain.

    The loop is the definition of greedy. At each step it asks, of every site
    still affordable within the remaining budget, which one would add the most
    demand served *right now* -- then commits to that site permanently and
    repeats. It stops when no affordable site would add anything, or when the
    cap on new stores is reached.

    The marginal gain is computed by actually re-running the dispatch rule with
    the candidate site open and measuring the difference. That is more expensive
    than estimating it from uncovered demand in range, but it is honest: it
    accounts for the site stealing volume from a neighbour that was already
    near capacity, which a naive estimate would double-count.

    What greedy cannot do is reconsider. Once a store is open it stays open, and
    the capital is gone. If opening a single large site consumes the budget that
    two smaller, better-placed sites would have used to serve more in total,
    greedy takes the large site and never learns otherwise -- it had no way to
    evaluate the pair, because it only ever evaluates one site at a time. This
    is the knapsack trap, and it is the specific failure the optimizer exists to
    avoid.

    value_weights and running_cost work exactly as they do in the optimizer,
    so the two are always judged against the same objective. With both set, a
    site's gain is the value of the extra orders it serves minus what it costs
    to run, and a site that would not cover its own running costs is never
    opened. Left as None, the behaviour is the original order-counting greedy.
    """
    start = time.perf_counter()
    n_loc, n_fac = distances.shape
    weights = np.ones(n_loc) if value_weights is None else np.asarray(value_weights, dtype=float)
    fixed = np.zeros(n_fac) if running_cost is None else np.asarray(running_cost, dtype=float)

    def value_of(alloc):
        """Weighted orders served by an allocation."""
        return float((alloc.sum(axis=1) * weights).sum())

    capacity = facilities["capacity"].to_numpy(dtype=float)
    capex = facilities["capex"].to_numpy(dtype=float)
    is_existing = facilities["is_existing"].to_numpy(dtype=bool)

    # The incumbent network is open before any decision is made.
    open_flags = is_existing.copy()
    assignment = nearest_first_allocation(demand, capacity, distances, mask, open_flags)
    served = assignment.sum()
    value = value_of(assignment)

    remaining_budget = float(budget)
    opened_order: list[str] = []

    if verbose:
        print(f"start: {int(is_existing.sum())} existing stores, "
              f"{served:,.0f}/{demand.sum():,.0f} orders served "
              f"({100 * served / demand.sum():.1f}%)")

    while True:
        if max_new_stores is not None and len(opened_order) >= max_new_stores:
            break

        best_gain = 0.0
        best_idx = -1
        best_assignment = None

        for j in range(len(facilities)):
            if open_flags[j] or capex[j] > remaining_budget:
                continue

            trial_flags = open_flags.copy()
            trial_flags[j] = True
            trial_assignment = nearest_first_allocation(
                demand, capacity, distances, mask, trial_flags
            )
            gain = value_of(trial_assignment) - value - fixed[j]

            if gain > best_gain + 1e-9:
                best_gain = gain
                best_idx = j
                best_assignment = trial_assignment

        if best_idx < 0:
            break

        open_flags[best_idx] = True
        assignment = best_assignment
        gained_orders = best_assignment.sum() - served
        served = best_assignment.sum()
        value = value_of(best_assignment)
        remaining_budget -= capex[best_idx]
        opened_order.append(str(facilities.loc[best_idx, "facility_key"]))

        if verbose:
            print(f"  open {facilities.loc[best_idx, 'facility_key']:>5}  "
                  f"capex Rs {capex[best_idx] / 1e7:.2f} cr  "
                  f"+{gained_orders:,.0f} orders  "
                  f"budget left Rs {remaining_budget / 1e7:.2f} cr")

    return NetworkResult(
        method="greedy",
        open_flags=open_flags,
        assignment=assignment,
        facilities=facilities,
        localities=localities,
        distances=distances,
        demand=demand,
        status="ok",
        solve_seconds=time.perf_counter() - start,
        notes={
            "opened_order": opened_order,
            "budget": budget,
            "budget_unused": remaining_budget,
        },
    )
