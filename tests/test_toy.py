"""Tests on a toy network small enough to solve by hand.

The value of a hand-checkable case is that it tests the *formulation*, not just
that the code runs. A model can execute cleanly, return "Optimal", and still be
solving the wrong problem -- a sign flipped in the objective, a constraint
written with the wrong inequality. The only defence is a case whose right answer
you worked out on paper first.

The toy here is deliberately the same example used to explain why greedy loses,
so the test and the explanation cannot drift apart: if the story stops being
true, the suite fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline import greedy_network
from src.geo import build_facilities, distance_matrix, feasible_mask
from src.optimizer import optimize_network

# --- the toy network -------------------------------------------------------
#
# Three localities demanding 1,200 orders/day each, 3,600 in total. Three
# candidate sites, all close enough to serve every locality, so geometry is
# neutral and the only thing being tested is the capital allocation:
#
#     site   capex      capacity
#     A      Rs 3.0 cr    2,000
#     B      Rs 2.5 cr    1,800
#     C      Rs 2.5 cr    1,700
#
# Budget is Rs 5 cr.
#
#   Greedy opens A first, because A alone covers more than B or C alone. That
#   leaves Rs 2 cr, which buys nothing, so greedy stops at 2,000 orders served
#   with Rs 2 cr stranded.
#
#   The optimum is B + C: exactly Rs 5 cr, combined capacity 3,500. Seventy-five
#   percent more demand served for identical capital.
#
# Greedy cannot find it because it only ever evaluates one site at a time and
# commits irrevocably. This is the knapsack trap.

TOY_BUDGET = 5.0e7
EXPECTED_GREEDY_SERVED = 2000.0
EXPECTED_OPTIMAL_SERVED = 3500.0


def build_toy():
    localities = pd.DataFrame(
        {
            "locality_id": [1, 2, 3],
            "locality_name": ["L1", "L2", "L3"],
            "latitude": [22.7000, 22.7050, 22.7100],
            "longitude": [75.8500, 75.8550, 75.8600],
        }
    )

    sites = pd.DataFrame(
        {
            "site_id": [1, 2, 3],
            "locality_id": [2, 2, 2],
            "latitude": [22.7050, 22.7050, 22.7050],
            "longitude": [75.8550, 75.8550, 75.8550],
            "estimated_rent_monthly": [100_000.0, 100_000.0, 100_000.0],
            "max_capacity_orders_per_day": [2000, 1800, 1700],
            "setup_capex": [3.0e7, 2.5e7, 2.5e7],
        }
    )

    # No incumbent network: this toy is about the capital decision alone.
    stores = pd.DataFrame(
        {
            "store_id": pd.Series(dtype="int64"),
            "locality_id": pd.Series(dtype="int64"),
            "latitude": pd.Series(dtype="float64"),
            "longitude": pd.Series(dtype="float64"),
            "max_capacity_orders_per_day": pd.Series(dtype="int64"),
            "opened_on": pd.Series(dtype="object"),
        }
    )

    facilities = build_facilities(sites, stores)
    distances = distance_matrix(localities, facilities)
    mask = feasible_mask(distances, radius_km=2.5)
    demand = np.array([1200.0, 1200.0, 1200.0])

    return localities, facilities, distances, mask, demand


def test_toy_geometry_is_neutral():
    """Every site can reach every locality, so the test isolates capital."""
    _, _, _, mask, _ = build_toy()
    assert mask.all(), "toy sites should all be in range of all localities"


def test_greedy_falls_into_the_knapsack_trap():
    localities, facilities, distances, mask, demand = build_toy()
    result = greedy_network(localities, facilities, distances, mask, demand,
                            budget=TOY_BUDGET)

    assert result.demand_served == pytest.approx(EXPECTED_GREEDY_SERVED)
    assert result.n_new_stores == 1
    # It took the expensive high-capacity site and stranded Rs 2 cr.
    assert result.capex_used == pytest.approx(3.0e7)
    assert result.notes["budget_unused"] == pytest.approx(2.0e7)


def test_milp_finds_the_combination_greedy_cannot():
    localities, facilities, distances, mask, demand = build_toy()
    result = optimize_network(localities, facilities, distances, mask, demand,
                              budget=TOY_BUDGET,
                              lambda_capex=0.0, lambda_distance=0.0)

    assert result.status == "Optimal"
    assert result.demand_served == pytest.approx(EXPECTED_OPTIMAL_SERVED)
    assert result.n_new_stores == 2
    assert result.capex_used == pytest.approx(5.0e7)

    # Specifically B and C, not A.
    opened = set(result.opened_sites()["facility_key"])
    assert opened == {"S2", "S3"}


def test_milp_beats_greedy_on_the_toy():
    localities, facilities, distances, mask, demand = build_toy()
    greedy = greedy_network(localities, facilities, distances, mask, demand,
                            budget=TOY_BUDGET)
    milp = optimize_network(localities, facilities, distances, mask, demand,
                            budget=TOY_BUDGET,
                            lambda_capex=0.0, lambda_distance=0.0)

    improvement = milp.demand_served / greedy.demand_served - 1
    assert improvement == pytest.approx(0.75, abs=0.01)


def test_demand_balance_holds():
    """Served plus unserved must equal total demand, exactly.

    This is the accounting identity that stops demand quietly evaporating
    inside the model, and it is worth asserting directly: a formulation bug
    that loses orders would otherwise show up only as suspiciously good
    coverage numbers.
    """
    localities, facilities, distances, mask, demand = build_toy()
    result = optimize_network(localities, facilities, distances, mask, demand,
                              budget=TOY_BUDGET,
                              lambda_capex=0.0, lambda_distance=0.0)

    assert result.demand_served + result.demand_unserved == pytest.approx(demand.sum())


def test_capacity_is_never_exceeded():
    localities, facilities, distances, mask, demand = build_toy()
    result = optimize_network(localities, facilities, distances, mask, demand,
                              budget=TOY_BUDGET,
                              lambda_capex=0.0, lambda_distance=0.0)

    inflow = result.assignment.sum(axis=0)
    capacity = facilities["capacity"].to_numpy(dtype=float)
    assert np.all(inflow <= capacity + 1e-6)

    # And a closed facility must receive nothing at all.
    assert np.all(inflow[~result.open_flags] < 1e-6)


def test_budget_is_respected():
    localities, facilities, distances, mask, demand = build_toy()
    for budget in [2.0e7, 3.0e7, 5.0e7, 8.0e7]:
        result = optimize_network(localities, facilities, distances, mask, demand,
                                  budget=budget,
                                  lambda_capex=0.0, lambda_distance=0.0)
        assert result.status == "Optimal"
        assert result.capex_used <= budget + 1e-6


def test_tighter_budget_never_serves_more():
    """Monotonicity: more capital cannot produce a worse optimum.

    A useful invariant because it catches sign errors and mis-stated
    inequalities that a single-point test would pass straight through.
    """
    localities, facilities, distances, mask, demand = build_toy()
    served = []
    for budget in [2.0e7, 3.0e7, 5.0e7, 8.0e7]:
        result = optimize_network(localities, facilities, distances, mask, demand,
                                  budget=budget,
                                  lambda_capex=0.0, lambda_distance=0.0)
        served.append(result.demand_served)

    assert served == sorted(served)


def test_scenario_multipliers_stay_feasible():
    """The model must return a solution at festive demand, not give up.

    This is the payoff from making coverage a soft constraint. Under a hard
    "serve everything" constraint, multiplying demand by 2.5 on a fixed budget
    would make the problem infeasible and the solver would report nothing
    useful. Here it returns a plan plus a quantified shortfall.
    """
    localities, facilities, distances, mask, demand = build_toy()
    for multiplier in [1.0, 1.5, 2.0, 2.5]:
        result = optimize_network(localities, facilities, distances, mask,
                                  demand * multiplier, budget=TOY_BUDGET,
                                  lambda_capex=0.0, lambda_distance=0.0)
        assert result.status == "Optimal", f"infeasible at {multiplier}x demand"
        assert result.demand_unserved >= 0
