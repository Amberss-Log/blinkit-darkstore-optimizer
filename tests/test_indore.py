"""Tests against the generated Indore dataset.

These need data/blinkit_sim.db to exist, so run `python generate_data.py`
first. They are skipped rather than failed if it is missing, because a fresh
clone legitimately has no database yet and a red suite for that reason would be
noise rather than signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.baseline import greedy_network
from src.db import DATA_DIR, get_engine, load_query
from src.geo import DEFAULT_RADIUS_KM, build_facilities, distance_matrix, feasible_mask
from src.optimizer import optimize_network

BUDGET = 20e7


@pytest.fixture(scope="module")
def network():
    import os

    if not (os.getenv("DATABASE_URL") or "").strip() and not (DATA_DIR / "blinkit_sim.db").exists():
        pytest.skip("no database; run `python generate_data.py` first")

    engine = get_engine()
    localities = pd.read_sql(text(load_query("05_rolling_demand_cte.sql")), engine)
    sites = pd.read_sql("SELECT * FROM candidate_sites", engine)
    stores = pd.read_sql("SELECT * FROM existing_stores", engine)

    facilities = build_facilities(sites, stores)
    distances = distance_matrix(localities, facilities)
    mask = feasible_mask(distances, DEFAULT_RADIUS_KM)
    demand = localities["smoothed_orders_per_day"].to_numpy(dtype=float)

    return localities, facilities, distances, mask, demand


def test_every_locality_is_reachable(network):
    """No locality should be stranded outside every facility's radius.

    If one were, its demand would be unservable at any budget -- a property of
    the candidate site list rather than a failure of the model, and much better
    known up front than inferred later from disappointing coverage.
    """
    _, _, distances, mask, _ = network
    assert mask.any(axis=1).all()


def test_milp_beats_greedy(network):
    localities, facilities, distances, mask, demand = network

    greedy = greedy_network(localities, facilities, distances, mask, demand, budget=BUDGET)
    milp = optimize_network(localities, facilities, distances, mask, demand, budget=BUDGET)

    assert milp.status == "Optimal"
    assert milp.demand_served > greedy.demand_served
    assert milp.capex_used <= BUDGET + 1e-6


def test_milp_buys_more_throughput_per_rupee(network):
    """The mechanism behind the improvement, asserted directly.

    Both approaches saturate their stores, so coverage is decided by how much
    capacity the budget buys. Greedy optimises capacity per *store*; the
    optimizer optimises capacity per *rupee*. Testing the ratio rather than
    just the headline number means a regression tells us which property broke.
    """
    localities, facilities, distances, mask, demand = network

    greedy = greedy_network(localities, facilities, distances, mask, demand, budget=BUDGET)
    milp = optimize_network(localities, facilities, distances, mask, demand, budget=BUDGET)

    greedy_ratio = greedy.demand_served / (greedy.capex_used / 1e7)
    milp_ratio = milp.demand_served / (milp.capex_used / 1e7)
    assert milp_ratio > greedy_ratio


def test_demand_balance_and_capacity_hold(network):
    localities, facilities, distances, mask, demand = network
    result = optimize_network(localities, facilities, distances, mask, demand, budget=BUDGET)

    assert result.demand_served + result.demand_unserved == pytest.approx(demand.sum())

    inflow = result.assignment.sum(axis=0)
    assert np.all(inflow <= facilities["capacity"].to_numpy(dtype=float) + 1e-6)
    assert np.all(inflow[~result.open_flags] < 1e-6)


def test_no_demand_assigned_outside_radius(network):
    """The radius is enforced structurally, so this should be impossible.

    Worth asserting anyway: the mask is built in one place and consumed in two,
    and a mismatch between them would silently produce deliveries the business
    could never actually make.
    """
    localities, facilities, distances, mask, demand = network
    result = optimize_network(localities, facilities, distances, mask, demand, budget=BUDGET)

    assert np.all(result.assignment[~mask] == 0)
    served_distances = result.distances[result.assignment > 0]
    assert served_distances.max() <= DEFAULT_RADIUS_KM + 1e-9


def test_festive_scenarios_stay_feasible(network):
    localities, facilities, distances, mask, demand = network

    previous_unserved = -1.0
    for multiplier in [1.0, 1.5, 2.0, 2.5]:
        result = optimize_network(localities, facilities, distances, mask,
                                  demand * multiplier, budget=BUDGET)
        assert result.status == "Optimal", f"infeasible at {multiplier}x"

        # Capacity is fixed by the budget, so more demand must mean more
        # unmet demand, never less.
        assert result.demand_unserved > previous_unserved
        previous_unserved = result.demand_unserved


def test_zero_budget_changes_nothing(network):
    """With no capital, the answer is the incumbent network -- unchanged.

    An edge case that catches a whole family of bugs at once: if the model
    opened anything here, the budget constraint or the existing-store handling
    would be wrong.
    """
    localities, facilities, distances, mask, demand = network
    result = optimize_network(localities, facilities, distances, mask, demand, budget=0.0)

    assert result.status == "Optimal"
    assert result.n_new_stores == 0
    assert result.capex_used == pytest.approx(0.0)


def test_rounding_the_relaxation_is_not_a_substitute_for_the_milp(network):
    """The LP relaxation scores within half a percent of the MILP, which makes
    "solve the easy version and round it" look tempting. This test pins down
    why it does not work here: rounding down or to nearest loses coverage
    against the MILP, and rounding up breaks the budget. If a data change ever
    made rounding competitive, this is where it would show up.
    """
    from src.optimizer import round_relaxation

    localities, facilities, distances, mask, demand = network
    table = round_relaxation(localities, facilities, distances, mask, demand, BUDGET)

    milp = table.loc["MILP"]
    assert milp["within_budget"]

    for plan in ["LP, round down", "LP, round to nearest"]:
        row = table.loc[plan]
        assert row["within_budget"]
        assert row["coverage_pct"] < milp["coverage_pct"], f"{plan} matched the MILP"

    # Rounding up buys every partly-opened site, which the budget cannot cover.
    assert not table.loc["LP, round up", "within_budget"]

    # The relaxation really is fractional; otherwise the test proves nothing.
    assert len(table.attrs["fractional_sites"]) > 0
