"""Tests for the money side of the model: value weights, running costs,
the derived margin, and the profit comparison.

The most important test here is the profit guarantee. The profit model's
objective is set up to equal profit exactly, so no other plan should ever beat
it on profit. If that ever fails, the equivalence has broken -- a unit
mismatch between the weights and the running costs, say -- and every profit
figure in the notebook becomes suspect.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src import economics as ec
from src.compare import margin_sweep, profit_settings, reference_network, run_objectives
from src.db import DATA_DIR, get_engine, load_query
from src.geo import DEFAULT_RADIUS_KM, build_facilities, distance_matrix, feasible_mask
from src.baseline import greedy_network
from src.optimizer import optimize_network

BUDGET = 20e7


@pytest.fixture(scope="module")
def network():
    if not (os.getenv("DATABASE_URL") or "").strip() and not (DATA_DIR / "blinkit_sim.db").exists():
        pytest.skip("no database; run `python generate_data.py` first")

    engine = get_engine()
    localities = pd.read_sql(text(load_query("05_rolling_demand_cte.sql")), engine)
    facilities = build_facilities(
        pd.read_sql("SELECT * FROM candidate_sites", engine),
        pd.read_sql("SELECT * FROM existing_stores", engine),
    )
    distances = distance_matrix(localities, facilities)
    mask = feasible_mask(distances, DEFAULT_RADIUS_KM)
    demand = localities["smoothed_orders_per_day"].to_numpy(dtype=float)
    return localities, facilities, distances, mask, demand


# --- value weights and running costs ------------------------------------------

def test_weights_off_are_all_ones(network):
    localities, _, _, _, demand = network
    weights = ec.value_weights(localities, demand, basket=False, loyalty=False)
    assert np.allclose(weights, 1.0)


@pytest.mark.parametrize("basket,loyalty", [(True, False), (False, True), (True, True)])
def test_weights_average_to_one_across_demand(network, basket, loyalty):
    """Keeps the objective in 'average orders', so weighted and unweighted
    models stay directly comparable."""
    localities, _, _, _, demand = network
    weights = ec.value_weights(localities, demand, basket=basket, loyalty=loyalty)
    assert (weights * demand).sum() / demand.sum() == pytest.approx(1.0)


def test_weights_favour_bigger_baskets_and_loyal_customers(network):
    localities, _, _, _, demand = network
    weights = ec.value_weights(localities, demand)
    by_tier = pd.Series(weights).groupby(localities["avg_household_income_tier"].to_numpy()).mean()
    assert by_tier["high"] > by_tier["mid"] > by_tier["low"]


def test_existing_stores_carry_no_running_cost(network):
    _, facilities, _, _, _ = network
    daily = ec.running_cost_per_day(facilities, ec.Economics(margin=0.045))
    assert np.all(daily[facilities["is_existing"].to_numpy()] == 0)
    assert np.all(daily[~facilities["is_existing"].to_numpy()] > 0)


# --- the derived margin --------------------------------------------------------

def test_derived_margin_reproduces_the_target_roce(network):
    """Feed the margin back in: a typical store should earn the disclosed 41.7%."""
    localities, facilities, _, _, demand = network
    margin = ec.derive_margin(localities, facilities, demand)

    candidates = facilities[~facilities["is_existing"]]
    running = (candidates["rent_monthly"].median() + ec.STAFF_COST_MONTHLY) / ec.DAYS_PER_MONTH
    daily_nov = ec.TYPICAL_CAPACITY * ec.mean_order_value(localities, demand)
    yearly_profit = (daily_nov * margin - running) * 365

    assert yearly_profit / ec.CAPEX_PER_STORE == pytest.approx(ec.TARGET_PRETAX_ROCE, rel=1e-9)
    assert 0.01 < margin < 0.15       # a plausible quick-commerce contribution margin


# --- the new options must not disturb the original model ------------------------

def test_neutral_options_reproduce_the_original_model(network):
    localities, facilities, distances, mask, demand = network
    original = optimize_network(localities, facilities, distances, mask, demand, BUDGET)
    neutral = optimize_network(
        localities, facilities, distances, mask, demand, BUDGET,
        value_weights=np.ones(len(localities)),
        running_cost=np.zeros(len(facilities)),
    )
    assert set(original.opened_sites()["facility_key"]) == set(neutral.opened_sites()["facility_key"])
    assert original.notes["objective_value"] == pytest.approx(neutral.notes["objective_value"])


def test_prohibitive_running_costs_open_nothing(network):
    """If no store can cover its running cost, neither method should open one."""
    localities, facilities, distances, mask, demand = network
    huge = np.full(len(facilities), 1e7)
    greedy = greedy_network(localities, facilities, distances, mask, demand, BUDGET, running_cost=huge)
    milp = optimize_network(localities, facilities, distances, mask, demand, BUDGET, running_cost=huge)
    assert greedy.n_new_stores == 0
    assert milp.n_new_stores == 0


# --- profit ---------------------------------------------------------------------

def test_plan_measured_against_itself_earns_nothing(network):
    localities, facilities, distances, mask, demand = network
    reference = reference_network(localities, facilities, distances, mask, demand)
    report = ec.profit_report(reference, ec.Economics(margin=0.045), reference)
    assert report["profit_cr_per_year"] == pytest.approx(0.0)
    assert report["new_stores"] == 0


def test_profit_model_objective_equals_its_profit(network):
    """The equivalence the guarantee rests on, checked directly: the solver's
    objective, converted to rupees, matches the independently computed profit
    once the reference network's constant is added back."""
    localities, facilities, distances, mask, demand = network
    econ = ec.Economics(margin=0.045)
    reference = reference_network(localities, facilities, distances, mask, demand)
    plan = optimize_network(localities, facilities, distances, mask, demand, BUDGET,
                            **profit_settings(localities, facilities, demand, econ))

    rupees_per_unit = ec.mean_order_value(localities, demand) * econ.margin
    objective_rupees = plan.notes["objective_value"] * rupees_per_unit
    new = plan.new_site_flags
    expected = (ec.contribution_per_day(plan, econ)
                - ec.running_cost_per_day(plan.facilities, econ)[new].sum())
    assert objective_rupees == pytest.approx(expected, rel=1e-6)


def test_profit_model_is_never_beaten_on_profit(network):
    """The guarantee. At every margin, no other plan earns more."""
    localities, facilities, distances, mask, demand = network
    econ = ec.default_economics(localities, facilities, demand)
    sweep = margin_sweep(localities, facilities, distances, mask, demand, BUDGET,
                         [0.03, econ.margin, 0.06, 0.08])

    for margin, row in sweep.iterrows():
        best_other = row.drop("MILP, profit").max()
        assert row["MILP, profit"] >= best_other - 1e-6, f"beaten at {margin}% margin"


def test_counting_orders_is_not_the_same_as_earning_money(network):
    """The finding that reframes the first result: at the derived margin, the
    order-counting MILP serves more orders than greedy but earns less."""
    localities, facilities, distances, mask, demand = network
    econ = ec.default_economics(localities, facilities, demand)
    reference = reference_network(localities, facilities, distances, mask, demand)
    plans = run_objectives(localities, facilities, distances, mask, demand, BUDGET, econ)

    greedy, milp = plans["greedy, orders"], plans["MILP, orders"]
    assert milp.demand_served > greedy.demand_served
    profit = lambda r: ec.profit_report(r, econ, reference)["profit_cr_per_year"]
    assert profit(milp) < profit(greedy)


def test_value_weighting_keeps_high_income_customers(network):
    """The fix for the original model's blind spot."""
    localities, facilities, distances, mask, demand = network
    econ = ec.default_economics(localities, facilities, demand)
    plans = run_objectives(localities, facilities, distances, mask, demand, BUDGET, econ)

    assert ec.served_by_tier(plans["MILP, orders"])["high"] < 100.0
    assert ec.served_by_tier(plans["MILP, profit"])["high"] == pytest.approx(100.0)
