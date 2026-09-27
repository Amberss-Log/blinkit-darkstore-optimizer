"""Run every objective on the same inputs and put the results side by side.

The model can now be told to maximise three different things: orders served,
profit, or profit weighted by customer loyalty. This module runs all of them,
greedy and MILP, against one shared reference, so the notebook and the tests
read their numbers from the same place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import economics as ec
from src.baseline import greedy_network
from src.optimizer import optimize_network


def profit_settings(localities, facilities, demand, econ) -> dict:
    """Optimizer settings under which the objective is exactly profit.

    Orders are weighted by basket size alone, each store pays its running cost,
    and both lambdas are zero. Multiply the objective by the contribution of an
    average order and what comes out is total daily contribution minus the new
    stores' running costs -- which, less a constant for the existing network,
    is precisely what profit_report measures. Capital is controlled by the
    budget constraint.

    That equivalence is what makes this model's profit a guarantee rather than
    a hope: no plan that fits the budget can earn more, because earning more is
    the one thing the solver was told to do.
    """
    return {
        "value_weights": ec.value_weights(localities, demand, basket=True, loyalty=False),
        "running_cost": ec.running_cost_in_orders(facilities, localities, demand, econ),
        "lambda_capex": 0.0,
        "lambda_distance": 0.0,
    }


def loyalty_settings(localities, facilities, demand, econ) -> dict:
    """Optimizer settings that also value customer loyalty.

    Orders are weighted by basket size times the 90-day repeat rate, so a loyal
    area counts for more than today's cash alone would justify. The lambdas stay
    at their defaults. This objective is deliberately not profit: it trades some
    of today's money for customers likelier to keep ordering, so it carries no
    profit guarantee and can come out behind the profit model on profit.
    """
    return {
        "value_weights": ec.value_weights(localities, demand, basket=True, loyalty=True),
        "running_cost": ec.running_cost_in_orders(facilities, localities, demand, econ),
    }


def reference_network(localities, facilities, distances, mask, demand):
    """The existing six stores and nothing else, routed to earn the most.

    Every plan's profit is measured as the improvement over this. It has to be
    the same network for every plan, or the comparison would reward whichever
    plan happened to get the more flattering baseline.
    """
    return optimize_network(
        localities, facilities, distances, mask, demand, budget=0.0,
        value_weights=ec.value_weights(localities, demand, basket=True, loyalty=False),
        lambda_capex=0.0, lambda_distance=0.0,
    )


def run_objectives(localities, facilities, distances, mask, demand, budget, econ) -> dict:
    """Solve every plan once. Returns {name: NetworkResult}."""
    profit = profit_settings(localities, facilities, demand, econ)
    loyalty = loyalty_settings(localities, facilities, demand, econ)
    greedy_profit = {k: profit[k] for k in ("value_weights", "running_cost")}

    args = (localities, facilities, distances, mask, demand, budget)
    return {
        "greedy, orders": greedy_network(*args),
        "MILP, orders": optimize_network(*args),
        "greedy, profit": greedy_network(*args, **greedy_profit),
        "MILP, profit": optimize_network(*args, **profit),
        "MILP, loyalty": optimize_network(*args, **loyalty),
    }


def comparison_table(results: dict, econ, reference, highlight: str = "Vijay Nagar") -> pd.DataFrame:
    """One row per plan: size, coverage, who is served, and what it earns."""
    rows = []
    for name, result in results.items():
        tiers = ec.served_by_tier(result)
        money = ec.profit_report(result, econ, reference)
        localities = result.localities
        focus = (localities["locality_name"] == highlight).to_numpy()
        focus_share = 100 * result.assignment[focus].sum() / result.demand[focus].sum()

        rows.append({
            "plan": name,
            "new_stores": result.n_new_stores,
            "capex_cr": round(result.capex_used / 1e7, 2),
            "coverage_pct": round(result.coverage_pct, 1),
            "low_pct": tiers["low"],
            "mid_pct": tiers["mid"],
            "high_pct": tiers["high"],
            f"{highlight.lower().replace(' ', '_')}_pct": round(focus_share),
            "profit_cr_per_year": money["profit_cr_per_year"],
            "payback_years": money["payback_years"],
            "roce_pct": money["roce_pct"],
        })
    return pd.DataFrame(rows).set_index("plan")


def compare_objectives(localities, facilities, distances, mask, demand, budget, econ) -> pd.DataFrame:
    reference = reference_network(localities, facilities, distances, mask, demand)
    results = run_objectives(localities, facilities, distances, mask, demand, budget, econ)
    return comparison_table(results, econ, reference)


def margin_sweep(localities, facilities, distances, mask, demand, budget, margins) -> pd.DataFrame:
    """Profit of every plan at several margins.

    Everything that depends on the margin is rebuilt each time: the running
    cost expressed in orders changes with it, because a store's rent and staff
    are covered by fewer orders when each order earns more.
    """
    reference = reference_network(localities, facilities, distances, mask, demand)
    rows = {}
    for margin in margins:
        econ = ec.Economics(margin=margin)
        results = run_objectives(localities, facilities, distances, mask, demand, budget, econ)
        rows[round(100 * margin, 2)] = {
            name: ec.profit_report(result, econ, reference)["profit_cr_per_year"]
            for name, result in results.items()
        }
    table = pd.DataFrame(rows).T
    table.index.name = "margin_pct"
    return table
