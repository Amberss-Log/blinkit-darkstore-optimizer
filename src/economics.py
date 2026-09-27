"""Order value, loyalty, running costs and profit.

The first version of the model counted orders. Every order was worth the same
to it, whether the basket was Rs 420 or Rs 720, and whether the customer
reorders every week or tried the service once. It also treated a store as
free to run once it was open. This module adds the three things that version
left out: what an order is worth, how loyal the customers behind it are, and
what a store costs to keep open.

Two of the inputs here are assumptions and are labelled as such. The profit
margin is not disclosed, so it is derived below from the one return figure
that is. Staff cost per store is assumed outright.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- disclosed ---------------------------------------------------------------
CAPEX_PER_STORE = 2.5e7          # Rs 2.5 crore, Eternal
TARGET_PRETAX_ROCE = 0.417       # 41.7%, Eternal

# --- assumed -----------------------------------------------------------------
# Staff and utilities for one dark store per month: roughly fifteen to twenty
# pickers, packers and a store manager. Not disclosed anywhere.
STAFF_COST_MONTHLY = 4.5e5
DAYS_PER_MONTH = 30.0

# --- derived -----------------------------------------------------------------
TYPICAL_CAPACITY = 1_800         # Rs 11 lakh NOV / ~Rs 600 AOV, see README


@dataclass(frozen=True)
class Economics:
    """The money assumptions behind a profit calculation.

    `margin` is contribution margin as a share of NOV: what is left of each
    rupee of orders after the cost of goods, delivery, discounts and payment
    fees, before the store's own fixed running costs.
    """

    margin: float
    staff_cost_monthly: float = STAFF_COST_MONTHLY
    days_per_month: float = DAYS_PER_MONTH


def mean_order_value(localities: pd.DataFrame, demand: np.ndarray) -> float:
    """Demand-weighted average basket, so busy areas count for more."""
    aov = localities["avg_order_value"].to_numpy(dtype=float)
    return float((aov * demand).sum() / demand.sum())


def derive_margin(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    demand: np.ndarray,
    staff_cost_monthly: float = STAFF_COST_MONTHLY,
) -> float:
    """The contribution margin at which a typical store earns the target ROCE.

    Blinkit's margin is not disclosed, but its target pre-tax return on
    capital is: about 41.7%. So instead of picking a margin, ask what margin a
    typical store would need to hit that target. A typical store costs Rs 2.5
    crore, packs about 1,800 orders a day, and has running costs made up of
    rent plus staff. Earning 41.7% a year on Rs 2.5 crore means roughly
    Rs 28,600 a day of profit after running costs, and the margin is whatever
    share of the store's daily NOV covers both that profit and the running
    costs.

    This ties an unknown to a disclosed figure the same way store capacity was
    derived from disclosed NOV. It inherits the same weakness: the result is
    only as good as the assumptions it is built from, here staff cost and the
    typical capacity. It also treats return on capital as return on capex
    alone, leaving out working capital, which is a simplification.
    """
    candidates = facilities[~facilities["is_existing"]]
    typical_rent = float(candidates["rent_monthly"].median())
    running_per_day = (typical_rent + staff_cost_monthly) / DAYS_PER_MONTH
    target_profit_per_day = TARGET_PRETAX_ROCE * CAPEX_PER_STORE / 365.0

    daily_nov = TYPICAL_CAPACITY * mean_order_value(localities, demand)
    return (target_profit_per_day + running_per_day) / daily_nov


def default_economics(
    localities: pd.DataFrame, facilities: pd.DataFrame, demand: np.ndarray
) -> Economics:
    return Economics(margin=derive_margin(localities, facilities, demand))


def value_weights(
    localities: pd.DataFrame,
    demand: np.ndarray,
    basket: bool = True,
    loyalty: bool = True,
) -> np.ndarray:
    """How much one order from each locality is worth, relative to an average one.

    Two things make an order worth more. A bigger basket brings in more money
    today. A more loyal customer is likelier to order again, so serving them
    now protects future orders as well; losing a loyal area to a competitor
    costs more than losing a fickle one. The weight multiplies the two:
    average order value times the 90-day repeat rate.

    The weights are scaled so their demand-weighted average is exactly 1. That
    keeps the objective in the same units as before -- "average orders" -- so
    the capex weight lambda_capex keeps its meaning of orders per crore, and a
    weighted model can be compared with the unweighted one directly. Turning
    both flags off returns all ones, which reproduces the original model.
    """
    weights = np.ones(len(localities), dtype=float)
    if basket:
        weights *= localities["avg_order_value"].to_numpy(dtype=float)
    if loyalty:
        weights *= localities["repeat_rate_90d"].to_numpy(dtype=float)

    return weights * demand.sum() / (weights * demand).sum()


def running_cost_per_day(facilities: pd.DataFrame, econ: Economics) -> np.ndarray:
    """Rent plus staff for each facility, in rupees per day.

    Zero for existing stores: they stay open under every plan, so their
    running cost is a constant that cannot change which sites get chosen.
    """
    daily = (facilities["rent_monthly"].to_numpy(dtype=float) + econ.staff_cost_monthly) / econ.days_per_month
    return np.where(facilities["is_existing"].to_numpy(dtype=bool), 0.0, daily)


def running_cost_in_orders(
    facilities: pd.DataFrame, localities: pd.DataFrame, demand: np.ndarray, econ: Economics
) -> np.ndarray:
    """Each facility's daily running cost, converted into orders.

    The optimizer's objective is counted in orders, so a rupee cost has to be
    translated before it can enter. Dividing by the contribution one average
    order brings in gives the number of orders a day a store must serve just to
    pay its own rent and staff. With the default assumptions that is roughly
    700 of a typical store's 1,800 -- a real hurdle, and the reason a store in
    a thin area can lose money even when it is full.
    """
    contribution_per_order = mean_order_value(localities, demand) * econ.margin
    return running_cost_per_day(facilities, econ) / contribution_per_order


def contribution_per_day(result, econ: Economics) -> float:
    """Contribution from every order a plan serves, in rupees per day."""
    aov = result.localities["avg_order_value"].to_numpy(dtype=float)
    return float((result.assignment.sum(axis=1) * aov).sum() * econ.margin)


def profit_report(result, econ: Economics, reference) -> dict:
    """What a plan's investment earns, measured against doing nothing.

    `reference` is the network with only the existing stores. The profit
    attributed to a plan is its total contribution minus the reference's total
    contribution, minus the running costs of the stores it opens.

    Measuring it this way, and not by summing what the new stores themselves
    serve, matters. The existing stores run full under every plan, but which
    customers they serve changes from plan to plan. When a new store takes
    over a nearby area, the existing store next to it is freed up to serve
    someone else. Crediting a plan only with its new stores' orders misses that
    knock-on effect and gets the comparison wrong. Comparing whole networks
    against the same reference captures it.

    Existing stores' own running costs are left out because they are paid
    under every plan, including the reference, so they cancel.

    Loyalty does not appear here. It shapes which customers the model chooses
    to serve, because it stands for future orders, but this report counts only
    today's money. The gap between the two is intentional: a plan can earn
    slightly less today while protecting more valuable customers.
    """
    new = result.new_site_flags
    incremental = contribution_per_day(result, econ) - contribution_per_day(reference, econ)
    running_per_day = float(running_cost_per_day(result.facilities, econ)[new].sum())
    profit_per_year = (incremental - running_per_day) * 365.0
    capex = result.capex_used

    return {
        "method": result.method,
        "new_stores": result.n_new_stores,
        "capex_cr": round(capex / 1e7, 2),
        "extra_contribution_lakh_per_day": round(incremental / 1e5, 2),
        "running_cost_lakh_per_day": round(running_per_day / 1e5, 2),
        "profit_cr_per_year": round(profit_per_year / 1e7, 2),
        "payback_years": round(capex / profit_per_year, 2) if profit_per_year > 0 else None,
        "roce_pct": round(100 * profit_per_year / capex, 1) if capex > 0 else None,
    }


def served_by_tier(result, demand: np.ndarray | None = None) -> pd.Series:
    """Share of each income tier's demand that a plan serves, in percent."""
    demand = result.demand if demand is None else demand
    tiers = result.localities["avg_household_income_tier"].to_numpy()
    served = result.assignment.sum(axis=1)
    return pd.Series(
        {t: round(100 * served[tiers == t].sum() / demand[tiers == t].sum(), 1)
         for t in ["low", "mid", "high"]},
        name=result.method,
    )
