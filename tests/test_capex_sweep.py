"""Tests for the bigger-budget experiment.

Each test pins down one claim the notebook makes about spending more: that
extra money stops buying anything past a point, that no plan builds stores to
sit idle, that a network sized for normal days absorbs only about 1/2.5 of a
Diwali rush, and that the Rs 20 cr rows agree with the headline figures.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import text

from src import economics as ec
from src.compare import capex_sweep
from src.db import DATA_DIR, get_engine, load_query
from src.geo import DEFAULT_RADIUS_KM, build_facilities, distance_matrix, feasible_mask

BUDGETS = [20, 30, 40, 60]


@pytest.fixture(scope="module")
def sweep():
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
    econ = ec.default_economics(localities, facilities, demand)

    return capex_sweep(localities, facilities, distances, mask, demand, BUDGETS, econ).set_index(
        ["model", "budget_cr"]
    )


def test_budget_rows_agree_with_the_headline(sweep):
    """At Rs 20 cr with no festive days, the sweep reproduces the main results."""
    assert sweep.loc[("profit", 20), "profit_cr_no_festival"] == pytest.approx(10.02, abs=0.01)
    assert sweep.loc[("orders", 20), "profit_cr_no_festival"] == pytest.approx(8.16, abs=0.01)


@pytest.mark.parametrize("model", ["orders", "profit"])
def test_extra_money_stops_buying_anything(sweep, model):
    """Past saturation the plan is identical, and the extra budget goes unspent."""
    at_40, at_60 = sweep.loc[(model, 40)], sweep.loc[(model, 60)]
    assert at_40["new_stores"] == at_60["new_stores"]
    assert at_40["capex_cr"] == at_60["capex_cr"]
    assert at_60["capex_cr"] < 40


def test_no_plan_builds_idle_stores(sweep):
    """A bigger budget never leaves stores with meaningful spare room on normal days."""
    assert (sweep["normal_utilisation_pct"] >= 99.0).all()


def test_festive_coverage_is_capped_by_normal_sizing(sweep):
    """With every store full, a 2.5x rush is served at about normal coverage / 2.5."""
    assert (sweep["festive_utilisation_pct"] >= 99.9).all()
    expected = sweep["normal_coverage_pct"] / 2.5
    assert ((sweep["festive_coverage_pct"] - expected).abs() <= 1.0).all()


def test_more_capex_earns_more_but_returns_less(sweep):
    profit = sweep.xs("profit", level="model")
    assert profit["profit_cr_per_year"].is_monotonic_increasing
    assert profit.loc[20, "roce_pct"] > profit.loc[30, "roce_pct"] > profit.loc[40, "roce_pct"]


def test_festive_days_barely_move_the_year(sweep):
    """Even counting 8 full 2.5x days, the yearly figure moves by under Rs 0.1 cr."""
    delta = (sweep["profit_cr_per_year"] - sweep["profit_cr_no_festival"]).abs()
    assert (delta < 0.1).all()
