"""Tests for the forecasting layer.

The tests that matter here are about leakage and benchmarking, not accuracy.
An accuracy assertion would either be so loose as to be meaningless or so tight
that it breaks whenever the seed changes. Leakage, by contrast, is a bug with a
definite right answer: either the model can see the future or it cannot.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.db import DATA_DIR, get_engine
from src.forecast import (
    build_features,
    fit_forecast,
    train_test_split_by_time,
)


@pytest.fixture(scope="module")
def raw():
    if not (os.getenv("DATABASE_URL") or "").strip() and not (DATA_DIR / "blinkit_sim.db").exists():
        pytest.skip("no database; run `python generate_data.py` first")

    engine = get_engine()
    demand = pd.read_sql("SELECT * FROM demand", engine)
    localities = pd.read_sql("SELECT * FROM localities", engine)
    return demand, localities


def test_lag_features_never_see_the_current_day(raw):
    """The leakage test, and the most important one in this file.

    lag_1d for a given day must equal the previous day's actual order count.
    If the shift were missing, it would equal the same day's count instead --
    which would make the model look excellent and be worthless, because that
    number does not exist at the moment a prediction has to be made.
    """
    demand, localities = raw
    frame = build_features(demand, localities)

    source = demand.copy()
    source["order_date"] = pd.to_datetime(source["order_date"])
    lookup = source.set_index(["locality_id", "order_date"])["order_count"]

    sample = frame.sample(n=40, random_state=0)
    for row in sample.itertuples():
        previous_day = row.order_date - pd.Timedelta(days=1)
        expected = lookup.get((row.locality_id, previous_day))
        assert expected is not None
        assert row.lag_1d == pytest.approx(float(expected))

        # And explicitly not today's value.
        today = float(lookup[(row.locality_id, row.order_date)])
        if today != float(expected):
            assert row.lag_1d != pytest.approx(today)


def test_split_is_chronological_and_disjoint(raw):
    demand, localities = raw
    frame = build_features(demand, localities)
    train, test = train_test_split_by_time(frame, test_days=14)

    assert len(train) > 0 and len(test) > 0
    assert train["order_date"].max() < test["order_date"].min()
    assert test["order_date"].nunique() == 14


def test_ridge_beats_the_naive_benchmark(raw):
    """A model that cannot beat the trailing mean has not earned its place.

    Asserting the direction rather than a threshold: the exact error depends on
    the seed, but a model that loses to the benchmark is broken regardless.
    """
    demand, localities = raw
    fit = fit_forecast(demand, localities)

    assert fit["ridge_mae"] < fit["naive_mae"]
    assert fit["improvement_pct"] > 0


def test_predictions_are_non_negative(raw):
    demand, localities = raw
    fit = fit_forecast(demand, localities)
    assert (fit["predictions"]["predicted"] >= 0).all()
