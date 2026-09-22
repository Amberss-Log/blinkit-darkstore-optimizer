"""Demand forecasting.

Everything up to this point feeds the optimizer with demand *observed* over the
last ninety days. But a store opened today serves demand that has not happened
yet, so siting on historical averages quietly assumes the future resembles the
past. This module replaces that assumption with a model and, more importantly,
measures whether the model is worth having.

The technique is deliberately simple. Ridge regression on interpretable
features beats a gradient-boosted ensemble here for the reason that matters in
an interview: you can state what the model learned and why, and defend it. A
black box that scores marginally better but cannot be explained is worse than
useless when someone asks why a particular locality was forecast to grow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Where to cut the history. The last fourteen days are held out.
TEST_DAYS = 14


def build_features(demand: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw daily table into a design matrix.

    Three families of feature, each justified rather than thrown in:

    Calendar features capture the weekly cycle the data demonstrably has -- a
    22% weekend uplift. Day of week is encoded as dummies rather than as an
    integer, because Saturday is not "five more than Monday" in any sense a
    linear model should believe.

    Locality features -- population density and income tier -- capture the
    cross-sectional variation. Density enters logged, because demand was
    generated with a sublinear response to it and because a raw density
    ranging from 1,900 to 25,700 would otherwise dominate the regularisation
    penalty purely through its scale.

    A lagged demand feature carries the locality's own recent level, which is
    by far the strongest single predictor. Note it is a *lag*, computed from
    days strictly before the one being predicted. Using same-day demand would
    be leakage: information unavailable at prediction time, producing an
    accuracy figure you could never reproduce in practice.
    """
    df = demand.merge(localities, on="locality_id", how="left").copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values(["locality_id", "order_date"]).reset_index(drop=True)

    df["dow"] = df["order_date"].dt.dayofweek
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True).astype(float)

    df["log_density"] = np.log(df["population_density"])
    tier_dummies = pd.get_dummies(
        df["avg_household_income_tier"], prefix="tier", drop_first=True
    ).astype(float)

    # Trailing seven-day mean, shifted by one so the current day never sees
    # itself. This is the single most important line in the module: get the
    # shift wrong and every metric below becomes fiction.
    df["lag_7d_mean"] = (
        df.groupby("locality_id")["order_count"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    )
    df["lag_1d"] = df.groupby("locality_id")["order_count"].shift(1)

    features = pd.concat(
        [df[["log_density", "lag_7d_mean", "lag_1d"]], dow_dummies, tier_dummies],
        axis=1,
    )

    out = pd.concat(
        [df[["locality_id", "locality_name", "order_date", "order_count"]], features],
        axis=1,
    )
    return out.dropna().reset_index(drop=True)


def seasonal_naive(frame: pd.DataFrame) -> np.ndarray:
    """The benchmark: predict each day as the same locality's trailing weekly mean.

    Every forecasting exercise needs one of these, and it is the step most
    often skipped. A model reporting a mean absolute error of 40 orders sounds
    respectable until you learn the naive rule scores 42, at which point all
    that work bought you two orders. Without a benchmark there is no way to
    tell the difference between a good model and an easy problem.
    """
    return frame["lag_7d_mean"].to_numpy(dtype=float)


def train_test_split_by_time(frame: pd.DataFrame, test_days: int = TEST_DAYS):
    """Split on the calendar, never at random.

    A random split would scatter days from the same week across both sides,
    letting the model see Wednesday while predicting Tuesday of the same week.
    With a lagged feature that is close to handing it the answer, and the test
    score would flatter the model enormously. Time series are split by time:
    train on the past, test on the future, exactly as the model would be used.
    """
    cutoff = frame["order_date"].max() - pd.Timedelta(days=test_days)
    train = frame[frame["order_date"] <= cutoff]
    test = frame[frame["order_date"] > cutoff]
    return train, test


def fit_forecast(
    demand: pd.DataFrame,
    localities: pd.DataFrame,
    alpha: float = 1.0,
    test_days: int = TEST_DAYS,
) -> dict:
    """Fit the ridge model and score it against the naive benchmark."""
    frame = build_features(demand, localities)
    train, test = train_test_split_by_time(frame, test_days)

    feature_columns = [
        c for c in frame.columns
        if c not in {"locality_id", "locality_name", "order_date", "order_count"}
    ]

    # Ridge penalises large coefficients, which keeps correlated features --
    # and lag_1d and lag_7d_mean are strongly correlated -- from producing the
    # huge offsetting coefficients that plain least squares would. Scaling
    # first matters because the penalty is applied in coefficient units, so
    # unscaled features are regularised unequally purely by their magnitude.
    scaler = StandardScaler().fit(train[feature_columns])
    model = Ridge(alpha=alpha).fit(
        scaler.transform(train[feature_columns]), train["order_count"]
    )

    predictions = model.predict(scaler.transform(test[feature_columns]))
    predictions = np.clip(predictions, 0, None)      # orders cannot be negative

    baseline = seasonal_naive(test)
    actual = test["order_count"].to_numpy(dtype=float)

    ridge_mae = mean_absolute_error(actual, predictions)
    naive_mae = mean_absolute_error(actual, baseline)

    coefficients = pd.Series(model.coef_, index=feature_columns).sort_values(
        key=np.abs, ascending=False
    )

    scored = test[["locality_id", "locality_name", "order_date"]].copy()
    scored["actual"] = actual
    scored["predicted"] = predictions
    scored["naive"] = baseline

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "ridge_mae": ridge_mae,
        "naive_mae": naive_mae,
        "improvement_pct": 100.0 * (naive_mae - ridge_mae) / naive_mae,
        "coefficients": coefficients,
        "predictions": scored,
        "train_days": train["order_date"].nunique(),
        "test_days": test["order_date"].nunique(),
    }


def forecast_demand_vector(fit: dict, localities: pd.DataFrame) -> pd.DataFrame:
    """Per-locality forecast demand, in the shape the optimizer expects.

    Averages each locality's predictions over the held-out window, producing a
    single orders-per-day figure per locality -- the same interface the
    smoothed historical demand presents, so the optimizer needs no change to
    accept it. Swapping one input for another and re-running is the whole point
    of having kept the model separate from the data it consumes.
    """
    predictions = fit["predictions"]
    forecast = (
        predictions.groupby(["locality_id", "locality_name"])["predicted"]
        .mean()
        .reset_index()
        .rename(columns={"predicted": "forecast_orders_per_day"})
    )

    out = localities.merge(forecast, on="locality_id", how="left", suffixes=("", "_f"))
    if "locality_name_f" in out.columns:
        out = out.drop(columns=["locality_name_f"])

    # A locality with too little history to produce a lag falls back to its
    # smoothed historical demand rather than being silently dropped, which
    # would change the total demand the two runs are compared on.
    if "smoothed_orders_per_day" in out.columns:
        out["forecast_orders_per_day"] = out["forecast_orders_per_day"].fillna(
            out["smoothed_orders_per_day"]
        )

    return out
