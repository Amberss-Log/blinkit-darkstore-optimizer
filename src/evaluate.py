"""Shared result type and dispatch rule.

Both the greedy baseline and the optimizer return a NetworkResult, and both
score their networks with the same metrics computed the same way. That is the
point of putting this in one place: if the two approaches measured coverage or
average distance even slightly differently, the headline comparison between
them would be an artefact of the measurement rather than a real difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def nearest_first_allocation(
    demand: np.ndarray,
    capacity: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    open_flags: np.ndarray,
) -> np.ndarray:
    """Assign demand to open facilities, closest pair first.

    Returns a matrix of orders per day flowing from each locality to each
    facility.

    This is a dispatch heuristic, not an optimization: walk every feasible
    locality-facility pair in ascending order of distance and push as much
    demand down each one as the remaining demand and remaining capacity allow.
    It is what a sensible operations team does by default -- serve each order
    from the nearest store that still has headroom.

    It is also deliberately *not* optimal, and understanding why matters for
    reading the results later. Committing a locality's demand to its nearest
    store can exhaust that store's capacity, leaving a second locality with no
    alternative within range and its demand stranded, when a slightly worse
    first assignment would have served both. The optimizer is free to make
    exactly that trade; this rule is not. That gap is part of what the
    comparison is measuring.
    """
    n_loc, n_fac = distances.shape
    assignment = np.zeros((n_loc, n_fac), dtype=float)

    remaining_demand = demand.astype(float).copy()
    remaining_capacity = np.where(open_flags, capacity.astype(float), 0.0)

    servable = mask & open_flags[None, :]
    loc_idx, fac_idx = np.nonzero(servable)
    order = np.argsort(distances[loc_idx, fac_idx], kind="stable")

    for i, j in zip(loc_idx[order], fac_idx[order]):
        qty = min(remaining_demand[i], remaining_capacity[j])
        if qty <= 0:
            continue
        assignment[i, j] = qty
        remaining_demand[i] -= qty
        remaining_capacity[j] -= qty

    return assignment


@dataclass
class NetworkResult:
    """A chosen network plus everything needed to judge it."""

    method: str
    open_flags: np.ndarray            # per facility, True if trading
    assignment: np.ndarray            # (n_loc, n_fac) orders/day
    facilities: pd.DataFrame
    localities: pd.DataFrame
    distances: np.ndarray
    demand: np.ndarray
    status: str = "ok"
    solve_seconds: float = 0.0
    notes: dict = field(default_factory=dict)

    # -- headline metrics ---------------------------------------------------

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())

    @property
    def demand_served(self) -> float:
        return float(self.assignment.sum())

    @property
    def demand_unserved(self) -> float:
        return self.total_demand - self.demand_served

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.demand_served / self.total_demand

    @property
    def new_site_flags(self) -> np.ndarray:
        return self.open_flags & ~self.facilities["is_existing"].to_numpy()

    @property
    def n_new_stores(self) -> int:
        return int(self.new_site_flags.sum())

    @property
    def capex_used(self) -> float:
        """Capital spent on *new* stores only. Existing stores carry capex 0."""
        return float(self.facilities["capex"].to_numpy()[self.open_flags].sum())

    @property
    def avg_delivery_km(self) -> float:
        """Demand-weighted average delivery distance.

        Weighting by demand is the whole point. A plain mean over open routes
        would let a near-empty route serving four orders a day count as heavily
        as one carrying two thousand, which tells you about the shape of the
        network rather than about what customers actually experience. The
        weighted figure answers the question an operator cares about: how far
        does the average *order* travel.
        """
        flows = self.assignment
        if flows.sum() == 0:
            return float("nan")
        return float((flows * self.distances).sum() / flows.sum())

    @property
    def capacity_utilisation_pct(self) -> float:
        open_capacity = self.facilities["capacity"].to_numpy()[self.open_flags].sum()
        if open_capacity == 0:
            return float("nan")
        return 100.0 * self.demand_served / float(open_capacity)

    # -- views --------------------------------------------------------------

    def opened_sites(self) -> pd.DataFrame:
        """The new stores this plan opens, in the order it opened them."""
        out = self.facilities[self.new_site_flags].copy()
        out["capex_cr"] = (out["capex"] / 1e7).round(2)
        served = self.assignment.sum(axis=0)[self.new_site_flags]
        out["orders_served"] = np.round(served, 1)
        out["utilisation_pct"] = np.round(100.0 * served / out["capacity"].to_numpy(), 1)
        cols = ["facility_key", "locality_id", "latitude", "longitude",
                "capacity", "capex_cr", "orders_served", "utilisation_pct"]
        return out[cols].reset_index(drop=True)

    def unserved_by_locality(self) -> pd.DataFrame:
        served = self.assignment.sum(axis=1)
        out = pd.DataFrame(
            {
                "locality_id": self.localities["locality_id"].to_numpy(),
                "locality_name": self.localities["locality_name"].to_numpy(),
                "demand": np.round(self.demand, 1),
                "served": np.round(served, 1),
                "unserved": np.round(self.demand - served, 1),
            }
        )
        out["coverage_pct"] = (100.0 * out["served"] / out["demand"]).round(1)
        return out.sort_values("unserved", ascending=False).reset_index(drop=True)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "status": self.status,
            "new_stores": self.n_new_stores,
            "capex_cr": round(self.capex_used / 1e7, 2),
            "demand_served": round(self.demand_served, 0),
            "coverage_pct": round(self.coverage_pct, 1),
            "avg_delivery_km": round(self.avg_delivery_km, 3),
            "capacity_util_pct": round(self.capacity_utilisation_pct, 1),
            "solve_seconds": round(self.solve_seconds, 2),
        }


def compare(results: list[NetworkResult]) -> pd.DataFrame:
    """Side-by-side comparison table for the notebook."""
    return pd.DataFrame([r.summary() for r in results]).set_index("method")
