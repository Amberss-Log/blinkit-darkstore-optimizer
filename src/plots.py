"""Figures for the notebook.

Kept out of the notebook itself so the notebook reads as an argument rather
than as forty lines of axis configuration between each point being made.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from src.evaluate import NetworkResult
from src.geo import DEFAULT_RADIUS_KM

# One consistent palette across every figure, so the same thing is the same
# colour everywhere.
C_DEMAND = "#7fb3d5"
C_EXISTING = "#34495e"
C_NEW = "#e8743b"
C_UNUSED = "#c8ccd0"
C_GREEDY = "#95a5a6"
C_MILP = "#e8743b"


def _km_to_deg(km: float, latitude: float) -> tuple[float, float]:
    """Convert a distance in km to degrees of latitude and longitude.

    A degree of latitude is ~111 km everywhere, but a degree of longitude is
    111 km x cos(latitude) -- about 0.92 of that at Indore. Drawing a radius as
    a circle in degree space without this correction produces a circle that is
    visibly wrong in the east-west direction, which is exactly the error the
    haversine calculation exists to avoid.
    """
    return km / 111.0, km / (111.0 * np.cos(np.radians(latitude)))


def plot_network(
    result: NetworkResult,
    radius_km: float = DEFAULT_RADIUS_KM,
    show_radius: bool = True,
    title: str | None = None,
    ax=None,
):
    """The map: demand density, chosen sites, and their coverage."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))

    localities = result.localities
    facilities = result.facilities

    lat = localities["latitude"].to_numpy(dtype=float)
    lon = localities["longitude"].to_numpy(dtype=float)

    # Bubble area scales with demand. Area rather than radius, because visual
    # comparison is done by area -- scaling the radius would exaggerate large
    # values by squaring them.
    sizes = 40 + 900 * (result.demand / result.demand.max())

    served_share = result.assignment.sum(axis=1) / result.demand

    scatter = ax.scatter(
        lon, lat, s=sizes, c=served_share, cmap="RdYlGn",
        vmin=0, vmax=1, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3,
    )
    plt.colorbar(scatter, ax=ax, label="share of locality demand served", shrink=0.8)

    existing = facilities[facilities["is_existing"]]
    new = facilities[result.new_site_flags]
    unused = facilities[~result.open_flags & ~facilities["is_existing"]]

    ax.scatter(unused["longitude"], unused["latitude"], marker="x", s=40,
               c=C_UNUSED, linewidths=1.2, zorder=4, label="candidate, not opened")
    ax.scatter(existing["longitude"], existing["latitude"], marker="s", s=110,
               c=C_EXISTING, edgecolors="white", linewidths=1.2, zorder=6,
               label="existing store")
    ax.scatter(new["longitude"], new["latitude"], marker="*", s=340,
               c=C_NEW, edgecolors="white", linewidths=1.0, zorder=7,
               label="newly opened")

    if show_radius:
        dlat, dlon = _km_to_deg(radius_km, float(lat.mean()))
        for _, facility in pd.concat([existing, new]).iterrows():
            ax.add_patch(
                Circle(
                    (facility["longitude"], facility["latitude"]),
                    radius=dlon, fill=False, linestyle="--", linewidth=0.6,
                    edgecolor=C_NEW if not facility["is_existing"] else C_EXISTING,
                    alpha=0.28, zorder=2,
                )
            )
        # Circles are drawn with the longitude radius, so the aspect ratio has
        # to compensate or they render as ellipses.
        ax.set_aspect(dlon / dlat)

    # Label the worst-served localities rather than the busiest. Labelling all
    # thirty-five is unreadable, and the busiest are clustered in the centre
    # where the labels collide with each other anyway. The places the network
    # fails are the point of the figure, they sit on the periphery where there
    # is room to write, and they are what someone looking at this map should
    # walk away remembering.
    if "locality_name" in localities.columns:
        unserved = result.demand - result.assignment.sum(axis=1)
        worst = np.argsort(unserved)[::-1][:5]
        for i in worst:
            if unserved[i] < 1:
                continue
            ax.annotate(
                localities.iloc[i]["locality_name"],
                (lon[i], lat[i]),
                fontsize=8, xytext=(7, 7), textcoords="offset points",
                zorder=8, color="#8b1a1a",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"{result.method}: {result.n_new_stores} new stores, "
                          f"{result.coverage_pct:.1f}% coverage")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    return ax


def plot_comparison(comparison: pd.DataFrame, ax=None):
    """Baseline against optimized on the metrics that matter.

    Four small panels rather than one chart with four series, because the
    metrics have different units and forcing them onto a shared axis would
    either need normalisation -- which hides the actual values -- or a second
    y-axis, which invites misreading.
    """
    metrics = [
        ("coverage_pct", "demand coverage (%)", False),
        ("demand_served", "orders served per day", False),
        ("capex_cr", "capital deployed (Rs cr)", False),
        ("avg_delivery_km", "avg delivery distance (km)", True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    colours = [C_GREEDY if m == "greedy" else C_MILP for m in comparison.index]

    for axis, (column, label, lower_is_better) in zip(axes, metrics):
        values = comparison[column]
        bars = axis.bar(comparison.index, values, color=colours, width=0.55)
        axis.set_title(label, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value:,.2f}" if value < 100 else f"{value:,.0f}",
                      ha="center", va="bottom", fontsize=9)
        axis.margins(y=0.18)
        if lower_is_better:
            axis.set_xlabel("lower is better", fontsize=8, color="#777")

    fig.tight_layout()
    return fig


def plot_sensitivity(sweep: pd.DataFrame, breakeven: float | None = 720.0):
    """Coverage and store count against the capital weight.

    The plateau is the finding, not the curve. A recommendation that holds
    across a wide band of lambda is one you can act on; one that moves with
    every nudge of the weight is an artefact of the weight.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(sweep.index, sweep["coverage_pct"], marker="o",
            color=C_MILP, linewidth=2, label="coverage (%)")
    ax.set_xlabel("lambda_capex  (orders/day considered worth one crore)")
    ax.set_ylabel("demand coverage (%)", color=C_MILP)
    ax.tick_params(axis="y", labelcolor=C_MILP)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.step(sweep.index, sweep["new_stores"], where="mid",
              color=C_EXISTING, linewidth=1.6, alpha=0.8, label="new stores")
    twin.set_ylabel("new stores opened", color=C_EXISTING)
    twin.tick_params(axis="y", labelcolor=C_EXISTING)

    if breakeven is not None:
        ax.axvline(breakeven, color="#888", linestyle="--", linewidth=1)
        ax.annotate(
            f"break-even ≈ {breakeven:.0f}\n(Rs 2.5 cr buying ~1,800 orders/day)",
            xy=(breakeven, ax.get_ylim()[0]),
            xytext=(6, 14), textcoords="offset points",
            fontsize=8, color="#555",
        )

    ax.set_title("Sensitivity to the capital weight")
    fig.tight_layout()
    return fig


def plot_scenarios(scenarios: pd.DataFrame):
    """Served versus unmet demand as festive load rises.

    Stacking served and unserved makes the ceiling visible: the served portion
    stops growing while the total keeps rising, which is capacity binding
    rather than budget.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))

    labels = [f"{m:g}x" for m in scenarios.index]
    served = scenarios["demand_served"]
    unserved = scenarios["unserved"]

    left.bar(labels, served, color=C_MILP, label="served")
    left.bar(labels, unserved, bottom=served, color=C_UNUSED, label="unmet")
    left.set_ylabel("orders per day")
    left.set_title("Demand served against demand lost")
    left.legend(fontsize=9)
    left.grid(axis="y", alpha=0.2)
    left.set_axisbelow(True)

    right.plot(scenarios.index, scenarios["coverage_pct"],
               marker="o", color=C_MILP, linewidth=2)
    right.set_xlabel("demand multiplier")
    right.set_ylabel("coverage (%)")
    right.set_title("Coverage collapses under festive load")
    right.grid(alpha=0.2)
    right.set_axisbelow(True)
    for multiplier, coverage in scenarios["coverage_pct"].items():
        right.annotate(f"{coverage:.1f}%", (multiplier, coverage),
                       xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=9)

    fig.tight_layout()
    return fig
