"""Build and populate the synthetic Indore dataset.

Run once, from the project root:

    python generate_data.py

Writes to MySQL if DATABASE_URL is set, otherwise to data/blinkit_sim.db.
The run is fully reproducible: same seed in, same database out.

EVERYTHING IN HERE IS SYNTHETIC. No real Blinkit order, rent or store data was
used. The generator is *calibrated* so that aggregates land near Blinkit's
publicly disclosed figures (~Rs 2.5 crore capex per store, ~Rs 11 lakh NOV per
store per day), but the row-level data is invented.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.db import (  # noqa: E402
    apply_schema,
    backend_name,
    describe_connection,
    ensure_database_exists,
    get_engine,
)

SEED = int(os.getenv("SEED", "42"))

# 90 days ending just after Diwali 2025 (20 October), so the data contains a
# real festive peak rather than an arbitrary one.
DATE_START = "2025-09-01"
N_DAYS = 90
DIWALI = pd.Timestamp("2025-10-20")

N_CANDIDATE_SITES = 50
N_EXISTING_STORES = 6

# --- Calibration anchors -----------------------------------------------------
# Disclosed:  ~Rs 2.5 crore capex per store, ~Rs 11 lakh NOV per store per day.
# Assumed:    average order value by income tier (see AOV_BY_TIER below).
# Derived:    capacity ~= 11,00,000 / 600 ~= 1,800 orders per store per day.
#             That figure inherits the AOV assumption and is NOT a disclosed
#             fact. Capacities are spread either side of it.
CAPEX_MEAN = 2.5e7          # Rs 2.5 crore
CAPEX_MIN = 1.5e7           # small store
CAPEX_MAX = 3.5e7           # warehouse-heavy store
CAPACITY_MIN = 1_200
CAPACITY_MAX = 2_400

AOV_BY_TIER = {"low": 420.0, "mid": 560.0, "high": 720.0}

# Indore city centre, for reference.
CITY_LAT, CITY_LON = 22.7196, 75.8577


# ---------------------------------------------------------------------------
# Localities
# ---------------------------------------------------------------------------
# Real Indore neighbourhood names with approximate coordinates. Approximate is
# fine and is labelled as such: the model needs plausible relative geometry,
# not survey-grade positions.
LOCALITIES: list[tuple[str, float, float]] = [
    ("Rajwada",           22.7177, 75.8545),
    ("Sarafa Bazar",      22.7161, 75.8562),
    ("Juni Indore",       22.7100, 75.8500),
    ("Chhoti Gwaltoli",   22.7215, 75.8650),
    ("Sanyogitaganj",     22.7180, 75.8690),
    ("Malviya Nagar",     22.7200, 75.8700),
    ("Navlakha",          22.7025, 75.8766),
    ("Geeta Bhawan",      22.7158, 75.8823),
    ("Manorama Ganj",     22.7192, 75.8799),
    ("New Palasia",       22.7266, 75.8812),
    ("Old Palasia",       22.7221, 75.8865),
    ("Palasia",           22.7244, 75.8839),
    ("Tilak Nagar",       22.7071, 75.8901),
    ("Saket Nagar",       22.7255, 75.8950),
    ("Ring Road",         22.7290, 75.9002),
    ("Khajrana",          22.7345, 75.9057),
    ("Bengali Square",    22.7412, 75.9155),
    ("Kanadia Road",      22.7180, 75.9150),
    ("Bicholi Mardana",   22.7455, 75.9270),
    ("Nipania",           22.7702, 75.9200),
    ("Vijay Nagar",       22.7532, 75.8937),
    ("Scheme 54",         22.7502, 75.8863),
    ("Scheme 78",         22.7580, 75.8790),
    ("MIG Colony",        22.7398, 75.8875),
    ("LIG Colony",        22.7370, 75.8814),
    ("Sukhliya",          22.7600, 75.8650),
    ("Dewas Naka",        22.7700, 75.8800),
    ("Mhow Naka",         22.6975, 75.8484),
    ("Bhawarkuan",        22.6934, 75.8608),
    ("Musakhedi",         22.6900, 75.8900),
    ("Pipliyahana",       22.7050, 75.9000),
    ("Annapurna",         22.6869, 75.8385),
    ("Sudama Nagar",      22.6820, 75.8254),
    ("Silicon City",      22.6740, 75.8800),
    ("Rau",               22.6206, 75.8083),
]

# Density cores: (lat, lon, peak people/sq km, decay km).
# Real cities are not uniform. Density peaks around a few centres and falls off
# with distance, which is what makes demand SPATIALLY CORRELATED — neighbouring
# localities resemble each other. Pure random density would look like static
# when plotted and would make the optimizer's job meaningless, because there
# would be no structure to exploit.
DENSITY_CORES = [
    (22.7177, 75.8545, 21_000, 2.2),   # Rajwada - dense old city
    (22.7244, 75.8839, 14_000, 2.6),   # Palasia - central commercial
    (22.7532, 75.8937, 11_000, 3.2),   # Vijay Nagar - newer commercial
    (22.6934, 75.8608,  8_000, 2.8),   # Bhawarkuan - student belt
]
DENSITY_FLOOR = 1_800

# Affluence cores, separate from density: the richest areas are not the densest.
AFFLUENCE_CORES = [
    (22.7532, 75.8937, 1.00, 3.0),     # Vijay Nagar
    (22.7266, 75.8812, 0.90, 2.4),     # New Palasia
    (22.7580, 75.8790, 0.75, 3.0),     # Scheme 78
    (22.7702, 75.9200, 0.70, 3.2),     # Nipania
]


def km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Rough planar distance in km. Fine at city scale; see src/geo.py for the
    haversine version used in the model itself."""
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot(dlat, dlon))


def _core_field(lat: float, lon: float, cores) -> float:
    """Sum of Gaussian bumps from each core — the mechanism that creates
    spatial correlation."""
    total = 0.0
    for core_lat, core_lon, peak, decay in cores:
        d = km_between(lat, lon, core_lat, core_lon)
        total += peak * np.exp(-((d / decay) ** 2))
    return total


def build_localities(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for idx, (name, lat, lon) in enumerate(LOCALITIES, start=1):
        density = DENSITY_FLOOR + _core_field(lat, lon, DENSITY_CORES)
        density *= rng.normal(1.0, 0.08)          # local variation
        affluence = _core_field(lat, lon, AFFLUENCE_CORES) + rng.normal(0, 0.06)
        rows.append(
            {
                "locality_id": idx,
                "locality_name": name,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "population_density": round(max(density, 500.0), 2),
                "_affluence": affluence,
            }
        )

    df = pd.DataFrame(rows)

    # Tier by rank, so we always get a sensible spread rather than whatever the
    # raw scores happen to produce.
    df["avg_household_income_tier"] = pd.qcut(
        df["_affluence"].rank(method="first"),
        q=[0, 0.35, 0.72, 1.0],
        labels=["low", "mid", "high"],
    ).astype(str)

    return df


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def build_demand(localities: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(DATE_START, periods=N_DAYS, freq="D")

    # Quick-commerce penetration rises with affluence: denser AND richer areas
    # order more per head, not just more in total.
    tier_uplift = {"low": 0.75, "mid": 1.00, "high": 1.45}

    # Scale chosen so total city demand lands near 32,000 orders/day. At the
    # derived ~1,800 orders/store/day that implies roughly 18 stores to serve
    # the city fully — so a 6-store incumbent base leaves a real decision to
    # make, and the budget cannot simply cover everything.
    base_scale = 0.307

    rows = []
    for loc in localities.itertuples():
        base = (
            base_scale
            * (loc.population_density ** 0.82)
            * tier_uplift[loc.avg_household_income_tier]
        )
        aov_base = AOV_BY_TIER[loc.avg_household_income_tier]

        for date in dates:
            # Weekly pattern: quiet Monday, busy Friday and weekend.
            dow = date.dayofweek                      # 0 = Monday
            weekly = {0: 0.93, 1: 0.95, 2: 0.97, 3: 1.00, 4: 1.09, 5: 1.21, 6: 1.16}[dow]

            # Festive: ramp into Diwali, sharp peak, quick fall. Mirrors the
            # capacity strain Blinkit has publicly described around Diwali.
            days_to_diwali = (date - DIWALI).days
            if -5 <= days_to_diwali <= 2:
                festive = 1.0 + 1.15 * np.exp(-((days_to_diwali + 1.5) ** 2) / 4.5)
            else:
                festive = 1.0

            mean_orders = base * weekly * festive

            # Poisson: order counts are non-negative integers generated by many
            # independent customers. Its variance equals its mean, so busy
            # localities are noisier in absolute terms and quieter ones are not
            # forced negative — which a Gaussian would do.
            order_count = int(rng.poisson(mean_orders))

            # Basket sizes rise slightly at the festive peak.
            aov = aov_base * (1.0 + 0.10 * (festive - 1.0)) * rng.normal(1.0, 0.035)

            rows.append(
                {
                    "locality_id": loc.locality_id,
                    "order_date": date.date(),
                    "order_count": order_count,
                    "avg_order_value": round(float(max(aov, 80.0)), 2),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate sites and existing stores
# ---------------------------------------------------------------------------
def build_candidate_sites(
    localities: pd.DataFrame, demand: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    avg_demand = demand.groupby("locality_id")["order_count"].mean()

    # Property developers chase demand, so candidate sites cluster where the
    # orders are — but every locality gets at least one, otherwise the optimizer
    # is never offered the option of covering a quiet area.
    weights = (avg_demand / avg_demand.sum()).reindex(localities["locality_id"]).values
    extra = N_CANDIDATE_SITES - len(localities)
    chosen = rng.choice(localities["locality_id"].values, size=extra, p=weights, replace=True)
    site_localities = np.concatenate([localities["locality_id"].values, chosen])

    loc_by_id = localities.set_index("locality_id")

    rows = []
    for site_id, locality_id in enumerate(sorted(site_localities), start=1):
        loc = loc_by_id.loc[locality_id]

        # Jitter within roughly 1 km of the locality centroid.
        lat = float(loc["latitude"]) + rng.normal(0, 0.006)
        lon = float(loc["longitude"]) + rng.normal(0, 0.006)

        # Capex: lognormal so it is right-skewed (a few expensive sites, most
        # near the middle) rather than symmetric, then clipped to the disclosed
        # plausible band.
        capex = float(np.clip(rng.lognormal(np.log(CAPEX_MEAN) - 0.02, 0.20),
                              CAPEX_MIN, CAPEX_MAX))

        # Bigger spend buys more throughput, with noise — larger sites hold more
        # inventory and support more pickers.
        capex_position = (capex - CAPEX_MIN) / (CAPEX_MAX - CAPEX_MIN)
        capacity = CAPACITY_MIN + capex_position * (CAPACITY_MAX - CAPACITY_MIN)
        capacity = int(np.clip(capacity * rng.normal(1.0, 0.07),
                               CAPACITY_MIN, CAPACITY_MAX))

        # Rent tracks density and affluence, independent of capex.
        tier_rent = {"low": 0.85, "mid": 1.00, "high": 1.35}[loc["avg_household_income_tier"]]
        rent = 55.0 * float(loc["population_density"]) ** 0.55 * tier_rent
        rent = float(rent * rng.normal(1.0, 0.12))

        rows.append(
            {
                "site_id": site_id,
                "locality_id": int(locality_id),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "estimated_rent_monthly": round(max(rent, 25_000.0), 2),
                "max_capacity_orders_per_day": capacity,
                "setup_capex": round(capex, 2),
            }
        )

    return pd.DataFrame(rows)


def build_existing_stores(
    localities: pd.DataFrame, demand: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Stores already trading.

    Placed in the highest-demand localities, because that is where a real
    operator would have expanded first. This matters: it means the optimizer
    inherits a sensible network and has to find *incremental* value, which is a
    much harder and more realistic test than starting from an empty city.
    """
    avg_demand = demand.groupby("locality_id")["order_count"].mean().sort_values(ascending=False)
    top = avg_demand.head(N_EXISTING_STORES).index.tolist()
    loc_by_id = localities.set_index("locality_id")

    rows = []
    for store_id, locality_id in enumerate(top, start=1):
        loc = loc_by_id.loc[locality_id]
        rows.append(
            {
                "store_id": store_id,
                "locality_id": int(locality_id),
                "latitude": round(float(loc["latitude"]) + rng.normal(0, 0.004), 6),
                "longitude": round(float(loc["longitude"]) + rng.normal(0, 0.004), 6),
                "max_capacity_orders_per_day": int(rng.integers(1_500, 2_100)),
                "opened_on": pd.Timestamp("2024-01-01").date(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
def sanity_check(demand: pd.DataFrame, sites: pd.DataFrame) -> None:
    """Verify the synthetic data lands near the disclosed figures.

    This is the difference between "numbers that look plausible" and "numbers
    calibrated to something real". If a change to the generator pushes the data
    somewhere implausible, this fails loudly instead of quietly producing a
    project whose scale cannot be defended.
    """
    daily_city = demand.groupby("order_date")["order_count"].sum()
    median_daily = float(daily_city.median())

    mean_capex_cr = float(sites["setup_capex"].mean()) / 1e7
    mean_capacity = float(sites["max_capacity_orders_per_day"].mean())
    mean_aov = float(demand["avg_order_value"].mean())

    # Implied NOV per store per day at full utilisation, against the disclosed
    # ~Rs 11 lakh target.
    implied_nov_lakh = mean_capacity * mean_aov / 1e5

    assert 2.2 <= mean_capex_cr <= 2.8, f"mean capex {mean_capex_cr:.2f} cr off target 2.5"
    assert 8.0 <= implied_nov_lakh <= 14.0, f"implied NOV {implied_nov_lakh:.2f} lakh off target 11"
    assert 20_000 <= median_daily <= 45_000, f"city demand {median_daily:.0f}/day implausible"

    print("\nCalibration against disclosed figures")
    print(f"  mean capex per site        Rs {mean_capex_cr:.2f} cr        (disclosed ~2.5 cr)")
    print(f"  implied NOV per store/day  Rs {implied_nov_lakh:.2f} lakh     (disclosed ~11 lakh)")
    print(f"  median city demand         {median_daily:,.0f} orders/day  (derived, not disclosed)")
    print(f"  mean AOV                   Rs {mean_aov:.0f}            (assumed, not disclosed)")


def main() -> None:
    rng = np.random.default_rng(SEED)

    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        ensure_database_exists(url)

    engine = get_engine()
    print(f"Backend: {backend_name(engine)}  ({describe_connection(engine)})")
    print(f"Seed:    {SEED}")

    localities = build_localities(rng)
    demand = build_demand(localities, rng)
    sites = build_candidate_sites(localities, demand, rng)
    stores = build_existing_stores(localities, demand, rng)

    apply_schema(engine)

    localities_out = localities.drop(columns=["_affluence"])
    with engine.begin() as conn:
        localities_out.to_sql("localities", conn, if_exists="append", index=False)
        demand.to_sql("demand", conn, if_exists="append", index=False, chunksize=1_000)
        sites.to_sql("candidate_sites", conn, if_exists="append", index=False)
        stores.to_sql("existing_stores", conn, if_exists="append", index=False)

    print("\nRows written")
    print(f"  localities       {len(localities_out):>6,}")
    print(f"  demand           {len(demand):>6,}   ({N_DAYS} days x {len(localities_out)} localities)")
    print(f"  candidate_sites  {len(sites):>6,}")
    print(f"  existing_stores  {len(stores):>6,}")

    sanity_check(demand, sites)
    print("\nDone.")


if __name__ == "__main__":
    main()
