"""Distances, facilities, and the feasibility mask.

Everything spatial lives here so that the baseline and the optimizer work from
exactly the same geometry. If they disagreed about which locality-site pairs
were within range, the comparison between them would be meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

# The delivery radius. Zepto disclosed an average delivery distance of roughly
# 1.7-1.8 km for FY26, so 2.5 km is a plausible outer bound for a ten-minute
# promise rather than an invented number: most deliveries fall well inside it,
# and the tail reaches out to about this.
DEFAULT_RADIUS_KM = 2.5


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres, vectorised over numpy arrays.

    Latitude and longitude are angles on a sphere, not coordinates on a plane,
    so Pythagoras does not apply directly. A degree of latitude is about 111 km
    everywhere, but a degree of longitude shrinks towards the poles by a factor
    of cos(latitude) -- roughly 0.92 at Indore's latitude. Treating lat/lon as
    planar therefore overstates east-west distances by about eight percent here,
    which is enough to put sites inside a 2.5 km radius that do not belong
    there.

    This is the ASIN form of haversine rather than the shorter spherical law of
    cosines. For points a few kilometres apart the cosine of the subtended angle
    sits a hair below 1.0, and in floating point it can round to just above it,
    which takes ACOS outside its domain and yields NaN. The half-angle sines
    used here stay well inside their domain at short range, which is the only
    range that matters for this problem.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_facilities(
    candidate_sites: pd.DataFrame, existing_stores: pd.DataFrame
) -> pd.DataFrame:
    """Combine already-open stores and candidate sites into one table.

    Both models reason about a single list of facilities, each flagged as
    already open or still a decision. Existing stores carry a capex of zero --
    that money is already spent, and charging it again against this year's
    budget would be a textbook sunk-cost error. It would also distort the
    comparison, because it would make the incumbent network look expensive
    relative to the new sites being considered.
    """
    existing = pd.DataFrame(
        {
            "facility_key": "E" + existing_stores["store_id"].astype(str),
            "source_id": existing_stores["store_id"].astype(int),
            "locality_id": existing_stores["locality_id"].astype(int),
            "latitude": existing_stores["latitude"].astype(float),
            "longitude": existing_stores["longitude"].astype(float),
            "capacity": existing_stores["max_capacity_orders_per_day"].astype(float),
            "capex": 0.0,
            "is_existing": True,
        }
    )

    candidates = pd.DataFrame(
        {
            "facility_key": "S" + candidate_sites["site_id"].astype(str),
            "source_id": candidate_sites["site_id"].astype(int),
            "locality_id": candidate_sites["locality_id"].astype(int),
            "latitude": candidate_sites["latitude"].astype(float),
            "longitude": candidate_sites["longitude"].astype(float),
            "capacity": candidate_sites["max_capacity_orders_per_day"].astype(float),
            "capex": candidate_sites["setup_capex"].astype(float),
            "is_existing": False,
        }
    )

    return pd.concat([existing, candidates], ignore_index=True)


def distance_matrix(localities: pd.DataFrame, facilities: pd.DataFrame) -> np.ndarray:
    """Distance in km from every locality to every facility, shape (n_loc, n_fac).

    Broadcasting does the work: reshaping the locality coordinates to a column
    and leaving the facility coordinates as a row makes numpy evaluate all
    n_loc x n_fac pairs in one vectorised pass, with no Python loop.
    """
    lat_l = localities["latitude"].to_numpy(dtype=float)[:, None]
    lon_l = localities["longitude"].to_numpy(dtype=float)[:, None]
    lat_f = facilities["latitude"].to_numpy(dtype=float)[None, :]
    lon_f = facilities["longitude"].to_numpy(dtype=float)[None, :]
    return haversine_km(lat_l, lon_l, lat_f, lon_f)


def feasible_mask(distances: np.ndarray, radius_km: float = DEFAULT_RADIUS_KM) -> np.ndarray:
    """Boolean matrix: which locality-facility pairs are servable at all.

    This is the delivery-radius constraint, and expressing it as a mask rather
    than as a constraint inside the model is a meaningful optimization. A pair
    outside the radius has no decision to make -- the answer is fixed at zero --
    so rather than creating a variable and then constraining it to zero, we
    never create the variable. On this dataset that drops the assignment
    variables from 35 x 56 = 1,960 down to roughly a tenth of that. Fewer
    variables is a smaller search tree, and for a mixed-integer program the
    search tree is the whole cost.
    """
    return distances <= radius_km


def coverage_report(
    localities: pd.DataFrame, distances: np.ndarray, radius_km: float = DEFAULT_RADIUS_KM
) -> pd.DataFrame:
    """How many facilities each locality can reach, and whether any exist.

    Useful before modelling: a locality no facility can reach is demand the
    optimizer is structurally unable to serve no matter how large the budget.
    That is a property of the candidate site list, not a failure of the model,
    and it is far better to know it up front than to discover it while staring
    at an unexpectedly low coverage figure.
    """
    mask = feasible_mask(distances, radius_km)
    return pd.DataFrame(
        {
            "locality_id": localities["locality_id"].to_numpy(),
            "locality_name": localities["locality_name"].to_numpy(),
            "reachable_facilities": mask.sum(axis=1),
            "nearest_facility_km": distances.min(axis=1).round(2),
        }
    )
