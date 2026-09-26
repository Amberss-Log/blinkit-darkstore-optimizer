"""The mixed-integer program.

This is the core of the project. Where the greedy baseline commits to one site
at a time and never reconsiders, this formulates the whole decision as a single
mathematical problem and hands it to a solver that searches combinations.

The practical difference is a guarantee. When CBC returns status "Optimal" it
is not reporting that it tried hard and this was the best it found -- it is
asserting that no better solution exists within the constraints as written.
That is something no heuristic can offer, and it is the honest answer to "how
do you know this is right".
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pulp

from src.evaluate import NetworkResult

# Capex is divided by this before entering the objective, so it is expressed in
# crore rather than rupees. See the note on units below -- this single line is
# what makes lambda_capex interpretable.
CRORE = 1e7

# Default weights.
#
# LAMBDA_CAPEX is in orders/day per crore: it states how much demand throughput
# one crore of capital has to buy before the spend is worth making. The natural
# reference point comes from the disclosed figures -- a store costing ~Rs 2.5 cr
# and serving ~1,800 orders/day implies roughly 720 orders per crore at
# break-even. Setting the default well below that (200) means the model is
# mildly capital-disciplined but still expansionary, which matches a business
# adding stores every quarter. The notebook sweeps this.
#
# LAMBDA_DISTANCE is in orders/day per km-order: the cost of routing one order
# one kilometre. At 0.05 an order delivered from 2 km away counts for 0.9 of an
# order delivered from next door, which is enough to break ties toward closer
# stores without letting distance override coverage.
LAMBDA_CAPEX = 200.0
LAMBDA_DISTANCE = 0.05


def optimize_network(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    lambda_capex: float = LAMBDA_CAPEX,
    lambda_distance: float = LAMBDA_DISTANCE,
    max_new_stores: int | None = None,
    relax_binaries: bool = False,
    verbose: bool = False,
) -> NetworkResult:
    """Choose which sites to open and how to route demand, optimally.

    Set relax_binaries to solve the LP relaxation instead -- the same model with
    the open/closed decisions allowed to take fractional values. That solution
    is meaningless as a plan (you cannot open 0.4 of a store) but it is exactly
    the bound the solver prunes against internally, and comparing it to the
    integer optimum shows what the indivisibility actually costs.
    """
    start = time.perf_counter()

    n_loc = len(localities)
    capacity = facilities["capacity"].to_numpy(dtype=float)
    capex = facilities["capex"].to_numpy(dtype=float)
    is_existing = facilities["is_existing"].to_numpy(dtype=bool)
    n_fac = len(facilities)

    problem = pulp.LpProblem("dark_store_network", pulp.LpMaximize)

    # -- decision variables -------------------------------------------------

    # Open/closed, for candidate sites only. Existing stores are not decisions:
    # they are already trading, so they enter the model as constants. Creating
    # variables for them and fixing those variables to 1 would work identically
    # but would hand the solver 6 more binaries to branch on for no reason.
    var_type = "Continuous" if relax_binaries else "Binary"
    x = {
        j: pulp.LpVariable(f"open_{facilities.loc[j, 'facility_key']}",
                           lowBound=0, upBound=1, cat=var_type)
        for j in range(n_fac)
        if not is_existing[j]
    }

    def is_open(j):
        """Openness of facility j as something the solver can use.

        Returns the literal 1 for existing stores and the decision variable for
        candidates, so the constraints below can be written once for both.
        """
        return 1 if is_existing[j] else x[j]

    # Assignment, created only over pairs the radius admits. A pair outside the
    # radius has a known answer -- zero -- so creating a variable and then
    # constraining it to zero would ask the solver to rediscover something we
    # already know, at the cost of a larger search tree. On this dataset this
    # is the difference between ~500 variables and 1,960.
    pairs = [(i, j) for i in range(n_loc) for j in range(n_fac) if mask[i, j]]
    s = {
        (i, j): pulp.LpVariable(f"serve_{i}_{j}", lowBound=0, cat="Continuous")
        for i, j in pairs
    }

    # Unserved demand. This is what makes coverage a soft constraint: demand
    # that cannot be served profitably flows here instead of making the whole
    # problem infeasible. A model that returns "no solution exists" tells you
    # nothing about which constraint was binding; a model that returns 30%
    # unserved demand tells you exactly where you are short.
    u = {
        i: pulp.LpVariable(f"unserved_{i}", lowBound=0, cat="Continuous")
        for i in range(n_loc)
    }

    # -- objective ----------------------------------------------------------
    #
    # maximize:  demand served
    #            - lambda_capex    * capital spent, in crore
    #            - lambda_distance * (distance x demand served)
    #
    # The units matter more than they look. Demand is orders/day, capex is
    # divided down to crore, distance is km. Without that normalisation the
    # capex term would be measured in tens of millions while the demand term
    # sits in the tens of thousands, and any weight that balanced them would be
    # an arbitrary number with no meaning. Normalised, lambda_capex reads as
    # "orders per day that one crore is worth", which is a quantity you can
    # argue about with a finance team.
    #
    # There is deliberately no third weight on unserved demand. The balance
    # constraint below forces served + unserved to equal total demand, and
    # total demand is a constant, so maximising served is algebraically the
    # same as minimising unserved. A separate penalty would only rescale the
    # other two terms while appearing to add a degree of freedom.
    served_total = pulp.lpSum(s.values())
    capex_total = pulp.lpSum(
        (capex[j] / CRORE) * x[j] for j in x
    )
    distance_total = pulp.lpSum(
        distances[i, j] * s[(i, j)] for (i, j) in pairs
    )

    problem += (
        served_total
        - lambda_capex * capex_total
        - lambda_distance * distance_total
    ), "net_value"

    # -- constraints --------------------------------------------------------

    # Budget. Only candidate sites appear: existing stores are sunk cost, and
    # charging them against this year's capital would mean paying twice.
    problem += capex_total <= budget / CRORE, "budget"

    # Capacity, with the linking to the open decision. For a candidate site the
    # right-hand side is capacity * x[j], which collapses to zero when the site
    # is closed -- so a closed store cannot be assigned any demand, and the two
    # decisions stay consistent. This is the constraint that couples the
    # location problem to the assignment problem; without it they would be two
    # independent and much easier problems.
    for j in range(n_fac):
        incoming = pulp.lpSum(s[(i, j)] for i in range(n_loc) if mask[i, j])
        problem += incoming <= capacity[j] * is_open(j), f"capacity_{j}"

    # Demand balance. An accounting identity rather than a business rule: every
    # order is either served by some facility or explicitly counted as lost.
    # Written as an equality precisely so that demand cannot quietly evaporate
    # somewhere in the middle of the model.
    for i in range(n_loc):
        outgoing = pulp.lpSum(s[(i, j)] for j in range(n_fac) if mask[i, j])
        problem += outgoing + u[i] == demand[i], f"demand_balance_{i}"

    # Operational cap on simultaneous openings -- hiring and training are real
    # constraints that money alone does not relieve.
    if max_new_stores is not None:
        problem += pulp.lpSum(x.values()) <= max_new_stores, "max_new_stores"

    # -- solve --------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    open_flags = is_existing.copy()
    assignment = np.zeros((n_loc, n_fac), dtype=float)

    if status == "Optimal":
        for j, var in x.items():
            # Binaries come back as 0.9999999 or 1.0000001 often enough that
            # comparing to 1 exactly is a real source of bugs.
            open_flags[j] = var.value() is not None and var.value() > 0.5
        for (i, j), var in s.items():
            value = var.value()
            if value:
                assignment[i, j] = value

    notes = {
        "lambda_capex": lambda_capex,
        "lambda_distance": lambda_distance,
        "budget": budget,
        "n_binary_vars": 0 if relax_binaries else len(x),
        "n_continuous_vars": len(s) + len(u) + (len(x) if relax_binaries else 0),
        "n_constraints": len(problem.constraints),
        "feasible_pairs": len(pairs),
        "objective_value": pulp.value(problem.objective) if status == "Optimal" else None,
        "relaxed": relax_binaries,
        # Raw value of each open/closed variable, keyed by site. For the integer
        # model these are all 0 or 1. For the relaxation they can be fractional
        # -- "open 0.4 of this store" -- which is what makes it interesting to
        # inspect, and what any round-the-relaxation shortcut has to deal with.
        "open_values": {
            str(facilities.loc[j, "facility_key"]): float(var.value() or 0.0)
            for j, var in x.items()
        } if status == "Optimal" else {},
    }

    if verbose:
        print(f"status {status}  |  {notes['n_binary_vars']} binary, "
              f"{notes['n_continuous_vars']} continuous vars, "
              f"{notes['n_constraints']} constraints")

    return NetworkResult(
        method="milp-relaxed" if relax_binaries else "milp",
        open_flags=open_flags,
        assignment=assignment,
        facilities=facilities,
        localities=localities,
        distances=distances,
        demand=demand,
        status=status,
        solve_seconds=time.perf_counter() - start,
        notes=notes,
    )


def relaxation_gap(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    **kwargs,
) -> dict:
    """Compare the integer optimum against its LP relaxation.

    Every branch-and-bound solver works by repeatedly solving the relaxation --
    the same problem with the integrality requirement dropped -- because that is
    fast and its objective is an upper bound on any integer solution. The solver
    then branches, forcing a fractional variable to 0 in one subproblem and 1 in
    another, and prunes any branch whose bound is already worse than the best
    integer solution found so far.

    The gap between the two is therefore the price of indivisibility: how much
    value is lost because stores must be opened whole. A small gap means the
    combinatorial part of the problem is nearly free and a rounded relaxation
    would have been close; a large gap means the integer structure genuinely
    matters and the branching earned its keep.
    """
    integer = optimize_network(
        localities, facilities, distances, mask, demand, budget,
        relax_binaries=False, **kwargs
    )
    relaxed = optimize_network(
        localities, facilities, distances, mask, demand, budget,
        relax_binaries=True, **kwargs
    )

    z_int = integer.notes["objective_value"]
    z_lp = relaxed.notes["objective_value"]
    gap_pct = 100.0 * (z_lp - z_int) / abs(z_lp) if z_lp else float("nan")

    return {
        "integer_objective": z_int,
        "relaxed_objective": z_lp,
        "gap_pct": gap_pct,
        "integer_coverage_pct": integer.coverage_pct,
        "relaxed_coverage_pct": relaxed.coverage_pct,
        "integer_result": integer,
        "relaxed_result": relaxed,
    }


def sensitivity_sweep(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    lambda_values: list[float],
    lambda_distance: float = LAMBDA_DISTANCE,
) -> pd.DataFrame:
    """Re-solve across a range of capital weights.

    The point is not the curve but the plateau. A recommendation that holds
    across a wide band of lambda_capex is one you can act on, because it does
    not depend on a judgment call nobody can verify. One that flips as soon as
    the weight moves is an artefact of the weight, and reporting it as a finding
    would be a mistake.
    """
    rows = []
    for lam in lambda_values:
        result = optimize_network(
            localities, facilities, distances, mask, demand, budget,
            lambda_capex=lam, lambda_distance=lambda_distance,
        )
        summary = result.summary()
        summary["lambda_capex"] = lam
        summary["sites"] = ",".join(sorted(result.opened_sites()["facility_key"]))
        rows.append(summary)

    return pd.DataFrame(rows).set_index("lambda_capex")


def scenario_analysis(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    multipliers: list[float] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Re-solve the same model under festive demand multipliers.

    Blinkit has publicly described the capacity strain it faces around Diwali,
    and the generated data reproduces it: the festive peak in this dataset
    reaches about 2.5 times the median day. So the top of this sweep is
    grounded in the data rather than chosen for effect.

    Nothing about the model changes here -- same objective, same constraints,
    same budget. Only the demand vector is scaled. That is the point: a model
    worth building is one you can re-point at a different question without
    rewriting it, and stress testing is the cheapest second result available.

    Watch which constraint binds as the multiplier rises. At normal demand the
    budget is the limit: there is more capital available than there is demand
    worth serving. Under festive load capacity becomes the limit instead, and
    unmet demand grows even though the plan is unchanged -- which is precisely
    the operational problem quick commerce faces in October.
    """
    if multipliers is None:
        multipliers = [1.0, 1.5, 2.0, 2.5]

    rows = []
    for multiplier in multipliers:
        scaled = demand * multiplier
        result = optimize_network(
            localities, facilities, distances, mask, scaled, budget, **kwargs
        )
        summary = result.summary()
        summary["multiplier"] = multiplier
        summary["total_demand"] = round(result.total_demand)
        summary["unserved"] = round(result.demand_unserved)
        summary["sites"] = ",".join(sorted(result.opened_sites()["facility_key"]))
        rows.append(summary)

    columns = ["multiplier", "total_demand", "new_stores", "capex_cr",
               "demand_served", "unserved", "coverage_pct", "avg_delivery_km",
               "capacity_util_pct", "sites"]
    return pd.DataFrame(rows)[columns].set_index("multiplier")


def solve_fixed_network(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    open_keys: set[str],
    **kwargs,
) -> NetworkResult:
    """Route demand optimally through a network whose stores are already chosen.

    Useful for scoring a plan that came from somewhere other than the MILP --
    a rounded LP relaxation, say -- on equal terms. The chosen candidate sites
    are treated as already open and the rest are removed, so the solver has no
    siting decision left to make and only works out the best routing.

    The returned result carries the true capex of the chosen sites, not zero,
    so it can be compared directly against the budget.
    """
    keep = (facilities["is_existing"] | facilities["facility_key"].isin(open_keys)).to_numpy()
    fixed = facilities[keep].copy().reset_index(drop=True)
    true_capex = fixed["capex"].copy()
    fixed["is_existing"] = True
    fixed["capex"] = 0.0

    result = optimize_network(
        localities, fixed, distances[:, keep], mask[:, keep], demand, budget=0.0, **kwargs
    )

    # Restore the real facility table so capex and "new store" counts are
    # reported against what the plan would actually cost.
    restored = fixed.copy()
    restored["capex"] = true_capex
    restored["is_existing"] = ~restored["facility_key"].isin(open_keys)
    result.facilities = restored
    result.method = "fixed"
    return result


def round_relaxation(
    localities: pd.DataFrame,
    facilities: pd.DataFrame,
    distances: np.ndarray,
    mask: np.ndarray,
    demand: np.ndarray,
    budget: float,
    **kwargs,
) -> pd.DataFrame:
    """Solve the LP relaxation, round it three ways, and score each result.

    Tests the tempting shortcut: if the relaxation scores within a fraction of
    a percent of the integer optimum, why not solve the easy fractional
    version and round it? Because a close objective says nothing about
    whether a *rounded plan* is any good. The relaxation spreads spending
    thinly across partly-opened sites, and every obvious rounding rule either
    wastes budget or breaks it.
    """
    relaxed = optimize_network(localities, facilities, distances, mask, demand, budget,
                               relax_binaries=True, **kwargs)
    integer = optimize_network(localities, facilities, distances, mask, demand, budget, **kwargs)
    values = relaxed.notes["open_values"]

    plans = {
        "MILP": set(integer.opened_sites()["facility_key"]),
        "LP, round down": {k for k, v in values.items() if v >= 1 - 1e-6},
        "LP, round to nearest": {k for k, v in values.items() if v >= 0.5},
        "LP, round up": {k for k, v in values.items() if v > 1e-6},
    }

    rows = []
    for name, keys in plans.items():
        scored = solve_fixed_network(localities, facilities, distances, mask, demand, keys, **kwargs)
        spend = float(facilities.loc[facilities["facility_key"].isin(keys), "capex"].sum())
        rows.append({
            "plan": name,
            "new_stores": len(keys),
            "capex_cr": round(spend / 1e7, 2),
            "within_budget": spend <= budget + 1e-6,
            "coverage_pct": round(scored.coverage_pct, 1),
        })

    fractional = {k: round(v, 3) for k, v in values.items() if 1e-6 < v < 1 - 1e-6}
    table = pd.DataFrame(rows).set_index("plan")
    table.attrs["fractional_sites"] = fractional
    return table
