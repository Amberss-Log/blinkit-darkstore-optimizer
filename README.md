# Blinkit dark-store network optimizer

Where should a quick-commerce operator open its next dark stores in a city, given a fixed capital budget?
Blinkit ran 2,443 dark stores as of June 2026 and keeps adding more at roughly ₹2.5 crore of capex each, so
store placement is a capital allocation decision made under three simultaneous limits: the budget, a delivery
radius the ten-minute promise cannot survive breaking, and a ceiling on how many orders one store can pick and
pack in a day. This project models that decision for Indore as a **capacitated facility location problem**,
solves it as a mixed-integer linear program with PuLP and CBC, and measures the result against the greedy
heuristic most teams would use informally. The comparison is the point — an optimizer with no benchmark
produces a number nobody can judge.

---

## What's real

These figures come from Eternal's public reporting and are used to calibrate the model. Nothing here is
extended beyond what was actually disclosed.

| Fact | Value | Source |
|---|---|---|
| Blinkit store count | 2,443 | Eternal Q1 FY27 (June 2026) |
| Capex per store, including warehousing | ~₹2.5 crore | Eternal, long-run assumption |
| Steady-state NOV per store per day | ~₹11 lakh | Eternal, steady-state target |
| Network working capital | ~12 days | Eternal |
| Target pre-tax ROCE | ~41.7% | Eternal, under the above assumptions |
| Network footprint | ~19 million sq ft across 300+ cities | Eternal |
| Inventory and spoilage loss | ~1.8% of NOV, "fairly asymptotic" | Eternal |
| Peer average delivery distance | ~1.7–1.8 km, FY26 | Zepto disclosure |

## What's synthetic

Everything below was invented for this project. **No real Blinkit order, rent, store-location or dispatch
data was used** — none is public at this granularity.

| Assumption | Value | Reasoning |
|---|---|---|
| City | Indore | Tier-2, thin quick-commerce presence, so expansion is a real question |
| Localities | 35, real names, approximate coordinates | Enough spatial structure to matter, small enough to solve in seconds |
| Demand history | 90 days, 1 Sep – 29 Nov 2025 | Contains Diwali (20 Oct 2025), so the festive peak is real rather than invented |
| Candidate sites | 50 | — |
| Existing stores | 6 | Makes the decision incremental, matching the real situation |
| Average order value | ~₹560 blended, varying by income tier | Assumed; drives the capacity derivation |
| Store capacity | ~1,200–2,400 orders/day | **Derived**, see below |
| Delivery radius | 2.5 km | Assumed, anchored on the ~1.7–1.8 km peer average |
| Budget | ₹20 crore | Assumed — roughly eight stores at the disclosed capex |

Demand is generated with spatial structure rather than drawn independently: density peaks around four real
focal points (Rajwada, Palasia, Vijay Nagar, Bhawarkuan) and decays with distance, so neighbouring localities
resemble each other. Affluence is generated as a separate field with its own centres, because the densest
areas of a real city are not the wealthiest. Order counts are Poisson, with a weekly cycle and a festive
window around Diwali. The generator asserts at the end that its aggregates land near the disclosed figures.

### One derivation worth stating explicitly

Store capacity is not disclosed anywhere. It is derived:

```
₹11,00,000 NOV per store per day  ÷  ~₹600 average order value  ≈  1,800 orders per store per day
```

The ₹11 lakh is disclosed. The ₹600 is an assumption. **So 1,800 is not a fact** — it inherits the weakest
link in its chain, and it moves substantially with the assumption (₹500 → ~2,200; ₹700 → ~1,570). Site
capacities in the data are spread across 1,200–2,400 rather than fixed at a spuriously precise 1,800.

---

## How to run

Requires Python 3.11+. MySQL is optional.

```bash
pip install -r requirements.txt

cp .env.example .env        # optional — see below
python generate_data.py     # builds and populates the database
python -m pytest            # 20 tests
jupyter lab notebook.ipynb  # the analysis
```

**Database.** The project runs on SQLite with no setup at all — leave `DATABASE_URL` unset and
`generate_data.py` creates `data/blinkit_sim.db` for you. To use MySQL 8 instead, set it in `.env`:

```
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/blinkit_sim
```

The database is created automatically if it does not exist. The five analytical queries are written to run
unchanged on both engines; the single place the dialects diverge — day-of-week extraction — is isolated in
`src/db.py`.

`.env` is gitignored and must never be committed.

### Layout

```
generate_data.py          builds and populates the database
notebook.ipynb            the analysis
sql/
  schema_mysql.sql        schema, MySQL 8
  schema_sqlite.sql       schema, SQLite
  queries/                the five analytical queries
src/
  db.py                   engine, dialect shim, schema loading
  geo.py                  haversine distance, feasibility mask
  evaluate.py             shared metrics, dispatch rule
  baseline.py             greedy heuristic
  optimizer.py            the MILP, sensitivity, scenarios
  forecast.py             ridge demand forecast
  plots.py                figures
tests/                    20 tests, including a hand-checkable toy MILP
```

---

## Results

Against a greedy baseline that respects every constraint the optimizer does — budget, capacity, radius and
the incumbent network — the MILP **opened 9 stores for ₹19.97 crore and served 75.8% of demand, against
greedy's 6 stores, ₹18.96 crore and 69.5%**: about **2,200 additional orders per day for the same capital**.

The mechanism is worth more than the number. Both networks run at 100% capacity utilisation, so coverage is
decided entirely by how much throughput the budget buys. Greedy took the individually-largest sites — ₹2.84 to
₹3.50 crore each — and stranded ₹1.04 crore it could not spend. The optimizer bought nine cheaper sites at
₹1.84 to ₹2.52 crore, more total capacity for the same money: 796 orders per crore against greedy's 722.
Greedy could not find this, because evaluating "which *combination* of sites buys the most capacity" is
precisely what a one-site-at-a-time heuristic cannot do.

Three further results:

**The recommendation is stable.** Sweeping the capital weight λ₁ from 0 to 600 orders per crore leaves the
answer unchanged at nine stores. It first moves at 720 — which is exactly where the analytical break-even sits
for a ₹2.5 crore store serving ~1,800 orders/day — and capital stops being deployed by 900.

**Festive demand cannot be bought away.** At 1.5×, 2× and 2.5× demand the plan does not change and demand
served does not move, because capacity is bought with capital and capital is fixed. Coverage falls from 75.8%
to 30.3%. The binding constraint flips from budget to capacity, which is an argument for temporary capacity
rather than permanent footprint.

**Better forecasting does not change the decision.** A ridge regression beats a seasonal-naive benchmark by
27% on mean absolute error, and re-running the optimizer on forecast demand selects *identical* sites. That is
a negative result and a useful one: the recommendation is robust to demand-estimate accuracy, so effort spent
improving the forecast would not have moved it.

---

## Limitations

The method is the contribution; the numbers are illustrative.

**The data is synthetic.** Demand is calibrated so aggregates land near disclosed figures, but locality-level
variation is invented. Every conclusion is a statement about the method applied to plausible data, not about
Indore.

**Distances are centroid-to-store and understate reality.** Each locality collapses to one point, so the model
reports ~0.63 km average delivery against a disclosed peer average of ~1.7–1.8 km. Real distance includes
travel within the locality and follows roads, not great circles. A road-network distance matrix would raise
every distance and shrink the feasible set.

**Capacity is a single daily figure.** Real stores have hourly peaks, and a store comfortable with 1,800
orders spread across a day may fail at the evening concentration.

**λ₁ and λ₂ are judgment calls.** The sensitivity analysis shows the recommendation holds across a wide band,
but the weights encode a preference about capital versus coverage that belongs to a finance team.

**Cannibalisation is modelled only through capacity.** Opening near an existing store shifts demand between
them but does not reduce total demand — no substitution effect, and no competitive response, which in a real
market would be the first-order consequence of a visible expansion.

**Spoilage is excluded.** Inventory loss runs ~1.8% of NOV. It is a real operating cost and would tighten the
economics of marginal sites, but it does not vary enough between sites to change their ranking.

**The equity trade-off is unaddressed.** The optimum leaves three localities with zero service rather than
spreading partial service everywhere. That is optimal against this objective and may be wrong against the
business's actual preferences; a minimum-coverage-per-locality constraint would express the alternative, at a
cost in total coverage.
