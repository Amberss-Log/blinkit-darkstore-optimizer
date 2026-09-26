# Blinkit dark-store network optimizer

Blinkit had 2,443 dark stores in June 2026 and still opens more every quarter, at about ₹2.5 crore of capex per
store. At that price, deciding where the next store goes is really a capital allocation problem. In this
project I model that decision for one city, Indore: given a fixed budget, which candidate sites should be
opened so that as many orders as possible get delivered? Three limits apply at once. The budget is fixed, a
store can only serve customers within the radius that keeps delivery at ten minutes, and each store can only
pick and pack a certain number of orders a day. I formulate this as a capacitated facility location problem,
solve it as a mixed-integer linear program (MILP) with PuLP and the CBC solver, and compare the result with a
greedy heuristic, which is roughly how a team would make the decision without a model. Without that
comparison there would be no way to tell whether the optimizer's answer is actually any good.

## What's real

The following figures come from Eternal's public disclosures. I use them to calibrate the synthetic data and
nowhere claim more than they state.

| Figure | Value | Source |
|---|---|---|
| Blinkit store count | 2,443 | Eternal Q1 FY27 (June 2026) |
| Capex per store, including warehousing | ~₹2.5 crore | Eternal, long-run assumption |
| Steady-state NOV per store per day | ~₹11 lakh | Eternal, steady-state target |
| Net working capital | ~12 days | Eternal |
| Target pre-tax ROCE | ~41.7% | Eternal, under the above assumptions |
| Network footprint | ~19 million sq ft across 300+ cities | Eternal |
| Inventory and spoilage loss | ~1.8% of NOV, "fairly asymptotic" | Eternal |
| Average delivery distance of a peer | ~1.7–1.8 km, FY26 | Zepto disclosure |

## What's synthetic

Everything else was generated for this project. No real Blinkit order, rent, location or dispatch data was
used, and none is publicly available at this level of detail.

| Assumption | Value | Reason |
|---|---|---|
| City | Indore | Tier-2 city with a thin quick-commerce presence, so expanding there is an open question |
| Localities | 35, real names, approximate coordinates | Enough spatial structure to matter, small enough to solve in seconds |
| Demand history | 90 days, 1 Sep to 29 Nov 2025 | Includes Diwali (20 Oct 2025), so the festive peak sits on a real date |
| Candidate sites | 50 | |
| Existing stores | 6 | Makes the decision incremental, as it is for Blinkit |
| Average order value | ~₹560 overall, varying by income tier | Assumed; the capacity estimate depends on it |
| Store capacity | ~1,200 to 2,400 orders per day | Derived, see below |
| Delivery radius | 2.5 km | Assumed, set a little above the 1.7–1.8 km peer average |
| Budget | ₹20 crore | Assumed, enough for about eight stores at the disclosed capex |

Demand is not drawn independently for each locality. Population density peaks around four real centres
(Rajwada, Palasia, Vijay Nagar and Bhawarkuan) and falls off with distance from them, so neighbouring areas
look alike, as they do in a real city. Affluence is generated separately, since the densest parts of a city are
rarely its richest. Daily order counts are Poisson distributed, with a weekly cycle and a festive spike around
Diwali. At the end of the run, the generator checks that its totals land close to the disclosed figures and
stops with an error if they don't.

### Where the capacity figure comes from

Store capacity is not disclosed, so I derived it:

```
₹11,00,000 NOV per store per day  ÷  ~₹600 average order value  ≈  1,800 orders per store per day
```

The ₹11 lakh comes from Eternal, but the ₹600 is my own assumption, so 1,800 is only as good as that
assumption. At ₹500 per order the same calculation gives about 2,200, and at ₹700 about 1,570. For that
reason the site capacities in the data are spread between 1,200 and 2,400 instead of all being set to 1,800.

## How to run

You need Python 3.11 or later. MySQL is optional.

```bash
python -m venv .venv
.venv\Scripts\activate           # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py          # builds and fills the database
python -m pytest                 # 25 tests
jupyter lab notebook.ipynb
```

With `DATABASE_URL` unset, everything runs on SQLite and `generate_data.py` creates `data/blinkit_sim.db`
itself. To use MySQL 8, copy `.env.example` to `.env` and set:

```
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/blinkit_sim
```

The database is created if it doesn't exist. It should be a dedicated one: the schema rebuild drops the
project's own tables, so the script checks the target first and refuses to run if it finds any table the
project didn't create. The five analysis queries run unchanged on both MySQL and SQLite. The one place the two
dialects disagree, extracting the day of the week, is handled in `src/db.py`. Keep `.env` out of git; it is
already in `.gitignore`.

```
generate_data.py      builds and fills the database
notebook.ipynb        the analysis
sql/                  MySQL and SQLite schemas, plus the five analysis queries
src/db.py             connection, dialect handling, schema rebuild and its safety check
src/geo.py            haversine distances, the delivery-radius mask
src/evaluate.py       shared metrics and the dispatch rule
src/baseline.py       greedy heuristic
src/optimizer.py      the MILP, sensitivity sweep, festive scenarios
src/forecast.py       ridge regression demand forecast
src/plots.py          figures
tests/                25 tests, including a toy problem small enough to solve by hand
```

## Results

The greedy baseline follows the same rules as the optimizer: the same budget, capacities, radius and existing
stores. With ₹20 crore, greedy opened 6 stores for ₹18.96 crore and served 69.5% of demand. The MILP opened 9
stores for ₹19.97 crore and served 75.8%, which is about 2,200 more orders a day from the same budget.

The optimizer did not win by choosing better locations. In both plans every store runs at full capacity,
because there is more demand in the city than the budget can buy capacity for. Coverage therefore depends only
on how much capacity the money buys. Greedy always picked the site that added the most orders on its own,
which usually meant the largest and most expensive one (₹2.8 to ₹3.5 crore). After six of those it was left
with ₹1.04 crore, too little for another site. The MILP chose nine cheaper sites (₹1.8 to ₹2.5 crore) that
together gave more capacity: 796 orders a day per crore spent, against 722 for greedy. Greedy can't find a plan
like that, because it looks at one site at a time and never compares combinations.

I tested the result in four ways.

**Sensitivity to the capex weight.** The model has a weight, λ₁, for how many orders a day one crore of
capital must bring in to be worth spending. Anywhere from 0 to 600, the answer stays the same nine stores. It
first changes at 720, which matches the break-even I worked out beforehand for a ₹2.5 crore store serving about
1,800 orders a day. Above about 900, spending never looks worthwhile, and the model opens nothing.

**Rounding the LP relaxation.** The LP relaxation is the same problem with stores allowed to open partially.
Its objective came within 0.5% of the MILP's, which suggests solving that easier version and rounding it
might have been enough. It isn't. The relaxation opens thirteen sites, seven of them only partly. Rounding
down gives 61.1% coverage, which is worse than greedy. Rounding to the nearest whole store gives 72.4%.
Rounding up gives 96.2%, but costs ₹30.95 crore, well over budget. A good plan that uses whole stores and
stays within budget only comes out of the MILP.

**Festive demand.** At 1.5, 2 and 2.5 times normal demand, the plan and the number of orders served stay the
same. Once every store is full, moving the same money around can't add capacity, and coverage falls to 30.3%
at 2.5 times demand. Permanent stores built for the peak would stand mostly idle the rest of the year, so
Diwali is better handled with temporary capacity, such as extra staff, overflow space and stock placed ahead
of time.

**Forecast demand.** A ridge regression forecast beats a seasonal-naive baseline by 27% on mean absolute
error. Running the optimizer on the forecast instead of historical averages picks exactly the same nine sites.
The recommendation holds, and a better forecast would not have changed it.

## Limitations

**The data is synthetic.** Totals are calibrated to disclosed figures, but everything at locality level is
invented. The results show how the method behaves on plausible data. They say nothing about Indore itself.

**The model counts orders, not their value.** A ₹420 order and a ₹720 order count the same, so the optimizer
will give up high-value customers if it can get more orders elsewhere. That is what happened here. High-income
localities went from 100% service under greedy to 84.7%, Vijay Nagar lost service completely, and mid-income
areas went from 40.4% to 77.5%. Daily revenue still rose, from ₹157.6 to ₹166.7 lakh, but only because the
extra orders outweighed the smaller baskets. The next version will weight each locality by order value and
customer loyalty.

**Operating costs are left out.** The model accounts for setup capex only. Staff, power and rent are ignored,
even though the site data includes rent. Nine stores need more store staff than six, so the optimizer's
advantage is somewhat smaller than it looks. Rider costs grow with orders rather than stores, and nearer stores
shorten trips: average delivery distance fell from 0.71 to 0.63 km.

**Distances are measured from locality centroids.** Each locality is treated as a single point, so the model
reports an average delivery distance of about 0.63 km. The disclosed peer average is 1.7 to 1.8 km, because
real deliveries follow roads and start from doorsteps, not from the middle of a neighbourhood. Using road
distances would make every trip longer and put fewer localities within range of each site.

**Capacity is a daily total.** In reality orders bunch up in the evening, and a store that can handle 1,800
orders spread over a day may still fall behind at 7 pm.

**Competition and cannibalisation are not modelled.** Opening a store near an existing one only moves demand
between them. Customers don't switch to a competitor, and competitors don't react to a visible expansion.

**Spoilage is left out.** Inventory loss is about 1.8% of NOV and is a real cost. It varies little from one
site to another, though, so it would not change which sites come out on top.
