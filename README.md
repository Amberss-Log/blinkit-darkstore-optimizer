# Blinkit dark-store network optimizer

Blinkit had 2,443 dark stores in June 2026 and still opens more every quarter, at about ₹2.5 crore of capex per
store. At that price, deciding where the next store goes is a capital allocation problem. This project models
that decision for one city, Indore: given a fixed budget, which candidate sites should Blinkit open? Three limits
apply at once. The budget is fixed, a store can only serve customers within the radius that keeps delivery at
ten minutes, and each store can pick and pack only so many orders a day. I set the problem up as capacitated
facility location, solve it as a mixed-integer linear program (MILP) with PuLP and the CBC solver, and compare
the result with a greedy heuristic, which is roughly how a team would make the decision without a model. The
comparison turned out to depend on what the model is asked to maximise. Counting orders and counting profit
lead to different recommendations.

## What's real

These figures come from Eternal's public disclosures. I use them to calibrate the synthetic data and never claim
more than they say.

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

Everything else was generated for this project. It contains no real Blinkit order, rent, location, retention or
dispatch data, and none of that is public at this level of detail.

| Assumption | Value | Reason |
|---|---|---|
| City | Indore | A Tier-2 city with little quick-commerce presence, so whether to expand there is a real question |
| Localities | 35, real names, approximate coordinates | Enough spatial structure to matter, small enough to solve in seconds |
| Demand history | 90 days, 1 Sep to 29 Nov 2025 | Includes Diwali (20 Oct 2025), so the festive peak falls on its real date |
| Candidate sites | 50, rent ₹0.4 to ₹2.8 lakh a month | |
| Existing stores | 6 | Makes the decision incremental, as it is for Blinkit |
| Average order value | about ₹420 to ₹730 by locality | Assumed; drives the capacity estimate |
| Loyalty (90-day repeat rate) | 0.44 to 0.79, higher in richer areas | Assumed |
| Staff cost per store | ₹4.5 lakh a month | Assumed |
| Store capacity | 1,200 to 2,400 orders a day | Derived, see below |
| Contribution margin | 4.45% of NOV | Derived, see below |
| Delivery radius | 2.5 km | Assumed, set a little above the 1.7–1.8 km peer average |
| Budget | ₹20 crore | Assumed, enough for about eight stores at the disclosed capex |

Demand has spatial structure. Each locality's figure depends on where it is, so the numbers aren't independent
draws. Population density peaks around four real centres (Rajwada, Palasia, Vijay Nagar and Bhawarkuan) and
falls off with distance from them, so neighbouring areas look alike, as they do in a real city. Affluence is
generated separately, since a city's densest parts are rarely its richest. Daily order counts follow a Poisson
distribution, with a weekly cycle and a festive spike around Diwali. The generator checks that its totals land
close to the disclosed figures and stops with an error if they don't.

### Two derived numbers

Store capacity isn't disclosed, so I derived it:

```
₹11,00,000 NOV per store per day  ÷  ~₹600 average order value  ≈  1,800 orders per store per day
```

The ₹11 lakh is Eternal's figure, but the ₹600 is my assumption, so 1,800 is only as good as that assumption. At
₹500 a basket the same division gives about 2,200, and at ₹700 about 1,570. That is why site capacities are
spread between 1,200 and 2,400 rather than fixed at one precise-looking value.

The contribution margin isn't disclosed either. I worked it out backwards from the 41.7% return target. For a
₹2.5 crore store packing 1,800 orders a day to earn 41.7% a year after paying its rent and staff, it has to keep
about 4.45% of the value of its orders. Like capacity, this is only as good as the assumptions behind it, here
the staff cost. So I check every profit result at margins from 3% to 8%.

## How to run

You need Python 3.11 or later. MySQL is optional.

```bash
python -m venv .venv
.venv\Scripts\activate           # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python generate_data.py          # builds and fills the database
python -m pytest                 # 46 tests
jupyter lab notebook.ipynb
```

If `DATABASE_URL` is unset, everything runs on SQLite, and `generate_data.py` creates `data/blinkit_sim.db`
itself. To use MySQL 8, copy `.env.example` to `.env` and set:

```
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/blinkit_sim
```

The database is created if it doesn't exist. Use a dedicated database: rebuilding the schema drops the project's
tables, so the script checks the target first and refuses to run if it finds any table the project didn't
create. The five analysis queries run unchanged on both MySQL and SQLite. The two dialects disagree in only one
place, extracting the day of the week, and `src/db.py` handles that. Keep `.env` out of git. It's already listed
in `.gitignore`.

```
generate_data.py      builds and fills the database
notebook.ipynb        the analysis
sql/                  MySQL and SQLite schemas, plus the five analysis queries
src/db.py             connection, dialect handling, schema rebuild and its safety check
src/geo.py            haversine distances, the delivery-radius mask
src/evaluate.py       shared metrics and the dispatch rule
src/baseline.py       greedy heuristic
src/optimizer.py      the MILP, sensitivity sweep, festive scenarios, LP rounding test
src/economics.py      order value, loyalty, running costs, the derived margin, profit
src/compare.py        runs every objective on the same inputs and tabulates them
src/forecast.py       ridge regression demand forecast
src/plots.py          figures
tests/                46 tests, including a toy problem small enough to solve by hand
```

## Results

### Counting orders

The first version of the model maximised orders served. Greedy followed the same rules as the optimizer: the
same budget, capacities, radius and existing stores. With ₹20 crore, greedy opened 6 stores for ₹18.96 crore and
served 69.5% of demand. The MILP opened 9 stores for ₹19.97 crore and served 75.8%, about 2,200 more orders a day
from the same budget.

The MILP won by buying more capacity per rupee, not by finding better locations. Both plans fill every store,
because the city wants more than the budget can pay for. Greedy kept picking the site that added the most orders
on its own, which usually meant the largest and most expensive (₹2.8 to ₹3.5 crore). After six of those it had
₹1.04 crore left over, too little for another site. The MILP instead chose nine cheaper sites (₹1.8 to ₹2.5 crore)
that together gave more capacity: 796 orders a day per crore spent, against 722 for greedy.

That result held up under four checks. The answer stays at the same nine stores for any capex weight λ₁ from 0 to
600 orders per crore, and first changes at 720, the break-even I worked out beforehand. Rounding the LP
relaxation doesn't produce a usable plan. Rounding down covers 61.1% of demand, rounding to the nearest whole
store covers 72.4%, and rounding up covers 96.2% but spends ₹30.95 crore, well over the budget. At festive demand
of 2.5 times normal, coverage falls to 30.3% and no reallocation of the same budget helps, which argues for
temporary capacity at the peak rather than permanent stores. A ridge forecast 27% more accurate than a
seasonal-naive baseline selects exactly the same sites.

### Counting money

Counting orders hid two problems. A ₹420 order counted the same as a ₹720 one, so the optimizer gave up
high-value customers: high-income localities went from 100% served under greedy to 84.7%, and Vijay Nagar lost
service entirely. It also treated stores as free to run once they were open. The second version weights each
order by its basket size and charges each new store its rent and staff. It measures profit against the existing
network, so knock-on effects on existing stores are counted too.

| Plan (derived 4.45% margin) | Stores | Coverage | High-income served | Vijay Nagar | Profit a year |
|---|---|---|---|---|---|
| Greedy, counting orders | 6 | 69.5% | 100% | 100% | ₹8.65 cr |
| MILP, counting orders | 9 | 75.8% | 84.7% | 0% | ₹8.16 cr |
| Greedy, maximising profit | 6 | 68.6% | 100% | 100% | ₹8.88 cr |
| **MILP, maximising profit** | **7** | **73.3%** | **100%** | **100%** | **₹10.02 cr** |
| MILP, profit weighted by loyalty | 7 | 72.2% | 100% | 100% | ₹9.71 cr |

Measured in money, the order-counting MILP earns less than greedy. Its extra stores cost more to run than their
orders bring in. The earlier result holds for orders, but not for profit.

The MILP set up to maximise profit is the recommendation. It opens 7 stores, serves 73.3% of demand, keeps every
high-income customer including Vijay Nagar, and earns ₹10.02 crore a year, a 50.5% return on the capital it
spends. That is ₹1.1 crore more than greedy with the same goal. Because its objective is profit, no plan within
the budget can earn more, and a test checks this at margins of 3%, 4.45%, 6% and 8%. The order-counting MILP only
overtakes order-counting greedy once the margin reaches about 6%.

Weighting orders by customer loyalty as well keeps Vijay Nagar, as the profit model already does. Beyond that it
drops service to low-income areas from 10.5% to none and costs about ₹0.3 crore a year. It amounts to a bet that
loyal customers are worth more than today's profit shows, and that can only be tested against real retention
data.

### Spending more

Every plan above runs its stores at full capacity, because ₹20 crore buys less capacity than the city needs. To
check whether a bigger budget would help, I raised it to ₹30, ₹40 and ₹60 crore, planned each network for
normal days, then ran the same stores at 2.5 times demand. Past about ₹34 crore nothing more is worth building:
almost all normal demand is already served, so the ₹40 and ₹60 crore plans are identical and leave money
unspent. The stores still run about 99% full on normal days, because neither model builds stores to sit idle,
so festival coverage tops out at 40%. More capex earns more (₹13.75 crore a year at ₹34 crore against ₹10.02
crore at ₹20 crore) but at a lower return on capital, 40.2% against 50.6%. Spending more buys coverage on
normal days, not festival capacity, which is why peaks are better met with temporary capacity.
## Limitations

**The data is synthetic.** Totals are calibrated to disclosed figures, but everything at the locality level,
loyalty included, is invented. The results show how the method behaves on plausible data. They say nothing about
Indore itself.

**The profit figures rest on assumptions.** The margin comes from the disclosed return target, but that
calculation depends on an assumed staff cost and leaves working capital out of the return on capital. The
recommendation holds for margins from 3% to 8%. The rupee amounts are illustrative.

**Value weighting favours richer areas.** Under the profit model, low-income localities have about 10% of their
demand served. A business that wanted to be present everywhere would need a minimum level of service per
locality, at some cost in profit.

**Rider cost per kilometre is left out.** The profit model gives distance no weight, so a longer delivery costs
nothing extra. In reality rider cost rises with distance, which would favour stores nearer their customers.

**Distances are measured from locality centroids.** The model treats each locality as a single point, so it
reports an average delivery distance of about 0.63 km. The disclosed peer average is 1.7 to 1.8 km, because real
deliveries follow roads and start from doorsteps. With road distances every trip would be longer, and fewer
localities would fall within range of each site.

**Capacity is a daily total.** Orders bunch up in the evening, so a store that handles 1,800 orders spread over
a day can still fall behind at 7 pm.

**Competition and cannibalisation are not modelled.** Opening a store near an existing one only moves demand
between them. Customers never switch to a competitor, and competitors don't respond to a visible expansion.

**Spoilage is left out.** Inventory loss is about 1.8% of NOV. That is a real cost, but it varies little between
sites, so it wouldn't change which sites come out on top.
