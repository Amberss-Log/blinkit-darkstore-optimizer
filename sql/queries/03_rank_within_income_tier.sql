-- Q3 — Rank localities by demand density within each income tier.
--
-- This is the window-function query, and window functions are the single topic
-- most worth being fluent in for these interviews. Every recorded account of
-- the Eternal online assessment mentions them by name.
--
-- The idea behind a window function is easier to hold onto if you contrast it
-- with GROUP BY. A GROUP BY collapses rows: feed it 35 localities across three
-- income tiers and you get three rows back, and the individual localities are
-- gone. A window function does the same arithmetic over the same groupings but
-- keeps every row intact, attaching the group-level answer alongside. You end
-- up with 35 rows, each carrying its own rank within its own tier. Nothing is
-- lost, which is precisely why you reach for it when the question is "where
-- does this row sit relative to its peers" rather than "what is the total".
--
-- The OVER clause is what defines the window. PARTITION BY splits the rows into
-- independent groups -- the ranking restarts from 1 inside each income tier --
-- and ORDER BY decides the order within each partition. If you omit PARTITION
-- BY entirely, the whole result set becomes one window and you get a global
-- ranking instead.
--
-- Three ranking functions exist and interviewers like asking the difference,
-- so it is worth seeing them side by side on the same data. Given values
-- 100, 90, 90, 80:
--
--     RANK()        -> 1, 2, 2, 4   ties share a rank, then it skips
--     DENSE_RANK()  -> 1, 2, 2, 3   ties share a rank, no gap afterwards
--     ROW_NUMBER()  -> 1, 2, 3, 4   never ties, breaks them arbitrarily
--
-- Note also what is being ranked. Raw order volume would simply rediscover that
-- big localities are big. Dividing demand by population density gives orders
-- per unit of density -- a measure of how hard a neighbourhood punches for its
-- size. That is the more interesting question, and choosing the right metric to
-- rank on is itself the analytical judgment being assessed.

WITH locality_stats AS (
    SELECT
        l.locality_id,
        l.locality_name,
        l.avg_household_income_tier,
        l.population_density,
        AVG(d.order_count) AS avg_orders_per_day
    FROM localities AS l
    JOIN demand AS d
        ON d.locality_id = l.locality_id
    GROUP BY
        l.locality_id,
        l.locality_name,
        l.avg_household_income_tier,
        l.population_density
)
SELECT
    locality_name,
    avg_household_income_tier,
    ROUND(avg_orders_per_day, 1) AS avg_orders_per_day,
    ROUND(population_density, 0) AS population_density,
    ROUND(avg_orders_per_day / population_density * 1000, 2) AS orders_per_1k_density,

    RANK() OVER (
        PARTITION BY avg_household_income_tier
        ORDER BY avg_orders_per_day / population_density DESC
    ) AS rank_in_tier,

    -- A second window over the same partition, this time an aggregate rather
    -- than a ranking. Because it keeps every row, each locality can be compared
    -- against its own tier's average on the same line -- something GROUP BY
    -- would need a self-join to achieve.
    ROUND(
        AVG(avg_orders_per_day) OVER (PARTITION BY avg_household_income_tier),
        1
    ) AS tier_avg_orders_per_day
FROM locality_stats
ORDER BY
    avg_household_income_tier,
    rank_in_tier;
