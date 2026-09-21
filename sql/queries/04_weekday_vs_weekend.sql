-- Q4 — Weekday versus weekend demand, per locality.
--
-- An operational question with real consequences: staffing, inventory and
-- rider supply all key off how much heavier the weekend runs. A locality that
-- spikes 30% on Saturdays needs a different roster from one that is flat.
--
-- The technique on display is conditional aggregation, and it is one of the
-- most useful patterns in analytical SQL. The trick is that an aggregate
-- function ignores nulls. Put a CASE WHEN inside AVG() so that it yields a
-- value on the days you care about and NULL everywhere else, and the aggregate
-- quietly computes over just that subset. Two of them side by side give you
-- weekday and weekend averages from a single pass over the table, where the
-- naive approach would run two queries and join them, or self-join the table
-- to itself.
--
-- Once you see the pattern you start seeing it everywhere. Pivoting rows into
-- columns, counting how many rows satisfy some condition with
-- SUM(CASE WHEN ... THEN 1 ELSE 0 END), computing conversion rates by status --
-- all the same move.
--
-- One portability note. Day-of-week extraction is the single place MySQL and
-- SQLite genuinely disagree: MySQL has DAYOFWEEK(), returning 1 for Sunday
-- through 7 for Saturday, while SQLite offers strftime('%w'), which returns
-- 0 for Sunday through 6 for Saturday, as a string. The {is_weekend}
-- placeholder below is substituted at runtime by src/db.py, which normalises
-- both to the same convention. Everything else in this file is identical
-- across the two engines.

SELECT
    l.locality_name,
    l.avg_household_income_tier,

    ROUND(AVG(CASE WHEN {is_weekend} THEN NULL ELSE d.order_count END), 1)
        AS avg_weekday_orders,

    ROUND(AVG(CASE WHEN {is_weekend} THEN d.order_count ELSE NULL END), 1)
        AS avg_weekend_orders,

    -- The uplift, expressed as a percentage. This is the number an operations
    -- planner would actually act on.
    ROUND(
        100.0 * (
            AVG(CASE WHEN {is_weekend} THEN d.order_count ELSE NULL END)
            / AVG(CASE WHEN {is_weekend} THEN NULL ELSE d.order_count END)
            - 1
        ),
        1
    ) AS weekend_uplift_pct,

    -- Counting rows that satisfy a condition, same CASE WHEN idea applied to
    -- SUM instead of AVG. Serves as a sanity check that the day-of-week logic
    -- actually split the data the way it was supposed to.
    SUM(CASE WHEN {is_weekend} THEN 1 ELSE 0 END) AS weekend_days,
    SUM(CASE WHEN {is_weekend} THEN 0 ELSE 1 END) AS weekday_days
FROM demand AS d
JOIN localities AS l
    ON l.locality_id = d.locality_id
GROUP BY
    l.locality_id,
    l.locality_name,
    l.avg_household_income_tier
ORDER BY
    weekend_uplift_pct DESC;
