-- Q1 — Total and average daily demand by locality, ranked.
--
-- The foundation query: it collapses 3,150 daily rows down to one row per
-- locality, which is the shape everything downstream expects.
--
-- Worth being precise about what the aggregates mean here. SUM(order_count)
-- is the total orders across the whole 90-day window, while AVG(order_count)
-- is the average of the daily values -- and because the grain of `demand` is
-- one row per locality per day, that average is genuinely "orders per day",
-- not an average of some other unit. Grain is what makes an aggregate
-- interpretable; get the grain wrong and AVG() silently answers a question
-- nobody asked.
--
-- The revenue column multiplies each day's order count by that same day's
-- average order value before summing. Doing it in that order matters: summing
-- the counts and multiplying by an overall average AOV would weight every day
-- equally regardless of volume, which quietly understates busy days. This is
-- the difference between a weighted and an unweighted average, and it is a
-- favourite interview trap.

SELECT
    l.locality_id,
    l.locality_name,
    l.avg_household_income_tier,
    l.population_density,
    COUNT(*)                                        AS days_observed,
    SUM(d.order_count)                              AS total_orders,
    ROUND(AVG(d.order_count), 1)                    AS avg_orders_per_day,
    ROUND(AVG(d.avg_order_value), 2)                AS avg_order_value,
    ROUND(SUM(d.order_count * d.avg_order_value), 2) AS total_revenue
FROM demand AS d
JOIN localities AS l
    ON l.locality_id = d.locality_id
GROUP BY
    l.locality_id,
    l.locality_name,
    l.avg_household_income_tier,
    l.population_density
ORDER BY
    avg_orders_per_day DESC;
