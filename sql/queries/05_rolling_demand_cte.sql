-- Q5 — Smoothed demand per locality, via a CTE and a rolling window.
--
-- This query is the bridge between the SQL layer and the optimization model.
-- Its output is the demand vector the optimizer consumes, which makes it the
-- most consequential query in the project: get it wrong and every downstream
-- result is wrong in a way no amount of careful modelling can rescue.
--
-- The motivation is statistical. Daily order counts are Poisson-distributed,
-- which means their variance equals their mean -- a locality averaging 400
-- orders a day will routinely swing between roughly 360 and 440 for no reason
-- other than chance. Siting a store on a single day's reading would be fitting
-- to noise. A seven-day trailing average both damps that variance and, because
-- the window is exactly one week long, cancels the weekday-weekend cycle
-- rather than letting it leak into the result. Choosing a window length equal
-- to the period of the seasonality you want to remove is the general principle,
-- and it is worth saying out loud when asked why seven and not five or ten.
--
-- Structurally there are two things to point at.
--
-- The CTE, introduced by WITH, gives a name to an intermediate result so the
-- final SELECT can read as a single clear statement instead of nesting three
-- subqueries inside each other. CTEs do not usually make a query faster -- most
-- engines inline them -- but they make it legible and debuggable, since you can
-- select from any stage in isolation while developing. That is reason enough.
--
-- The frame clause, ROWS BETWEEN 6 PRECEDING AND CURRENT ROW, is the part
-- people most often get wrong. A window function without a frame clause but
-- with an ORDER BY defaults to RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
-- ROW, which is a running total from the beginning of the partition rather than
-- a rolling window. Spelling the frame out is what makes this a seven-day
-- average and not a cumulative one.
--
-- There is a subtlety in ROWS versus RANGE that is worth carrying into an
-- interview: ROWS counts physical rows, RANGE counts logical values of the
-- ORDER BY expression. They coincide here only because the data has no gaps --
-- every locality has a row for every one of the 90 days. Had a locality gone
-- missing for a day, ROWS would silently reach back an extra calendar day to
-- find its sixth preceding row, and the "seven-day" average would quietly span
-- eight days. Dense data is an assumption this query makes, and knowing that
-- you are making it is the difference between a correct answer and a lucky one.

WITH daily AS (
    SELECT
        d.locality_id,
        d.order_date,
        d.order_count,
        d.avg_order_value,
        AVG(d.order_count) OVER (
            PARTITION BY d.locality_id
            ORDER BY d.order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_orders,
        ROW_NUMBER() OVER (
            PARTITION BY d.locality_id
            ORDER BY d.order_date
        ) AS day_index
    FROM demand AS d
),

-- The first six days of each locality's history have fewer than seven rows
-- behind them, so their "seven-day" average is really a one-, two- or
-- three-day average and is not comparable to the rest. Dropping them is the
-- honest choice; keeping them would leave a series whose early values are
-- systematically noisier than its later ones -- a warm-up artefact that is
-- easy to mistake for a real trend.
settled AS (
    SELECT *
    FROM daily
    WHERE day_index >= 7
)

SELECT
    l.locality_id,
    l.locality_name,
    l.latitude,
    l.longitude,
    l.avg_household_income_tier,
    l.population_density,
    l.repeat_rate_90d,
    ROUND(AVG(s.rolling_7d_orders), 1) AS smoothed_orders_per_day,
    ROUND(AVG(s.avg_order_value), 2)   AS avg_order_value,
    COUNT(*)                           AS days_used
FROM settled AS s
JOIN localities AS l
    ON l.locality_id = s.locality_id
GROUP BY
    l.locality_id,
    l.locality_name,
    l.latitude,
    l.longitude,
    l.avg_household_income_tier,
    l.population_density,
    l.repeat_rate_90d
ORDER BY
    smoothed_orders_per_day DESC;
