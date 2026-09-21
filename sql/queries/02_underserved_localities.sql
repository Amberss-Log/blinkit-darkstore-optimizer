-- Q2 — The ten highest-demand localities with no existing store within X km.
--
-- This is the query that actually identifies the business opportunity: places
-- where people order heavily but no dark store sits close enough to serve them
-- inside the delivery promise. Everything the optimizer does later is a more
-- rigorous version of this same question.
--
-- Three ideas are doing the work here.
--
-- The first is computing distance in SQL. There is no geography type in play,
-- so the haversine formula is written out longhand: convert degrees to radians,
-- take half the angular differences, and multiply the result by the Earth's
-- mean radius of 6371 km. It looks heavy, but it is only arithmetic, and both
-- MySQL 8 and SQLite provide every function it needs.
--
-- It is worth knowing why this is the ASIN form rather than the shorter
-- ACOS/spherical-law-of-cosines version. For two points a couple of kilometres
-- apart, the cosine of the angle between them is a hair under 1.0, and in
-- floating point it can round to just over 1.0 -- at which point ACOS is
-- outside its domain and the query errors or returns null. The haversine form
-- works with squared sines of half-differences, which stay comfortably inside
-- their domain at short distances. Short distances are the only ones we care
-- about, so this is the form that belongs here.
--
-- The third idea is NOT EXISTS, and it is the part most worth defending out
-- loud. The instinctive approach is a LEFT JOIN onto existing_stores followed
-- by a filter for rows where the join matched nothing. That produces the right
-- answer, but it materialises every locality-store pair before discarding
-- almost all of them. NOT EXISTS expresses a semi-join: the engine can stop
-- scanning the instant it finds one qualifying store, because a single nearby
-- store is enough to disqualify the locality. On 35 localities and 6 stores the
-- difference is invisible. On a national dataset it is the difference between
-- a query that returns and one that does not.
--
-- Takes one bind parameter, radius_km.
--
-- Note the colon is deliberately left off that name in this comment. Bind
-- parameters are substituted by scanning the text of the statement, comments
-- included, whereas the database strips comments before parsing. Write the
-- placeholder syntax in a comment and the driver emits two placeholders while
-- the engine only sees one, which fails with an unhelpful complaint about the
-- number of bindings supplied.

WITH locality_demand AS (
    SELECT
        d.locality_id,
        AVG(d.order_count) AS avg_orders_per_day
    FROM demand AS d
    GROUP BY d.locality_id
)
SELECT
    l.locality_id,
    l.locality_name,
    l.avg_household_income_tier,
    ROUND(ld.avg_orders_per_day, 1) AS avg_orders_per_day,
    ROUND(l.population_density, 0)  AS population_density
FROM localities AS l
JOIN locality_demand AS ld
    ON ld.locality_id = l.locality_id
WHERE NOT EXISTS (
    SELECT 1
    FROM existing_stores AS s
    WHERE 2 * 6371.0 * ASIN(
              SQRT(
                  POWER(SIN((RADIANS(s.latitude) - RADIANS(l.latitude)) / 2), 2)
                  + COS(RADIANS(l.latitude)) * COS(RADIANS(s.latitude))
                  * POWER(SIN((RADIANS(s.longitude) - RADIANS(l.longitude)) / 2), 2)
              )
          ) <= :radius_km
)
ORDER BY ld.avg_orders_per_day DESC
LIMIT 10;
