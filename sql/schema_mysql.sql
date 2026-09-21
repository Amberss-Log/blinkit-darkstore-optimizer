-- Blinkit dark-store network optimizer — MySQL 8 schema
--
-- Run automatically by generate_data.py, or by hand:
--     mysql -u <user> -p blinkit_sim < sql/schema_mysql.sql
--
-- Tables are dropped children-first because a foreign key stops you from
-- deleting a parent row that something still points at. Same reason applies to
-- dropping whole tables: demand references localities, so demand goes first.

DROP TABLE IF EXISTS demand;
DROP TABLE IF EXISTS candidate_sites;
DROP TABLE IF EXISTS existing_stores;
DROP TABLE IF EXISTS localities;


-- ---------------------------------------------------------------------------
-- localities — dimension table.
-- Grain: one row per neighbourhood in the city.
-- ---------------------------------------------------------------------------
CREATE TABLE localities (
    locality_id               INT            NOT NULL AUTO_INCREMENT,
    locality_name             VARCHAR(80)    NOT NULL,

    -- DECIMAL(9,6) = 9 significant digits, 6 after the point, e.g. 22.719568.
    -- Six decimal places is roughly 11 cm of precision — far more than enough,
    -- and exact, unlike a float.
    latitude                  DECIMAL(9,6)   NOT NULL,
    longitude                 DECIMAL(9,6)   NOT NULL,

    population_density        DECIMAL(10,2)  NOT NULL,  -- people per sq km
    avg_household_income_tier VARCHAR(4)     NOT NULL,  -- 'low' | 'mid' | 'high'

    PRIMARY KEY (locality_id),

    -- Two localities with the same name would be a data-generation bug, and the
    -- database is a cheaper place to catch it than a notebook cell.
    UNIQUE KEY uq_localities_name (locality_name),

    CONSTRAINT chk_localities_tier
        CHECK (avg_household_income_tier IN ('low', 'mid', 'high')),
    CONSTRAINT chk_localities_density
        CHECK (population_density > 0),
    CONSTRAINT chk_localities_lat  CHECK (latitude  BETWEEN -90  AND 90),
    CONSTRAINT chk_localities_lon  CHECK (longitude BETWEEN -180 AND 180)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- demand — fact table. This is the only table that grows with time.
-- Grain: one row per locality per day.
--
-- The column is order_date, not `date`: DATE is a type name in SQL, and a
-- column called `date` has to be quoted everywhere it appears. Naming it
-- order_date keeps every query readable.
-- ---------------------------------------------------------------------------
CREATE TABLE demand (
    locality_id     INT           NOT NULL,
    order_date      DATE          NOT NULL,
    order_count     INT           NOT NULL,

    -- Money is DECIMAL, never FLOAT. Binary floating point cannot represent
    -- 0.10 exactly, so floats accumulate error under SUM() — the classic
    -- "total is off by two paise" bug. DECIMAL(8,2) is exact base-10.
    avg_order_value DECIMAL(8,2)  NOT NULL,

    -- Composite primary key: the grain, enforced. It is now physically
    -- impossible to insert the same locality twice for the same day.
    -- In InnoDB the primary key IS the clustered index, so this also gives us
    -- the (locality_id, order_date) index for free — no separate index needed.
    PRIMARY KEY (locality_id, order_date),

    -- Separate index on order_date alone. The PK index is ordered by
    -- locality_id first, so it cannot answer "all localities for one date"
    -- efficiently. Date-range scans are common enough here to justify it.
    KEY idx_demand_order_date (order_date),

    CONSTRAINT fk_demand_locality
        FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_demand_count CHECK (order_count >= 0),
    CONSTRAINT chk_demand_aov   CHECK (avg_order_value > 0)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- candidate_sites — dimension table.
-- Grain: one row per location Blinkit could open a store at.
-- ---------------------------------------------------------------------------
CREATE TABLE candidate_sites (
    site_id                     INT            NOT NULL AUTO_INCREMENT,
    locality_id                 INT            NOT NULL,
    latitude                    DECIMAL(9,6)   NOT NULL,
    longitude                   DECIMAL(9,6)   NOT NULL,

    estimated_rent_monthly      DECIMAL(12,2)  NOT NULL,  -- INR/month
    max_capacity_orders_per_day INT            NOT NULL,  -- throughput ceiling

    -- INR. DECIMAL(14,2) holds up to 999,999,999,999.99 — comfortably past the
    -- ~2.5 crore (25,000,000) per store we are calibrating to.
    setup_capex                 DECIMAL(14,2)  NOT NULL,

    PRIMARY KEY (site_id),

    -- Foreign keys are not indexed automatically in every engine, and we join
    -- sites to localities constantly.
    KEY idx_sites_locality (locality_id),

    CONSTRAINT fk_sites_locality
        FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_sites_capacity CHECK (max_capacity_orders_per_day > 0),
    CONSTRAINT chk_sites_capex    CHECK (setup_capex > 0),
    CONSTRAINT chk_sites_rent     CHECK (estimated_rent_monthly > 0)
) ENGINE=InnoDB;


-- ---------------------------------------------------------------------------
-- existing_stores — dimension table.
-- Grain: one row per store already trading in the city.
--
-- No setup_capex column: that money is already spent. It is a sunk cost and
-- must not enter the budget constraint. The optimizer chooses *additional*
-- stores, which is the decision Blinkit actually faces with 2,443 already open.
-- ---------------------------------------------------------------------------
CREATE TABLE existing_stores (
    store_id                    INT            NOT NULL AUTO_INCREMENT,
    locality_id                 INT            NOT NULL,
    latitude                    DECIMAL(9,6)   NOT NULL,
    longitude                   DECIMAL(9,6)   NOT NULL,
    max_capacity_orders_per_day INT            NOT NULL,
    opened_on                   DATE           NOT NULL,

    PRIMARY KEY (store_id),
    KEY idx_stores_locality (locality_id),

    CONSTRAINT fk_stores_locality
        FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_stores_capacity CHECK (max_capacity_orders_per_day > 0)
) ENGINE=InnoDB;
