-- Blinkit dark-store network optimizer — SQLite schema
--
-- Mirror of sql/schema_mysql.sql. Used when DATABASE_URL is not set, so the
-- project runs with no database server installed.
--
-- Differences from the MySQL file, and why:
--   * INTEGER PRIMARY KEY AUTOINCREMENT instead of INT AUTO_INCREMENT.
--   * No ENGINE=InnoDB — SQLite has one storage engine.
--   * Indexes are separate CREATE INDEX statements; SQLite has no inline KEY.
--   * Types are advisory. SQLite uses dynamic typing with "type affinity":
--     DECIMAL(8,2) gets NUMERIC affinity and is stored as an 8-byte float,
--     so it is NOT exact the way MySQL's DECIMAL is. Harmless here (synthetic
--     money, two decimal places, no accounting), but worth knowing before
--     trusting SQLite with real currency.
--   * Foreign keys are OFF by default in SQLite and must be switched on per
--     connection with `PRAGMA foreign_keys = ON`. src/db.py does this on every
--     connect — without it the FK clauses below are silently decorative.

DROP TABLE IF EXISTS demand;
DROP TABLE IF EXISTS candidate_sites;
DROP TABLE IF EXISTS existing_stores;
DROP TABLE IF EXISTS localities;


-- localities — dimension. Grain: one row per neighbourhood.
CREATE TABLE localities (
    locality_id               INTEGER       PRIMARY KEY AUTOINCREMENT,
    locality_name             VARCHAR(80)   NOT NULL UNIQUE,
    latitude                  DECIMAL(9,6)  NOT NULL,
    longitude                 DECIMAL(9,6)  NOT NULL,
    population_density        DECIMAL(10,2) NOT NULL,
    avg_household_income_tier VARCHAR(4)    NOT NULL,

    CHECK (avg_household_income_tier IN ('low', 'mid', 'high')),
    CHECK (population_density > 0),
    CHECK (latitude  BETWEEN -90  AND 90),
    CHECK (longitude BETWEEN -180 AND 180)
);


-- demand — fact table. Grain: one row per locality per day.
CREATE TABLE demand (
    locality_id     INTEGER      NOT NULL,
    order_date      DATE         NOT NULL,
    order_count     INTEGER      NOT NULL,
    avg_order_value DECIMAL(8,2) NOT NULL,

    -- Composite PK enforces the grain, exactly as in MySQL.
    PRIMARY KEY (locality_id, order_date),

    FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CHECK (order_count >= 0),
    CHECK (avg_order_value > 0)
);

CREATE INDEX idx_demand_order_date ON demand (order_date);


-- candidate_sites — dimension. Grain: one row per possible store location.
CREATE TABLE candidate_sites (
    site_id                     INTEGER       PRIMARY KEY AUTOINCREMENT,
    locality_id                 INTEGER       NOT NULL,
    latitude                    DECIMAL(9,6)  NOT NULL,
    longitude                   DECIMAL(9,6)  NOT NULL,
    estimated_rent_monthly      DECIMAL(12,2) NOT NULL,
    max_capacity_orders_per_day INTEGER       NOT NULL,
    setup_capex                 DECIMAL(14,2) NOT NULL,

    FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CHECK (max_capacity_orders_per_day > 0),
    CHECK (setup_capex > 0),
    CHECK (estimated_rent_monthly > 0)
);

CREATE INDEX idx_sites_locality ON candidate_sites (locality_id);


-- existing_stores — dimension. Grain: one row per store already trading.
-- No setup_capex: already spent, sunk, must not enter the budget constraint.
CREATE TABLE existing_stores (
    store_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    locality_id                 INTEGER NOT NULL,
    latitude                    DECIMAL(9,6) NOT NULL,
    longitude                   DECIMAL(9,6) NOT NULL,
    max_capacity_orders_per_day INTEGER NOT NULL,
    opened_on                   DATE    NOT NULL,

    FOREIGN KEY (locality_id) REFERENCES localities (locality_id)
        ON DELETE CASCADE,

    CHECK (max_capacity_orders_per_day > 0)
);

CREATE INDEX idx_stores_locality ON existing_stores (locality_id);
