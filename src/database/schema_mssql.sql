-- =============================================================================
-- HVAC Market Analysis — Star Schema for SQL Server
-- =============================================================================
-- This schema is the SQL Server equivalent of schema.sql (SQLite).
-- Main differences:
--   - BOOLEAN → BIT (0/1)
--   - AUTOINCREMENT → IDENTITY(1,1)
--   - INSERT OR IGNORE → MERGE ... WHEN NOT MATCHED
--   - TRUE/FALSE → 1/0
--   - VARCHAR without mandatory length
--
-- Usage: executed automatically by db_manager.py when DB_TYPE=mssql
-- =============================================================================


-- =============================================
-- DIMENSION: Time
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dim_time')
CREATE TABLE dim_time (
    date_id     INT PRIMARY KEY,        -- Format YYYYMM (e.g., 202401)
    year        INT NOT NULL,
    month       INT NOT NULL,
    quarter     INT NOT NULL,
    is_heating  BIT NOT NULL,           -- 1 = heating season (Oct-Mar)
    is_cooling  BIT NOT NULL,           -- 1 = cooling season (Jun-Sep)

    CHECK (month BETWEEN 1 AND 12),
    CHECK (quarter BETWEEN 1 AND 4),
    CHECK (year BETWEEN 2015 AND 2030)
);


-- =============================================
-- DIMENSION: Geography
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dim_geo')
CREATE TABLE dim_geo (
    geo_id      INT IDENTITY(1,1) PRIMARY KEY,
    dept_code   VARCHAR(3)  NOT NULL UNIQUE,
    dept_name   VARCHAR(50) NOT NULL,
    city_ref    VARCHAR(50) NOT NULL,
    latitude    DECIMAL(7,4),
    longitude   DECIMAL(7,4),
    region_code VARCHAR(3) DEFAULT '84'
);


-- =============================================
-- DIMENSION: Equipment type (reference table)
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dim_equipment_type')
CREATE TABLE dim_equipment_type (
    equip_type_id   INT IDENTITY(1,1) PRIMARY KEY,
    label           VARCHAR(100) NOT NULL,
    categorie       VARCHAR(50)  NOT NULL,
    energie_source  VARCHAR(50),
    is_renewable    BIT DEFAULT 0
);


-- =============================================
-- FACT TABLE: HVAC Installations
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'fact_hvac_installations')
CREATE TABLE fact_hvac_installations (
    fact_id                 INT IDENTITY(1,1) PRIMARY KEY,
    date_id                 INT NOT NULL,
    geo_id                  INT NOT NULL,

    -- Target variable (installation proxy via DPE)
    nb_dpe_total            INT,
    nb_installations_pac    INT,
    nb_installations_clim   INT,
    nb_dpe_classe_ab        INT,

    -- Weather features (department-specific)
    temp_mean               DECIMAL(5,2),
    temp_max                DECIMAL(5,2),
    temp_min                DECIMAL(5,2),
    heating_degree_days     DECIMAL(8,2),
    cooling_degree_days     DECIMAL(8,2),
    precipitation_cumul     DECIMAL(8,2),
    nb_jours_canicule       INT,
    nb_jours_gel            INT,

    -- Real estate features (department-specific)
    nb_permis_construire    INT,
    nb_logements_autorises  INT,

    -- Energy features (department-specific)
    conso_elec_mwh          DECIMAL(12,2),
    conso_gaz_mwh           DECIMAL(12,2),

    FOREIGN KEY (date_id) REFERENCES dim_time(date_id),
    FOREIGN KEY (geo_id) REFERENCES dim_geo(geo_id),
    UNIQUE(date_id, geo_id)
);


-- =============================================
-- FACT TABLE: National economic context
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'fact_economic_context')
CREATE TABLE fact_economic_context (
    date_id                 INT PRIMARY KEY,

    -- INSEE indicators (national)
    confiance_menages       DECIMAL(6,2),
    climat_affaires_indus   DECIMAL(6,2),
    climat_affaires_bat     DECIMAL(6,2),
    opinion_achats          DECIMAL(6,2),
    situation_fin_future    DECIMAL(6,2),
    ipi_manufacturing       DECIMAL(6,2),

    -- Eurostat indicators (national)
    ipi_hvac_c28            DECIMAL(6,2),
    ipi_hvac_c2825          DECIMAL(6,2),

    FOREIGN KEY (date_id) REFERENCES dim_time(date_id)
);


-- =============================================
-- INDEXES
-- =============================================
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_fact_hvac_date')
    CREATE INDEX idx_fact_hvac_date ON fact_hvac_installations(date_id);
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_fact_hvac_geo')
    CREATE INDEX idx_fact_hvac_geo ON fact_hvac_installations(geo_id);
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_fact_hvac_date_geo')
    CREATE INDEX idx_fact_hvac_date_geo ON fact_hvac_installations(date_id, geo_id);
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_dim_geo_dept')
    CREATE INDEX idx_dim_geo_dept ON dim_geo(dept_code);


-- =============================================
-- REFERENCE DATA: dim_geo
-- =============================================
-- MERGE for idempotency (equivalent to INSERT OR IGNORE)
MERGE dim_geo AS target
USING (VALUES
    ('01', 'Ain',           'Bourg-en-Bresse', 46.21, 5.23, '84'),
    ('07', 'Ardèche',       'Privas',          44.74, 4.60, '84'),
    ('26', 'Drôme',         'Valence',         44.93, 4.89, '84'),
    ('38', 'Isère',         'Grenoble',        45.19, 5.72, '84'),
    ('42', 'Loire',         'Saint-Etienne',   45.44, 4.39, '84'),
    ('69', 'Rhône',         'Lyon',            45.76, 4.84, '84'),
    ('73', 'Savoie',        'Chambéry',        45.57, 5.92, '84'),
    ('74', 'Haute-Savoie',  'Annecy',          45.90, 6.13, '84')
) AS source (dept_code, dept_name, city_ref, latitude, longitude, region_code)
ON target.dept_code = source.dept_code
WHEN NOT MATCHED THEN
    INSERT (dept_code, dept_name, city_ref, latitude, longitude, region_code)
    VALUES (source.dept_code, source.dept_name, source.city_ref,
            source.latitude, source.longitude, source.region_code);


-- =============================================
-- REFERENCE DATA: dim_equipment_type
-- =============================================
MERGE dim_equipment_type AS target
USING (VALUES
    ('PAC air-air',              'mixte',          'electricite', 1),
    ('PAC air-eau',              'chauffage',      'electricite', 1),
    ('PAC géothermique',         'chauffage',      'electricite', 1),
    ('Climatisation réversible', 'climatisation',  'electricite', 0),
    ('Climatisation split',      'climatisation',  'electricite', 0),
    ('Chaudière gaz',            'chauffage',      'gaz',         0),
    ('Chaudière fioul',          'chauffage',      'fioul',       0),
    ('Chaudière bois',           'chauffage',      'bois',        1),
    ('Radiateur électrique',     'chauffage',      'electricite', 0),
    ('Poêle à bois',             'chauffage',      'bois',        1)
) AS source (label, categorie, energie_source, is_renewable)
ON target.label = source.label
WHEN NOT MATCHED THEN
    INSERT (label, categorie, energie_source, is_renewable)
    VALUES (source.label, source.categorie, source.energie_source, source.is_renewable);


-- =============================================
-- REFERENCE DATA: dim_time (2019-01 to 2025-12)
-- =============================================
-- Generated via recursive CTE (more elegant than 84 rows)
;WITH months AS (
    SELECT 201901 AS date_id, 2019 AS year, 1 AS month
    UNION ALL
    SELECT
        CASE WHEN month = 12
            THEN (year + 1) * 100 + 1
            ELSE year * 100 + month + 1
        END,
        CASE WHEN month = 12 THEN year + 1 ELSE year END,
        CASE WHEN month = 12 THEN 1 ELSE month + 1 END
    FROM months
    WHERE date_id < 202512
)
MERGE dim_time AS target
USING (
    SELECT
        date_id,
        year,
        month,
        (month - 1) / 3 + 1 AS quarter,
        CASE WHEN month IN (10, 11, 12, 1, 2, 3) THEN 1 ELSE 0 END AS is_heating,
        CASE WHEN month IN (6, 7, 8, 9) THEN 1 ELSE 0 END AS is_cooling
    FROM months
) AS source
ON target.date_id = source.date_id
WHEN NOT MATCHED THEN
    INSERT (date_id, year, month, quarter, is_heating, is_cooling)
    VALUES (source.date_id, source.year, source.month,
            source.quarter, source.is_heating, source.is_cooling)
OPTION (MAXRECURSION 100);
