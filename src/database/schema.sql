-- =============================================================================
-- HVAC Market Analysis — Schéma en étoile (Star Schema)
-- =============================================================================
-- Base : SQLite (compatible PostgreSQL avec ajustements mineurs)
--
-- ARCHITECTURE :
--   Ce schéma sépare les données en deux catégories :
--   1. Données géo-temporelles (grain = mois × département)
--      → Table de faits : fact_hvac_installations
--   2. Données économiques nationales (grain = mois seulement)
--      → Table de faits : fact_economic_context
--
--   Cette séparation évite de dupliquer artificiellement les indicateurs
--   INSEE (nationaux) sur 8 départements.
--
-- CORRECTIONS AUDIT :
--   - Suppression de la duplication fact/dim des indicateurs économiques
--   - Ajout de FK explicite vers dim_time pour fact_economic_context
--   - dim_equipment transformée en table de référence
--   - Contrainte UNIQUE(date_id, geo_id) sur la table de faits
-- =============================================================================


-- =============================================
-- DIMENSION : Temps
-- =============================================
-- Grain le plus fin = mois (format YYYYMM)
-- Utilisée comme FK par les deux tables de faits
CREATE TABLE IF NOT EXISTS dim_time (
    date_id     INTEGER PRIMARY KEY,   -- Format YYYYMM (ex: 202401 = janvier 2024)
    year        INTEGER NOT NULL,      -- Année (2019-2025)
    month       INTEGER NOT NULL,      -- Mois (1-12)
    quarter     INTEGER NOT NULL,      -- Trimestre (1-4)
    is_heating  BOOLEAN NOT NULL,      -- Saison de chauffage (octobre-mars)
    is_cooling  BOOLEAN NOT NULL,      -- Saison de climatisation (juin-septembre)

    -- Contraintes de cohérence
    CHECK (month BETWEEN 1 AND 12),
    CHECK (quarter BETWEEN 1 AND 4),
    CHECK (year BETWEEN 2015 AND 2030)
);


-- =============================================
-- DIMENSION : Géographie
-- =============================================
-- Une ligne par département de la région AURA
-- Chaque département a une ville de référence pour la météo
CREATE TABLE IF NOT EXISTS dim_geo (
    geo_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    dept_code   VARCHAR(3)  NOT NULL UNIQUE,  -- Code département (01, 07, etc.)
    dept_name   VARCHAR(50) NOT NULL,         -- Nom du département
    city_ref    VARCHAR(50) NOT NULL,         -- Ville de référence météo
    latitude    DECIMAL(7,4),                 -- Latitude de la ville de référence
    longitude   DECIMAL(7,4),                -- Longitude de la ville de référence
    region_code VARCHAR(3) DEFAULT '84'       -- Code région (84 = AURA)
);


-- =============================================
-- DIMENSION : Type d'équipement (table de référence)
-- =============================================
-- Catalogue des types d'équipements HVAC identifiables dans les DPE.
-- N'est PAS une FK dans la table de faits (les faits sont agrégés
-- par mois×département, pas par type d'équipement individuel).
-- Sert pour les analyses détaillées et le filtrage des DPE bruts.
CREATE TABLE IF NOT EXISTS dim_equipment_type (
    equip_type_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    label           VARCHAR(100) NOT NULL,    -- Ex: "PAC air-air", "Chaudière gaz"
    categorie       VARCHAR(50)  NOT NULL,    -- "chauffage", "climatisation", "mixte"
    energie_source  VARCHAR(50),              -- "electricite", "gaz", "fioul", "bois"
    is_renewable    BOOLEAN DEFAULT FALSE     -- Énergie renouvelable (PAC, bois)
);


-- =============================================
-- TABLE DE FAITS : Installations HVAC
-- =============================================
-- Grain = mois × département
-- Contient les métriques spécifiques à chaque département :
--   - Comptages d'installations (variable cible ML)
--   - Météo locale (HDD/CDD)
--   - Permis de construire locaux
--   - Consommation énergie locale
CREATE TABLE IF NOT EXISTS fact_hvac_installations (
    fact_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    date_id                 INTEGER NOT NULL,   -- FK → dim_time (YYYYMM)
    geo_id                  INTEGER NOT NULL,   -- FK → dim_geo

    -- === Variable cible (Y) : proxy installations via DPE ===
    nb_dpe_total            INTEGER,            -- Total DPE du mois/département
    nb_installations_pac    INTEGER,            -- DPE mentionnant une PAC
    nb_installations_clim   INTEGER,            -- DPE mentionnant une climatisation
    nb_dpe_classe_ab        INTEGER,            -- DPE classe A ou B (bâtiment performant)

    -- === Features météo (spécifiques au département) ===
    temp_mean               DECIMAL(5,2),       -- Température moyenne mensuelle (°C)
    temp_max                DECIMAL(5,2),       -- Température max mensuelle (°C)
    temp_min                DECIMAL(5,2),       -- Température min mensuelle (°C)
    heating_degree_days     DECIMAL(8,2),       -- HDD mensuel (somme, base 18°C)
    cooling_degree_days     DECIMAL(8,2),       -- CDD mensuel (somme, base 18°C)
    precipitation_cumul     DECIMAL(8,2),       -- Précipitations cumulées (mm)
    nb_jours_canicule       INTEGER,            -- Jours avec T > 35°C
    nb_jours_gel            INTEGER,            -- Jours avec T < 0°C

    -- === Features immobilier (spécifiques au département) ===
    nb_permis_construire    INTEGER,            -- Nombre de permis accordés
    nb_logements_autorises  INTEGER,            -- Nombre de logements autorisés

    -- === Features énergie (spécifiques au département) ===
    conso_elec_mwh          DECIMAL(12,2),      -- Consommation électrique (MWh)
    conso_gaz_mwh           DECIMAL(12,2),      -- Consommation gaz (MWh)

    -- Contraintes
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id),
    FOREIGN KEY (geo_id) REFERENCES dim_geo(geo_id),
    UNIQUE(date_id, geo_id)  -- Une seule ligne par mois × département
);


-- =============================================
-- TABLE DE FAITS : Contexte économique national
-- =============================================
-- Grain = mois seulement (données nationales, pas départementales)
-- Les indicateurs INSEE et Eurostat sont identiques pour tous les départements.
-- La jointure avec fact_hvac_installations se fait via date_id.
CREATE TABLE IF NOT EXISTS fact_economic_context (
    date_id                 INTEGER PRIMARY KEY, -- FK → dim_time (YYYYMM)

    -- === Indicateurs INSEE (nationaux, mensuels) ===
    confiance_menages       DECIMAL(6,2),       -- Indice synthétique (base 100)
    climat_affaires_indus   DECIMAL(6,2),       -- Climat affaires industrie (base 100)
    climat_affaires_bat     DECIMAL(6,2),       -- Climat affaires bâtiment (base 100)
    opinion_achats          DECIMAL(6,2),       -- Opportunité achats importants (solde)
    situation_fin_future    DECIMAL(6,2),       -- Situation financière future (solde)
    ipi_manufacturing       DECIMAL(6,2),       -- IPI industrie manufacturière (indice)

    -- === Indicateurs Eurostat (nationaux, mensuels) ===
    ipi_hvac_c28            DECIMAL(6,2),       -- IPI machines C28 (indice base 2021)
    ipi_hvac_c2825          DECIMAL(6,2),       -- IPI équipements clim C2825

    -- Contrainte
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id)
);


-- =============================================
-- INDEX pour les performances de requêtage
-- =============================================
CREATE INDEX IF NOT EXISTS idx_fact_hvac_date ON fact_hvac_installations(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_hvac_geo ON fact_hvac_installations(geo_id);
CREATE INDEX IF NOT EXISTS idx_fact_hvac_date_geo ON fact_hvac_installations(date_id, geo_id);
CREATE INDEX IF NOT EXISTS idx_dim_geo_dept ON dim_geo(dept_code);


-- =============================================
-- DONNÉES DE RÉFÉRENCE : dim_geo (8 départements AURA)
-- =============================================
INSERT OR IGNORE INTO dim_geo (dept_code, dept_name, city_ref, latitude, longitude, region_code) VALUES
    ('01', 'Ain',           'Bourg-en-Bresse', 46.21, 5.23, '84'),
    ('07', 'Ardèche',       'Privas',          44.74, 4.60, '84'),
    ('26', 'Drôme',         'Valence',         44.93, 4.89, '84'),
    ('38', 'Isère',         'Grenoble',        45.19, 5.72, '84'),
    ('42', 'Loire',         'Saint-Etienne',   45.44, 4.39, '84'),
    ('69', 'Rhône',         'Lyon',            45.76, 4.84, '84'),
    ('73', 'Savoie',        'Chambéry',        45.57, 5.92, '84'),
    ('74', 'Haute-Savoie',  'Annecy',          45.90, 6.13, '84');


-- =============================================
-- DONNÉES DE RÉFÉRENCE : dim_equipment_type
-- =============================================
INSERT OR IGNORE INTO dim_equipment_type (label, categorie, energie_source, is_renewable) VALUES
    ('PAC air-air',              'mixte',          'electricite', TRUE),
    ('PAC air-eau',              'chauffage',      'electricite', TRUE),
    ('PAC géothermique',         'chauffage',      'electricite', TRUE),
    ('Climatisation réversible', 'climatisation',  'electricite', FALSE),
    ('Climatisation split',      'climatisation',  'electricite', FALSE),
    ('Chaudière gaz',            'chauffage',      'gaz',         FALSE),
    ('Chaudière fioul',          'chauffage',      'fioul',       FALSE),
    ('Chaudière bois',           'chauffage',      'bois',        TRUE),
    ('Radiateur électrique',     'chauffage',      'electricite', FALSE),
    ('Poêle à bois',             'chauffage',      'bois',        TRUE);


-- =============================================
-- DONNÉES DE RÉFÉRENCE : dim_time (2019-01 à 2025-12)
-- =============================================
-- Généré dynamiquement : 84 mois (7 ans × 12 mois)
-- is_heating = octobre à mars (mois 10,11,12,1,2,3)
-- is_cooling = juin à septembre (mois 6,7,8,9)
INSERT OR IGNORE INTO dim_time (date_id, year, month, quarter, is_heating, is_cooling) VALUES
    -- 2019
    (201901, 2019, 1, 1, TRUE, FALSE), (201902, 2019, 2, 1, TRUE, FALSE),
    (201903, 2019, 3, 1, TRUE, FALSE), (201904, 2019, 4, 2, FALSE, FALSE),
    (201905, 2019, 5, 2, FALSE, FALSE), (201906, 2019, 6, 2, FALSE, TRUE),
    (201907, 2019, 7, 3, FALSE, TRUE), (201908, 2019, 8, 3, FALSE, TRUE),
    (201909, 2019, 9, 3, FALSE, TRUE), (201910, 2019, 10, 4, TRUE, FALSE),
    (201911, 2019, 11, 4, TRUE, FALSE), (201912, 2019, 12, 4, TRUE, FALSE),
    -- 2020
    (202001, 2020, 1, 1, TRUE, FALSE), (202002, 2020, 2, 1, TRUE, FALSE),
    (202003, 2020, 3, 1, TRUE, FALSE), (202004, 2020, 4, 2, FALSE, FALSE),
    (202005, 2020, 5, 2, FALSE, FALSE), (202006, 2020, 6, 2, FALSE, TRUE),
    (202007, 2020, 7, 3, FALSE, TRUE), (202008, 2020, 8, 3, FALSE, TRUE),
    (202009, 2020, 9, 3, FALSE, TRUE), (202010, 2020, 10, 4, TRUE, FALSE),
    (202011, 2020, 11, 4, TRUE, FALSE), (202012, 2020, 12, 4, TRUE, FALSE),
    -- 2021
    (202101, 2021, 1, 1, TRUE, FALSE), (202102, 2021, 2, 1, TRUE, FALSE),
    (202103, 2021, 3, 1, TRUE, FALSE), (202104, 2021, 4, 2, FALSE, FALSE),
    (202105, 2021, 5, 2, FALSE, FALSE), (202106, 2021, 6, 2, FALSE, TRUE),
    (202107, 2021, 7, 3, FALSE, TRUE), (202108, 2021, 8, 3, FALSE, TRUE),
    (202109, 2021, 9, 3, FALSE, TRUE), (202110, 2021, 10, 4, TRUE, FALSE),
    (202111, 2021, 11, 4, TRUE, FALSE), (202112, 2021, 12, 4, TRUE, FALSE),
    -- 2022
    (202201, 2022, 1, 1, TRUE, FALSE), (202202, 2022, 2, 1, TRUE, FALSE),
    (202203, 2022, 3, 1, TRUE, FALSE), (202204, 2022, 4, 2, FALSE, FALSE),
    (202205, 2022, 5, 2, FALSE, FALSE), (202206, 2022, 6, 2, FALSE, TRUE),
    (202207, 2022, 7, 3, FALSE, TRUE), (202208, 2022, 8, 3, FALSE, TRUE),
    (202209, 2022, 9, 3, FALSE, TRUE), (202210, 2022, 10, 4, TRUE, FALSE),
    (202211, 2022, 11, 4, TRUE, FALSE), (202212, 2022, 12, 4, TRUE, FALSE),
    -- 2023
    (202301, 2023, 1, 1, TRUE, FALSE), (202302, 2023, 2, 1, TRUE, FALSE),
    (202303, 2023, 3, 1, TRUE, FALSE), (202304, 2023, 4, 2, FALSE, FALSE),
    (202305, 2023, 5, 2, FALSE, FALSE), (202306, 2023, 6, 2, FALSE, TRUE),
    (202307, 2023, 7, 3, FALSE, TRUE), (202308, 2023, 8, 3, FALSE, TRUE),
    (202309, 2023, 9, 3, FALSE, TRUE), (202310, 2023, 10, 4, TRUE, FALSE),
    (202311, 2023, 11, 4, TRUE, FALSE), (202312, 2023, 12, 4, TRUE, FALSE),
    -- 2024
    (202401, 2024, 1, 1, TRUE, FALSE), (202402, 2024, 2, 1, TRUE, FALSE),
    (202403, 2024, 3, 1, TRUE, FALSE), (202404, 2024, 4, 2, FALSE, FALSE),
    (202405, 2024, 5, 2, FALSE, FALSE), (202406, 2024, 6, 2, FALSE, TRUE),
    (202407, 2024, 7, 3, FALSE, TRUE), (202408, 2024, 8, 3, FALSE, TRUE),
    (202409, 2024, 9, 3, FALSE, TRUE), (202410, 2024, 10, 4, TRUE, FALSE),
    (202411, 2024, 11, 4, TRUE, FALSE), (202412, 2024, 12, 4, TRUE, FALSE),
    -- 2025
    (202501, 2025, 1, 1, TRUE, FALSE), (202502, 2025, 2, 1, TRUE, FALSE),
    (202503, 2025, 3, 1, TRUE, FALSE), (202504, 2025, 4, 2, FALSE, FALSE),
    (202505, 2025, 5, 2, FALSE, FALSE), (202506, 2025, 6, 2, FALSE, TRUE),
    (202507, 2025, 7, 3, FALSE, TRUE), (202508, 2025, 8, 3, FALSE, TRUE),
    (202509, 2025, 9, 3, FALSE, TRUE), (202510, 2025, 10, 4, TRUE, FALSE),
    (202511, 2025, 11, 4, TRUE, FALSE), (202512, 2025, 12, 4, TRUE, FALSE);


-- =============================================
-- TABLE BRUTE : DPE (données unitaires)
-- =============================================
-- Stockage des DPE individuels (~1.4M lignes pour AURA)
-- Grain = 1 ligne par DPE
-- Cette table est la source primaire pour calculer les agrégats
-- dans fact_hvac_installations (nb_dpe_total, nb_installations_pac, etc.)
CREATE TABLE IF NOT EXISTS raw_dpe (
    numero_dpe                              VARCHAR(20) PRIMARY KEY,
    date_etablissement_dpe                  DATE,
    date_visite_diagnostiqueur              DATE,

    -- Localisation
    code_postal_ban                         VARCHAR(5),
    code_departement_ban                    VARCHAR(3),
    code_insee_ban                          VARCHAR(5),
    nom_commune_ban                         VARCHAR(100),

    -- Performance énergétique
    etiquette_dpe                           VARCHAR(1),     -- A à G
    etiquette_ges                           VARCHAR(1),     -- A à G
    conso_5_usages_par_m2_ep                DECIMAL(8,2),
    conso_5_usages_par_m2_ef                DECIMAL(8,2),
    emission_ges_5_usages_par_m2            DECIMAL(8,2),

    -- Bâtiment
    type_batiment                           VARCHAR(30),
    annee_construction                      INTEGER,
    periode_construction                    VARCHAR(30),
    surface_habitable_logement              DECIMAL(8,2),
    nombre_niveau_logement                  INTEGER,
    hauteur_sous_plafond                    DECIMAL(4,2),

    -- HVAC : Chauffage
    type_energie_principale_chauffage       VARCHAR(50),
    type_installation_chauffage             VARCHAR(80),
    type_generateur_chauffage_principal     VARCHAR(80),

    -- HVAC : ECS
    type_energie_principale_ecs             VARCHAR(50),

    -- HVAC : Climatisation / Froid
    type_generateur_froid                   VARCHAR(80),

    -- Isolation
    qualite_isolation_enveloppe             VARCHAR(30),
    qualite_isolation_murs                  VARCHAR(30),
    qualite_isolation_menuiseries           VARCHAR(30),
    qualite_isolation_plancher_haut_comble_perdu VARCHAR(30),

    -- Coûts
    cout_chauffage                          DECIMAL(10,2),
    cout_ecs                                DECIMAL(10,2),
    cout_total_5_usages                     DECIMAL(10,2)
);

-- Index pour les requêtes d'agrégation DPE
CREATE INDEX IF NOT EXISTS idx_raw_dpe_date ON raw_dpe(date_etablissement_dpe);
CREATE INDEX IF NOT EXISTS idx_raw_dpe_dept ON raw_dpe(code_departement_ban);
CREATE INDEX IF NOT EXISTS idx_raw_dpe_date_dept ON raw_dpe(date_etablissement_dpe, code_departement_ban);
CREATE INDEX IF NOT EXISTS idx_raw_dpe_etiquette ON raw_dpe(etiquette_dpe)
