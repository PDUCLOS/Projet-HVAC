# -*- coding: utf-8 -*-
"""
Configuration centralisée du projet HVAC Market Analysis.
=========================================================

Regroupe TOUS les paramètres du projet en un seul endroit.
Les valeurs sont chargées depuis le fichier .env (via python-dotenv)
avec des valeurs par défaut sensées.

Architecture :
    ProjectConfig
    ├── GeoConfig       — Région, départements, villes de référence
    ├── TimeConfig      — Périodes de collecte et split train/val/test
    ├── NetworkConfig   — Timeouts, retries, rate limiting
    ├── DatabaseConfig  — Type de BDD et chemin/credentials
    └── ModelConfig     — Hyperparamètres ML, lags, rolling windows

Usage:
    >>> from config.settings import config
    >>> print(len(config.geo.departments))  # 96 depts (France)
    96
    >>> print(config.geo.region_code)
    'FR'

Extensibilité:
    Pour ajouter un nouveau paramètre :
    1. Ajouter l'attribut dans la dataclass appropriée
    2. Ajouter la variable d'environnement correspondante dans .env.example
    3. Mapper la variable dans from_env() si nécessaire
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Charger le .env dès l'import du module
load_dotenv()


# =============================================================================
# Sous-configurations thématiques
# =============================================================================

# =============================================================================
# Reference geographique : 96 departements de France metropolitaine
# =============================================================================
# Chaque entree = prefecture du departement avec coordonnees GPS
# Organise par region (13 regions metropolitaines)
# Usage : GeoConfig filtre automatiquement selon TARGET_DEPARTMENTS

FRANCE_DEPARTMENTS: Dict[str, Dict[str, Any]] = {
    # --- Auvergne-Rhone-Alpes (84) ---
    "Bourg-en-Bresse":     {"lat": 46.21, "lon": 5.23, "dept": "01", "region": "84"},
    "Moulins":             {"lat": 46.57, "lon": 3.33, "dept": "03", "region": "84"},
    "Privas":              {"lat": 44.74, "lon": 4.60, "dept": "07", "region": "84"},
    "Aurillac":            {"lat": 44.93, "lon": 2.44, "dept": "15", "region": "84"},
    "Valence":             {"lat": 44.93, "lon": 4.89, "dept": "26", "region": "84"},
    "Grenoble":            {"lat": 45.19, "lon": 5.72, "dept": "38", "region": "84"},
    "Saint-Etienne":       {"lat": 45.44, "lon": 4.39, "dept": "42", "region": "84"},
    "Le Puy-en-Velay":     {"lat": 45.04, "lon": 3.88, "dept": "43", "region": "84"},
    "Clermont-Ferrand":    {"lat": 45.78, "lon": 3.08, "dept": "63", "region": "84"},
    "Lyon":                {"lat": 45.76, "lon": 4.84, "dept": "69", "region": "84"},
    "Chambery":            {"lat": 45.57, "lon": 5.92, "dept": "73", "region": "84"},
    "Annecy":              {"lat": 45.90, "lon": 6.13, "dept": "74", "region": "84"},
    # --- Bourgogne-Franche-Comte (27) ---
    "Dijon":               {"lat": 47.32, "lon": 5.04, "dept": "21", "region": "27"},
    "Besancon":            {"lat": 47.24, "lon": 6.02, "dept": "25", "region": "27"},
    "Lons-le-Saunier":    {"lat": 46.67, "lon": 5.55, "dept": "39", "region": "27"},
    "Nevers":              {"lat": 46.99, "lon": 3.16, "dept": "58", "region": "27"},
    "Vesoul":              {"lat": 47.62, "lon": 6.15, "dept": "70", "region": "27"},
    "Macon":               {"lat": 46.31, "lon": 4.83, "dept": "71", "region": "27"},
    "Auxerre":             {"lat": 47.80, "lon": 3.57, "dept": "89", "region": "27"},
    "Belfort":             {"lat": 47.64, "lon": 6.86, "dept": "90", "region": "27"},
    # --- Bretagne (53) ---
    "Saint-Brieuc":        {"lat": 48.51, "lon": -2.76, "dept": "22", "region": "53"},
    "Quimper":             {"lat": 48.00, "lon": -4.10, "dept": "29", "region": "53"},
    "Rennes":              {"lat": 48.11, "lon": -1.68, "dept": "35", "region": "53"},
    "Vannes":              {"lat": 47.66, "lon": -2.76, "dept": "56", "region": "53"},
    # --- Centre-Val de Loire (24) ---
    "Bourges":             {"lat": 47.08, "lon": 2.40, "dept": "18", "region": "24"},
    "Chartres":            {"lat": 48.46, "lon": 1.50, "dept": "28", "region": "24"},
    "Chateauroux":         {"lat": 46.81, "lon": 1.69, "dept": "36", "region": "24"},
    "Tours":               {"lat": 47.39, "lon": 0.69, "dept": "37", "region": "24"},
    "Blois":               {"lat": 47.59, "lon": 1.33, "dept": "41", "region": "24"},
    "Orleans":             {"lat": 47.90, "lon": 1.90, "dept": "45", "region": "24"},
    # --- Corse (94) ---
    "Ajaccio":             {"lat": 41.93, "lon": 8.74, "dept": "2A", "region": "94"},
    "Bastia":              {"lat": 42.70, "lon": 9.45, "dept": "2B", "region": "94"},
    # --- Grand Est (44) ---
    "Charleville-Mezieres": {"lat": 49.77, "lon": 4.72, "dept": "08", "region": "44"},
    "Troyes":              {"lat": 48.30, "lon": 4.07, "dept": "10", "region": "44"},
    "Chalons-en-Champagne": {"lat": 48.96, "lon": 4.36, "dept": "51", "region": "44"},
    "Chaumont":            {"lat": 48.11, "lon": 5.14, "dept": "52", "region": "44"},
    "Nancy":               {"lat": 48.69, "lon": 6.18, "dept": "54", "region": "44"},
    "Bar-le-Duc":          {"lat": 48.77, "lon": 5.16, "dept": "55", "region": "44"},
    "Metz":                {"lat": 49.12, "lon": 6.18, "dept": "57", "region": "44"},
    "Strasbourg":          {"lat": 48.57, "lon": 7.75, "dept": "67", "region": "44"},
    "Colmar":              {"lat": 48.08, "lon": 7.36, "dept": "68", "region": "44"},
    "Epinal":              {"lat": 48.17, "lon": 6.45, "dept": "88", "region": "44"},
    # --- Hauts-de-France (32) ---
    "Laon":                {"lat": 49.56, "lon": 3.62, "dept": "02", "region": "32"},
    "Lille":               {"lat": 50.63, "lon": 3.06, "dept": "59", "region": "32"},
    "Beauvais":            {"lat": 49.43, "lon": 2.08, "dept": "60", "region": "32"},
    "Arras":               {"lat": 50.29, "lon": 2.78, "dept": "62", "region": "32"},
    "Amiens":              {"lat": 49.89, "lon": 2.30, "dept": "80", "region": "32"},
    # --- Ile-de-France (11) ---
    "Paris":               {"lat": 48.86, "lon": 2.35, "dept": "75", "region": "11"},
    "Melun":               {"lat": 48.54, "lon": 2.66, "dept": "77", "region": "11"},
    "Versailles":          {"lat": 48.80, "lon": 2.13, "dept": "78", "region": "11"},
    "Evry":                {"lat": 48.63, "lon": 2.44, "dept": "91", "region": "11"},
    "Nanterre":            {"lat": 48.89, "lon": 2.21, "dept": "92", "region": "11"},
    "Bobigny":             {"lat": 48.91, "lon": 2.44, "dept": "93", "region": "11"},
    "Creteil":             {"lat": 48.79, "lon": 2.46, "dept": "94", "region": "11"},
    "Pontoise":            {"lat": 49.05, "lon": 2.10, "dept": "95", "region": "11"},
    # --- Normandie (28) ---
    "Caen":                {"lat": 49.18, "lon": -0.37, "dept": "14", "region": "28"},
    "Evreux":              {"lat": 49.02, "lon": 1.15, "dept": "27", "region": "28"},
    "Saint-Lo":            {"lat": 49.12, "lon": -1.09, "dept": "50", "region": "28"},
    "Alencon":             {"lat": 48.43, "lon": 0.09, "dept": "61", "region": "28"},
    "Rouen":               {"lat": 49.44, "lon": 1.10, "dept": "76", "region": "28"},
    # --- Nouvelle-Aquitaine (75) ---
    "Angouleme":           {"lat": 45.65, "lon": 0.16, "dept": "16", "region": "75"},
    "La Rochelle":         {"lat": 46.16, "lon": -1.15, "dept": "17", "region": "75"},
    "Tulle":               {"lat": 45.27, "lon": 1.77, "dept": "19", "region": "75"},
    "Gueret":              {"lat": 46.17, "lon": 1.87, "dept": "23", "region": "75"},
    "Perigueux":           {"lat": 45.19, "lon": 0.72, "dept": "24", "region": "75"},
    "Bordeaux":            {"lat": 44.84, "lon": -0.58, "dept": "33", "region": "75"},
    "Mont-de-Marsan":      {"lat": 43.89, "lon": -0.50, "dept": "40", "region": "75"},
    "Agen":                {"lat": 44.20, "lon": 0.62, "dept": "47", "region": "75"},
    "Pau":                 {"lat": 43.30, "lon": -0.37, "dept": "64", "region": "75"},
    "Niort":               {"lat": 46.33, "lon": -0.46, "dept": "79", "region": "75"},
    "Poitiers":            {"lat": 46.58, "lon": 0.34, "dept": "86", "region": "75"},
    "Limoges":             {"lat": 45.83, "lon": 1.26, "dept": "87", "region": "75"},
    # --- Occitanie (76) ---
    "Foix":                {"lat": 42.97, "lon": 1.61, "dept": "09", "region": "76"},
    "Carcassonne":         {"lat": 43.21, "lon": 2.35, "dept": "11", "region": "76"},
    "Rodez":               {"lat": 44.35, "lon": 2.57, "dept": "12", "region": "76"},
    "Nimes":               {"lat": 43.84, "lon": 4.36, "dept": "30", "region": "76"},
    "Toulouse":            {"lat": 43.60, "lon": 1.44, "dept": "31", "region": "76"},
    "Auch":                {"lat": 43.65, "lon": 0.59, "dept": "32", "region": "76"},
    "Montpellier":         {"lat": 43.61, "lon": 3.88, "dept": "34", "region": "76"},
    "Cahors":              {"lat": 44.45, "lon": 1.44, "dept": "46", "region": "76"},
    "Mende":               {"lat": 44.52, "lon": 3.50, "dept": "48", "region": "76"},
    "Tarbes":              {"lat": 43.23, "lon": 0.08, "dept": "65", "region": "76"},
    "Perpignan":           {"lat": 42.70, "lon": 2.90, "dept": "66", "region": "76"},
    "Albi":                {"lat": 43.93, "lon": 2.15, "dept": "81", "region": "76"},
    "Montauban":           {"lat": 44.02, "lon": 1.35, "dept": "82", "region": "76"},
    # --- Pays de la Loire (52) ---
    "Nantes":              {"lat": 47.22, "lon": -1.55, "dept": "44", "region": "52"},
    "Angers":              {"lat": 47.47, "lon": -0.56, "dept": "49", "region": "52"},
    "Laval":               {"lat": 48.07, "lon": -0.77, "dept": "53", "region": "52"},
    "Le Mans":             {"lat": 48.00, "lon": 0.20, "dept": "72", "region": "52"},
    "La Roche-sur-Yon":    {"lat": 46.67, "lon": -1.43, "dept": "85", "region": "52"},
    # --- Provence-Alpes-Cote d'Azur (93) ---
    "Digne-les-Bains":    {"lat": 44.09, "lon": 6.24, "dept": "04", "region": "93"},
    "Gap":                 {"lat": 44.56, "lon": 6.08, "dept": "05", "region": "93"},
    "Nice":                {"lat": 43.70, "lon": 7.27, "dept": "06", "region": "93"},
    "Marseille":           {"lat": 43.30, "lon": 5.37, "dept": "13", "region": "93"},
    "Toulon":              {"lat": 43.12, "lon": 5.93, "dept": "83", "region": "93"},
    "Avignon":             {"lat": 43.95, "lon": 4.81, "dept": "84", "region": "93"},
}

# Mapping code region → nom
REGION_NAMES: Dict[str, str] = {
    "84": "Auvergne-Rhone-Alpes",
    "27": "Bourgogne-Franche-Comte",
    "53": "Bretagne",
    "24": "Centre-Val de Loire",
    "94": "Corse",
    "44": "Grand Est",
    "32": "Hauts-de-France",
    "11": "Ile-de-France",
    "28": "Normandie",
    "75": "Nouvelle-Aquitaine",
    "76": "Occitanie",
    "52": "Pays de la Loire",
    "93": "Provence-Alpes-Cote d'Azur",
    "FR": "France metropolitaine",
}


def _get_departments_for_scope(scope: str) -> List[str]:
    """Retourne la liste des departements selon le perimetre choisi.

    Args:
        scope: Code region INSEE ("84" pour AURA) ou "FR" pour toute la France.

    Returns:
        Liste des codes departements.
    """
    if scope.upper() == "FR":
        return sorted({info["dept"] for info in FRANCE_DEPARTMENTS.values()})
    return sorted(
        info["dept"]
        for info in FRANCE_DEPARTMENTS.values()
        if info["region"] == scope
    )


def _get_cities_for_departments(departments: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retourne les villes de reference pour les departements donnes.

    Args:
        departments: Liste des codes departements.

    Returns:
        Dictionnaire {ville: {lat, lon, dept, region}}.
    """
    dept_set = set(departments)
    return {
        city: info
        for city, info in FRANCE_DEPARTMENTS.items()
        if info["dept"] in dept_set
    }


@dataclass(frozen=True)
class GeoConfig:
    """Configuration geographique du perimetre d'etude.

    Supporte 3 modes :
    - Region specifique : TARGET_REGION=84 (AURA, 12 depts)
    - France complete   : TARGET_REGION=FR (96 departements)
    - Liste manuelle    : TARGET_DEPARTMENTS=69,38,42 (override)

    Attributes:
        region_code: Code INSEE de la region ("FR" pour toute la France).
        departments: Liste des codes departements a analyser.
        cities: Dictionnaire des villes de reference avec coordonnees.
    """
    region_code: str = "FR"
    departments: List[str] = field(
        default_factory=lambda: _get_departments_for_scope("FR")
    )
    cities: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(FRANCE_DEPARTMENTS)
    )


@dataclass(frozen=True)
class TimeConfig:
    """Configuration temporelle : périodes de collecte et split ML.

    Le split temporel est conçu pour respecter la chronologie :
    - Train  : période la plus ancienne (apprentissage)
    - Val    : période intermédiaire (tuning hyperparamètres)
    - Test   : période la plus récente (évaluation finale)

    Attributes:
        start_date: Début de la période de collecte (format YYYY-MM-DD).
        end_date: Fin de la période de collecte.
        dpe_start_date: Début des DPE v2 (juillet 2021) — variable cible.
        train_end: Fin de la période d'entraînement.
        val_end: Fin de la période de validation.
        frequency: Fréquence temporelle du dataset ML ('MS' = Month Start).
    """
    start_date: str = "2019-01-01"
    end_date: str = "2026-02-28"
    dpe_start_date: str = "2021-07-01"    # DPE v2 disponible depuis cette date
    train_end: str = "2024-06-30"          # 36 mois de train
    val_end: str = "2024-12-31"            # 6 mois de validation
    frequency: str = "MS"                  # Month Start pour pandas


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration réseau pour les collecteurs HTTP.

    Attributes:
        request_timeout: Timeout HTTP en secondes par requête.
        max_retries: Nombre max de tentatives sur erreur réseau/serveur.
        retry_backoff_factor: Facteur multiplicatif entre les retries
                              (1.0 → 1s, 2s, 4s de délai entre retries).
        rate_limit_delay: Pause minimale entre deux appels API (en secondes).
    """
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    rate_limit_delay: float = 0.5


@dataclass(frozen=True)
class DatabaseConfig:
    """Configuration de la base de données.

    Trois moteurs supportés, par ordre de complexité :
      1. SQLite  (défaut) — aucune installation, fichier local
      2. SQL Server (mssql) — via pyodbc, pour environnement entreprise
      3. PostgreSQL — pour déploiement cloud/production

    Le basculement se fait uniquement via la variable DB_TYPE dans .env.

    Attributes:
        db_type: Type de BDD ('sqlite', 'mssql' ou 'postgresql').
        db_path: Chemin du fichier SQLite (ignoré si autre moteur).
        db_host: Hôte du serveur (SQL Server ou PostgreSQL).
        db_port: Port du serveur (1433 pour SQL Server, 5432 pour PostgreSQL).
        db_name: Nom de la base.
        db_user: Utilisateur (vide = Windows Authentication pour SQL Server).
        db_password: Mot de passe (vide = Windows Authentication pour SQL Server).
        db_driver: Driver ODBC pour SQL Server (ex: 'ODBC Driver 17 for SQL Server').
        allow_non_local: Autorise l'utilisation de bases non locales.
    """
    db_type: str = "sqlite"
    db_path: str = "data/hvac_market.db"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "hvac_market"
    db_user: str = ""
    db_password: str = ""
    db_driver: str = "ODBC Driver 17 for SQL Server"
    allow_non_local: bool = False

    @property
    def connection_string(self) -> str:
        """Génère la chaîne de connexion SQLAlchemy selon le moteur choisi.

        Moteurs supportés :
          - sqlite   → sqlite:///data/hvac_market.db
          - mssql    → mssql+pyodbc://user:pass@host:port/db?driver=...
                        Si user est vide → Windows Authentication (Trusted_Connection=yes)
          - postgresql → postgresql://user:pass@host:port/db

        Returns:
            URL de connexion au format SQLAlchemy.

        Raises:
            ValueError: Si une base non locale est utilisée sans autorisation,
                        ou si le db_type est inconnu.
        """
        # SQLite : toujours local, pas de restriction
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_path}"

        # Pour les bases réseau, vérifier l'autorisation explicite
        if not self.allow_non_local:
            raise ValueError(
                f"Base de données '{self.db_type}' non locale désactivée. "
                f"Définir ALLOW_NON_LOCAL=true dans .env pour autoriser."
            )

        # SQL Server via pyodbc
        if self.db_type == "mssql":
            driver_encoded = self.db_driver.replace(" ", "+")
            if self.db_user:
                # Authentification SQL Server (login/mot de passe)
                return (
                    f"mssql+pyodbc://{self.db_user}:{self.db_password}"
                    f"@{self.db_host}:{self.db_port}/{self.db_name}"
                    f"?driver={driver_encoded}"
                )
            # Windows Authentication (pas de login/mdp)
            return (
                f"mssql+pyodbc://@{self.db_host}:{self.db_port}/{self.db_name}"
                f"?driver={driver_encoded}&Trusted_Connection=yes"
            )

        # PostgreSQL
        if self.db_type == "postgresql":
            return (
                f"postgresql://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )

        raise ValueError(
            f"Type de BDD inconnu : '{self.db_type}'. "
            f"Valeurs acceptées : sqlite, mssql, postgresql."
        )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration des modèles ML.

    NOTES AUDIT :
    - Les lags sont limités à 6 mois max (lag_12m élimine 33% des données).
    - La saisonnalité annuelle est capturée par les features calendaires.
    - Les rolling windows sont plafonnées à 6 mois.
    - Le Transformer a été retiré (insuffisance de données).

    Attributes:
        max_lag_months: Lag maximum en mois pour les features retardées.
        rolling_windows: Tailles des fenêtres glissantes (en mois).
        hdd_base_temp: Température de base pour les HDD (°C).
        cdd_base_temp: Température de base pour les CDD (°C).
        lightgbm_params: Hyperparamètres LightGBM par défaut
                         (régularisation forte pour petit dataset).
    """
    max_lag_months: int = 6
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6])
    hdd_base_temp: float = 18.0  # Heating Degree Days base temperature
    cdd_base_temp: float = 18.0  # Cooling Degree Days base temperature
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
    })


# =============================================================================
# Configuration principale (agrège toutes les sous-configs)
# =============================================================================

@dataclass(frozen=True)
class ProjectConfig:
    """Configuration globale du projet.

    Regroupe toutes les sous-configurations thématiques.
    Peut être instanciée directement ou via la factory `from_env()`.

    Attributes:
        geo: Configuration géographique.
        time: Configuration temporelle.
        network: Configuration réseau.
        database: Configuration BDD.
        model: Configuration ML.
        raw_data_dir: Répertoire des données brutes.
        processed_data_dir: Répertoire des données nettoyées.
        features_data_dir: Répertoire des features ML.
        log_level: Niveau de logging global.
    """
    geo: GeoConfig = field(default_factory=GeoConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    features_data_dir: Path = Path("data/features")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> ProjectConfig:
        """Construit la configuration à partir des variables d'environnement.

        Charge les variables depuis le fichier .env et les mappe
        sur les sous-configurations. Utilise les valeurs par défaut
        si une variable est absente.

        Returns:
            Instance de ProjectConfig complètement initialisée.
        """
        region_code = os.getenv("TARGET_REGION", "FR")
        # Si TARGET_DEPARTMENTS est defini, l'utiliser en priorite
        # Sinon, deduire les departements depuis TARGET_REGION
        departments_env = os.getenv("TARGET_DEPARTMENTS", "")
        if departments_env:
            departments = departments_env.split(",")
        else:
            departments = _get_departments_for_scope(region_code)
        cities = _get_cities_for_departments(departments)

        return cls(
            geo=GeoConfig(
                region_code=region_code,
                departments=departments,
                cities=cities,
            ),
            time=TimeConfig(
                start_date=os.getenv("DATA_START_DATE", "2019-01-01"),
                end_date=os.getenv("DATA_END_DATE", "2026-02-28"),
            ),
            network=NetworkConfig(
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                retry_backoff_factor=float(os.getenv("RETRY_BACKOFF", "1.0")),
                rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "0.5")),
            ),
            database=DatabaseConfig(
                db_type=os.getenv("DB_TYPE", "sqlite"),
                db_path=os.getenv("DB_PATH", "data/hvac_market.db"),
                db_host=os.getenv("DB_HOST", "localhost"),
                db_port=int(os.getenv("DB_PORT", "1433")),
                db_name=os.getenv("DB_NAME", "hvac_market"),
                db_user=os.getenv("DB_USER", ""),
                db_password=os.getenv("DB_PASSWORD", ""),
                db_driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
                allow_non_local=os.getenv("ALLOW_NON_LOCAL", "").lower() in ("true", "1", "yes"),
            ),
            raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
            features_data_dir=Path(os.getenv("FEATURES_DATA_DIR", "data/features")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# =============================================================================
# Instance globale — importable directement
# =============================================================================
# Usage: from config.settings import config
config = ProjectConfig.from_env()
