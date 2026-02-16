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
    >>> print(config.geo.departments)
    ['01', '07', '26', '38', '42', '69', '73', '74']
    >>> print(config.time.start_date)
    '2019-01-01'

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

@dataclass(frozen=True)
class GeoConfig:
    """Configuration géographique du périmètre d'étude.

    Attributes:
        region_code: Code INSEE de la région cible (84 = AURA).
        departments: Liste des codes départements à analyser.
        cities: Dictionnaire des villes de référence avec coordonnées.
                Chaque ville sert de point de collecte météo pour son département.
                NOTE AUDIT: Clermont-Ferrand (63) et Villeurbanne (69 doublon)
                ont été retirés suite à l'audit de cohérence.
    """
    region_code: str = "84"
    departments: List[str] = field(default_factory=lambda: [
        # AURA (Auvergne-Rhône-Alpes) — 8 départements
        "01", "07", "26", "38", "42", "69", "73", "74",
        # BFC (Bourgogne-Franche-Comté) — 8 départements
        "21", "25", "39", "58", "70", "71", "89", "90",
        # PACA (Provence-Alpes-Côte d'Azur) — 6 départements
        "04", "05", "06", "13", "83", "84",
    ])
    # Une seule ville de référence par département pour éviter les doublons
    # Clé = nom de ville, Valeur = {lat, lon, dept}
    cities: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # --- AURA ---
        "Lyon":            {"lat": 45.76, "lon": 4.84, "dept": "69"},
        "Grenoble":        {"lat": 45.19, "lon": 5.72, "dept": "38"},
        "Saint-Etienne":   {"lat": 45.44, "lon": 4.39, "dept": "42"},
        "Annecy":          {"lat": 45.90, "lon": 6.13, "dept": "74"},
        "Valence":         {"lat": 44.93, "lon": 4.89, "dept": "26"},
        "Chambery":        {"lat": 45.57, "lon": 5.92, "dept": "73"},
        "Bourg-en-Bresse": {"lat": 46.21, "lon": 5.23, "dept": "01"},
        "Privas":          {"lat": 44.74, "lon": 4.60, "dept": "07"},
        # --- BFC ---
        "Dijon":           {"lat": 47.32, "lon": 5.04, "dept": "21"},
        "Besancon":        {"lat": 47.24, "lon": 6.02, "dept": "25"},
        "Lons-le-Saunier": {"lat": 46.67, "lon": 5.55, "dept": "39"},
        "Nevers":          {"lat": 46.99, "lon": 3.16, "dept": "58"},
        "Vesoul":          {"lat": 47.62, "lon": 6.16, "dept": "70"},
        "Macon":           {"lat": 46.31, "lon": 4.83, "dept": "71"},
        "Auxerre":         {"lat": 47.80, "lon": 3.57, "dept": "89"},
        "Belfort":         {"lat": 47.64, "lon": 6.86, "dept": "90"},
        # --- PACA ---
        "Digne-les-Bains": {"lat": 44.09, "lon": 6.24, "dept": "04"},
        "Gap":             {"lat": 44.56, "lon": 6.08, "dept": "05"},
        "Nice":            {"lat": 43.70, "lon": 7.27, "dept": "06"},
        "Marseille":       {"lat": 43.30, "lon": 5.37, "dept": "13"},
        "Toulon":          {"lat": 43.12, "lon": 5.93, "dept": "83"},
        "Avignon":         {"lat": 43.95, "lon": 4.81, "dept": "84"},
    })


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
    end_date: str = "2025-12-31"
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
        return cls(
            geo=GeoConfig(
                region_code=os.getenv("TARGET_REGION", "84"),
                departments=os.getenv(
                    "TARGET_DEPARTMENTS",
                    "01,07,26,38,42,69,73,74,21,25,39,58,70,71,89,90,04,05,06,13,83,84"
                ).split(","),
            ),
            time=TimeConfig(
                start_date=os.getenv("DATA_START_DATE", "2019-01-01"),
                end_date=os.getenv("DATA_END_DATE", "2025-12-31"),
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
