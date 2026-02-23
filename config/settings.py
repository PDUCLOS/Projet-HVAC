# -*- coding: utf-8 -*-
"""
Centralized configuration for the HVAC Market Analysis project.
================================================================

Groups ALL project parameters in a single place.
Values are loaded from the .env file (via python-dotenv)
with sensible default values.

Architecture:
    ProjectConfig
    ├── GeoConfig       — Region, departments, reference cities
    ├── TimeConfig      — Collection periods and train/val/test split
    ├── NetworkConfig   — Timeouts, retries, rate limiting
    ├── DatabaseConfig  — Database type and path/credentials
    └── ModelConfig     — ML hyperparameters, lags, rolling windows

Usage:
    >>> from config.settings import config
    >>> print(len(config.geo.departments))  # 96 depts (France)
    96
    >>> print(config.geo.region_code)
    'FR'

Extensibility:
    To add a new parameter:
    1. Add the attribute in the appropriate dataclass
    2. Add the corresponding environment variable in .env.example
    3. Map the variable in from_env() if necessary
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load the .env file as soon as the module is imported
load_dotenv()


# =============================================================================
# Thematic sub-configurations
# =============================================================================

# =============================================================================
# Geographic reference: 96 departments of metropolitan France
# =============================================================================
# Each entry = department prefecture with GPS coordinates
# Organized by region (13 metropolitan regions)
# Usage: GeoConfig automatically filters based on TARGET_DEPARTMENTS

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

# Mapping region code → name
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

# Mapping department code → department name (all 96 metropolitan departments)
DEPT_NAMES: Dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier",
    "04": "Alpes-de-Haute-Provence", "05": "Hautes-Alpes",
    "06": "Alpes-Maritimes", "07": "Ardeche", "08": "Ardennes",
    "09": "Ariege", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhone", "14": "Calvados", "15": "Cantal",
    "16": "Charente", "17": "Charente-Maritime", "18": "Cher",
    "19": "Correze", "21": "Cote-d'Or", "22": "Cotes-d'Armor",
    "23": "Creuse", "24": "Dordogne", "25": "Doubs", "26": "Drome",
    "27": "Eure", "28": "Eure-et-Loir", "29": "Finistere",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse", "30": "Gard",
    "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Herault", "35": "Ille-et-Vilaine", "36": "Indre",
    "37": "Indre-et-Loire", "38": "Isere", "39": "Jura",
    "40": "Landes", "41": "Loir-et-Cher", "42": "Loire",
    "43": "Haute-Loire", "44": "Loire-Atlantique", "45": "Loiret",
    "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozere",
    "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan",
    "57": "Moselle", "58": "Nievre", "59": "Nord", "60": "Oise",
    "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dome",
    "64": "Pyrenees-Atlantiques", "65": "Hautes-Pyrenees",
    "66": "Pyrenees-Orientales", "67": "Bas-Rhin", "68": "Haut-Rhin",
    "69": "Rhone", "70": "Haute-Saone", "71": "Saone-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime",
    "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sevres", "80": "Somme", "81": "Tarn",
    "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse",
    "85": "Vendee", "86": "Vienne", "87": "Haute-Vienne",
    "88": "Vosges", "89": "Yonne",
    "90": "Territoire de Belfort", "91": "Essonne",
    "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}


# Prefecture elevations (meters above sea level) — source: Open-Meteo Elevation API
# (Copernicus DEM GLO-90, 90m resolution) + IGN BD ALTI validation.
# Used for PAC (heat pump) efficiency modeling: high altitude = lower COP.
PREFECTURE_ELEVATIONS: Dict[str, int] = {
    "01": 224,   # Bourg-en-Bresse
    "02": 82,    # Laon
    "03": 242,   # Moulins
    "04": 608,   # Digne-les-Bains
    "05": 735,   # Gap
    "06": 10,    # Nice
    "07": 300,   # Privas
    "08": 160,   # Charleville-Mezieres
    "09": 390,   # Foix
    "10": 113,   # Troyes
    "11": 111,   # Carcassonne
    "12": 635,   # Rodez
    "13": 12,    # Marseille
    "14": 25,    # Caen
    "15": 631,   # Aurillac
    "16": 100,   # Angouleme
    "17": 8,     # La Rochelle
    "18": 153,   # Bourges
    "19": 455,   # Tulle
    "21": 245,   # Dijon
    "22": 135,   # Saint-Brieuc
    "23": 457,   # Gueret
    "24": 86,    # Perigueux
    "25": 307,   # Besancon
    "26": 126,   # Valence
    "27": 64,    # Evreux
    "28": 142,   # Chartres
    "29": 63,    # Quimper
    "2A": 18,    # Ajaccio
    "2B": 5,     # Bastia
    "30": 39,    # Nimes
    "31": 141,   # Toulouse
    "32": 169,   # Auch
    "33": 6,     # Bordeaux
    "34": 27,    # Montpellier
    "35": 40,    # Rennes
    "36": 155,   # Chateauroux
    "37": 60,    # Tours
    "38": 212,   # Grenoble
    "39": 255,   # Lons-le-Saunier
    "40": 60,    # Mont-de-Marsan
    "41": 73,    # Blois
    "42": 517,   # Saint-Etienne
    "43": 629,   # Le Puy-en-Velay
    "44": 8,     # Nantes
    "45": 116,   # Orleans
    "46": 135,   # Cahors
    "47": 50,    # Agen
    "48": 731,   # Mende
    "49": 47,    # Angers
    "50": 20,    # Saint-Lo
    "51": 83,    # Chalons-en-Champagne
    "52": 318,   # Chaumont
    "53": 90,    # Laval
    "54": 212,   # Nancy
    "55": 188,   # Bar-le-Duc
    "56": 10,    # Vannes
    "57": 178,   # Metz
    "58": 194,   # Nevers
    "59": 20,    # Lille
    "60": 67,    # Beauvais
    "61": 135,   # Alencon
    "62": 57,    # Arras
    "63": 401,   # Clermont-Ferrand
    "64": 190,   # Pau
    "65": 304,   # Tarbes
    "66": 40,    # Perpignan
    "67": 142,   # Strasbourg
    "68": 194,   # Colmar
    "69": 175,   # Lyon
    "70": 221,   # Vesoul
    "71": 180,   # Macon
    "72": 55,    # Le Mans
    "73": 270,   # Chambery
    "74": 448,   # Annecy
    "75": 35,    # Paris
    "76": 12,    # Rouen
    "77": 43,    # Melun
    "78": 132,   # Versailles
    "79": 62,    # Niort
    "80": 34,    # Amiens
    "81": 174,   # Albi
    "82": 82,    # Montauban
    "83": 2,     # Toulon
    "84": 23,    # Avignon
    "85": 72,    # La Roche-sur-Yon
    "86": 120,   # Poitiers
    "87": 300,   # Limoges
    "88": 324,   # Epinal
    "89": 100,   # Auxerre
    "90": 352,   # Belfort
    "91": 80,    # Evry
    "92": 35,    # Nanterre
    "93": 49,    # Bobigny
    "94": 37,    # Creteil
    "95": 30,    # Pontoise
}


def _get_departments_for_scope(scope: str) -> List[str]:
    """Return the list of departments for the chosen scope.

    Args:
        scope: INSEE region code or "FR" for metropolitan France.

    Returns:
        List of department codes.
    """
    if scope.upper() == "FR":
        return sorted({info["dept"] for info in FRANCE_DEPARTMENTS.values()})
    return sorted(
        info["dept"]
        for info in FRANCE_DEPARTMENTS.values()
        if info["region"] == scope
    )


def _get_cities_for_departments(departments: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return the reference cities for the given departments.

    Args:
        departments: List of department codes.

    Returns:
        Dictionary {city: {lat, lon, dept, region}}.
    """
    dept_set = set(departments)
    return {
        city: info
        for city, info in FRANCE_DEPARTMENTS.items()
        if info["dept"] in dept_set
    }


@dataclass(frozen=True)
class GeoConfig:
    """Geographic configuration for the study area.

    Supports 3 modes:
    - Metropolitan France: TARGET_REGION=FR (96 depts)
    - Full France:         TARGET_REGION=FR (96 departments)
    - Manual list:         TARGET_DEPARTMENTS=69,38,42 (override)

    Attributes:
        region_code: INSEE region code ("FR" for all of France).
        departments: List of department codes to analyze.
        cities: Dictionary of reference cities with coordinates.
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
    """Temporal configuration: collection periods and ML split.

    The temporal split is designed to respect chronological order:
    - Train: oldest period (learning)
    - Val:   intermediate period (hyperparameter tuning)
    - Test:  most recent period (final evaluation)

    Attributes:
        start_date: Start of the collection period (format YYYY-MM-DD).
        end_date: End of the collection period.
        dpe_start_date: Start of DPE v2 (July 2021) — target variable.
        train_end: End of the training period.
        val_end: End of the validation period.
        frequency: Temporal frequency of the ML dataset ('MS' = Month Start).
    """
    start_date: str = "2019-01-01"
    end_date: str = "2026-02-28"
    dpe_start_date: str = "2021-07-01"    # DPE v2 available since this date
    train_end: str = "2024-06-30"          # 36 months of training
    val_end: str = "2024-12-31"            # 6 months of validation
    frequency: str = "MS"                  # Month Start for pandas


@dataclass(frozen=True)
class NetworkConfig:
    """Network configuration for HTTP collectors.

    Attributes:
        request_timeout: HTTP timeout in seconds per request.
        max_retries: Maximum number of attempts on network/server errors.
        retry_backoff_factor: Multiplicative factor between retries
                              (1.0 → 1s, 2s, 4s delay between retries).
        rate_limit_delay: Minimum pause between two API calls (in seconds).
    """
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    rate_limit_delay: float = 10.0


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration.

    Three supported engines, in order of complexity:
      1. SQLite  (default) — no installation required, local file
      2. SQL Server (mssql) — via pyodbc, for enterprise environments
      3. PostgreSQL — for cloud/production deployments

    Switching is done solely via the DB_TYPE variable in .env.

    Attributes:
        db_type: Database type ('sqlite', 'mssql', or 'postgresql').
        db_path: SQLite file path (ignored for other engines).
        db_host: Server host (SQL Server or PostgreSQL).
        db_port: Server port (1433 for SQL Server, 5432 for PostgreSQL).
        db_name: Database name.
        db_user: User (empty = Windows Authentication for SQL Server).
        db_password: Password (empty = Windows Authentication for SQL Server).
        db_driver: ODBC driver for SQL Server (e.g., 'ODBC Driver 17 for SQL Server').
        allow_non_local: Allow usage of non-local databases.
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
        """Generate the SQLAlchemy connection string based on the chosen engine.

        Supported engines:
          - sqlite     → sqlite:///data/hvac_market.db
          - mssql      → mssql+pyodbc://user:pass@host:port/db?driver=...
                          If user is empty → Windows Authentication (Trusted_Connection=yes)
          - postgresql → postgresql://user:pass@host:port/db

        Returns:
            Connection URL in SQLAlchemy format.

        Raises:
            ValueError: If a non-local database is used without authorization,
                        or if the db_type is unknown.
        """
        # SQLite: always local, no restriction
        if self.db_type == "sqlite":
            return f"sqlite:///{self.db_path}"

        # For network databases, check explicit authorization
        if not self.allow_non_local:
            raise ValueError(
                f"Non-local database '{self.db_type}' is disabled. "
                f"Set ALLOW_NON_LOCAL=true in .env to enable."
            )

        # SQL Server via pyodbc
        if self.db_type == "mssql":
            driver_encoded = self.db_driver.replace(" ", "+")
            if self.db_user:
                # SQL Server authentication (login/password)
                return (
                    f"mssql+pyodbc://{self.db_user}:{self.db_password}"
                    f"@{self.db_host}:{self.db_port}/{self.db_name}"
                    f"?driver={driver_encoded}"
                )
            # Windows Authentication (no login/password)
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
            f"Unknown database type: '{self.db_type}'. "
            f"Accepted values: sqlite, mssql, postgresql."
        )


@dataclass(frozen=True)
class ThresholdsConfig:
    """Thresholds for weather events and cleaning rules.

    Attributes:
        heatwave_temp: Max temperature above which a day counts as heatwave (°C).
        frost_temp: Min temperature below which a day counts as frost (°C).
        cost_outlier_max: Maximum cost value before flagging as outlier.
    """
    heatwave_temp: float = 35.0
    frost_temp: float = 0.0
    pac_inefficiency_temp: float = -7.0   # Below this, air-source heat pump COP drops critically
    cost_outlier_max: float = 50_000.0


@dataclass(frozen=True)
class ProcessingConfig:
    """Processing configuration for data pipeline.

    Attributes:
        dpe_import_chunk_size: Chunk size for importing DPE into the database.
        dpe_clean_chunk_size: Chunk size for reading DPE during cleaning/merge.
        dpe_api_page_size: Page size for ADEME DPE API calls.
    """
    dpe_import_chunk_size: int = 50_000
    dpe_clean_chunk_size: int = 200_000
    dpe_api_page_size: int = 10_000


@dataclass(frozen=True)
class ModelConfig:
    """ML model configuration.

    AUDIT NOTES:
    - Lags are limited to 6 months max (lag_12m eliminates 33% of data).
    - Annual seasonality is captured by calendar features.
    - Rolling windows are capped at 6 months.
    - The Transformer was removed (insufficient data).

    Attributes:
        max_lag_months: Maximum lag in months for lagged features.
        rolling_windows: Rolling window sizes (in months).
        hdd_base_temp: Base temperature for HDD (°C).
        cdd_base_temp: Base temperature for CDD (°C).
        lightgbm_params: Default LightGBM hyperparameters
                         (strong regularization for small dataset).
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
# Main configuration (aggregates all sub-configs)
# =============================================================================

@dataclass(frozen=True)
class ProjectConfig:
    """Global project configuration.

    Groups all thematic sub-configurations.
    Can be instantiated directly or via the `from_env()` factory method.

    Attributes:
        geo: Geographic configuration.
        time: Temporal configuration.
        network: Network configuration.
        database: Database configuration.
        model: ML configuration.
        raw_data_dir: Directory for raw data.
        processed_data_dir: Directory for cleaned data.
        features_data_dir: Directory for ML features.
        log_level: Global logging level.
    """
    geo: GeoConfig = field(default_factory=GeoConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    features_data_dir: Path = Path("data/features")
    models_dir: Path = Path("data/models")
    log_level: str = "INFO"

    @property
    def features_dataset_path(self) -> Path:
        """Path to the features dataset CSV."""
        return self.features_data_dir / "hvac_features_dataset.csv"

    @property
    def ml_dataset_path(self) -> Path:
        """Path to the ML dataset CSV."""
        return self.features_data_dir / "hvac_ml_dataset.csv"

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return Path(self.database.db_path)

    @property
    def outlier_report_path(self) -> Path:
        """Path to the outlier detection report."""
        return Path("data/analysis") / "outlier_report.json"

    @property
    def evaluation_results_path(self) -> Path:
        """Path to the evaluation results JSON."""
        return self.models_dir / "evaluation_results.json"

    @property
    def pcloud_sync_state_path(self) -> Path:
        """Path to the pCloud synchronization state file."""
        return Path("data/.pcloud_sync_state.json")

    @classmethod
    def from_env(cls) -> ProjectConfig:
        """Build the configuration from environment variables.

        Loads variables from the .env file and maps them
        to sub-configurations. Uses default values
        if a variable is absent.

        Returns:
            Fully initialized ProjectConfig instance.
        """
        region_code = os.getenv("TARGET_REGION", "FR")
        # If TARGET_DEPARTMENTS is defined, use it with priority
        # Otherwise, derive departments from TARGET_REGION
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
                rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "10.0")),
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
            thresholds=ThresholdsConfig(
                heatwave_temp=float(os.getenv("HEATWAVE_TEMP", "35.0")),
                frost_temp=float(os.getenv("FROST_TEMP", "0.0")),
                cost_outlier_max=float(os.getenv("COST_OUTLIER_MAX", "50000")),
            ),
            processing=ProcessingConfig(
                dpe_import_chunk_size=int(os.getenv("DPE_IMPORT_CHUNK_SIZE", "50000")),
                dpe_clean_chunk_size=int(os.getenv("DPE_CLEAN_CHUNK_SIZE", "200000")),
                dpe_api_page_size=int(os.getenv("DPE_API_PAGE_SIZE", "10000")),
            ),
            raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
            features_data_dir=Path(os.getenv("FEATURES_DATA_DIR", "data/features")),
            models_dir=Path(os.getenv("MODELS_DIR", "data/models")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# =============================================================================
# Global instance — directly importable
# =============================================================================
# Usage: from config.settings import config
config = ProjectConfig.from_env()
