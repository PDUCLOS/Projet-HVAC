# -*- coding: utf-8 -*-
"""
Collecteur météo — Open-Meteo Archive API.
===========================================

Récupère les données météorologiques historiques quotidiennes pour les
villes de référence en Auvergne-Rhône-Alpes via l'API Open-Meteo.

Source : https://open-meteo.com/en/docs/historical-weather-api
Authentification : Aucune (API gratuite, fair use ~10 000 appels/jour)

Données collectées (quotidiennes) :
    - temperature_2m_max/min/mean (°C)
    - precipitation_sum (mm)
    - wind_speed_10m_max (km/h)

Données calculées :
    - HDD (Heating Degree Days) = max(0, 18 - temp_mean)
      → Indicateur de la demande en chauffage
    - CDD (Cooling Degree Days) = max(0, temp_mean - 18)
      → Indicateur de la demande en climatisation

NOTE : L'API archive ne fournit pas directement les HDD/CDD,
       ils sont calculés à partir de la température moyenne (base 18°C).

Extensibilité :
    Pour ajouter une ville, il suffit de l'ajouter dans le dictionnaire
    `config.geo.cities` dans config/settings.py.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

import pandas as pd

from src.collectors.base import BaseCollector

# URL de base de l'API Open-Meteo Archive
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables météo quotidiennes à collecter
DAILY_VARIABLES = [
    "temperature_2m_max",     # Température max journalière (°C)
    "temperature_2m_min",     # Température min journalière (°C)
    "temperature_2m_mean",    # Température moyenne journalière (°C)
    "precipitation_sum",      # Précipitations cumulées (mm)
    "wind_speed_10m_max",     # Vitesse max du vent à 10m (km/h)
]


class WeatherCollector(BaseCollector):
    """Collecteur de données météo historiques via Open-Meteo.

    Collecte les données quotidiennes pour chaque ville de référence
    configurée dans `config.geo.cities`. Une ville = un département.

    Le collecteur est tolérant aux pannes : si une ville échoue,
    les autres sont quand même collectées (collecte partielle).

    Auto-enregistré comme 'weather' dans le CollectorRegistry.
    """

    source_name: ClassVar[str] = "weather"
    output_subdir: ClassVar[str] = "weather"
    output_filename: ClassVar[str] = "weather_france.csv"

    def collect(self) -> pd.DataFrame:
        """Collecte les données météo pour toutes les villes AURA.

        Pour chaque ville, effectue un appel API unique couvrant toute
        la période configurée (2019-2025). L'API retourne les données
        quotidiennes en un seul bloc JSON.

        Après la collecte, calcule les HDD et CDD (base 18°C) :
        - HDD (Heating Degree Days) = demande en chauffage
        - CDD (Cooling Degree Days) = demande en climatisation

        Returns:
            DataFrame avec colonnes : time, city, dept,
            temperature_2m_max/min/mean, precipitation_sum,
            wind_speed_10m_max, hdd, cdd.
        """
        # Charger la configuration des villes depuis settings
        from config.settings import config as project_config
        cities = project_config.geo.cities

        all_frames: List[pd.DataFrame] = []
        errors: List[str] = []

        for city, coords in cities.items():
            self.logger.info(
                "Collecte météo : %s (lat=%.2f, lon=%.2f, dept=%s)",
                city, coords["lat"], coords["lon"], coords["dept"],
            )

            # Paramètres de la requête API
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "Europe/Paris",
            }

            try:
                # Appel API — le retry est géré automatiquement par la session
                data = self.fetch_json(OPENMETEO_BASE_URL, params=params)

                # Vérifier que la clé 'daily' existe dans la réponse
                if "daily" not in data:
                    raise ValueError(
                        f"Réponse API inattendue pour {city} : "
                        f"clé 'daily' absente. Clés reçues : {list(data.keys())}"
                    )

                # Convertir en DataFrame et enrichir avec métadonnées
                df = pd.DataFrame(data["daily"])
                df["city"] = city
                df["dept"] = coords["dept"]
                all_frames.append(df)

                self.logger.info(
                    "  ✓ %s : %d jours collectés (%s → %s)",
                    city, len(df),
                    df["time"].iloc[0] if len(df) > 0 else "?",
                    df["time"].iloc[-1] if len(df) > 0 else "?",
                )

            except Exception as exc:
                # Collecter l'erreur mais continuer avec les autres villes
                error_msg = f"Échec pour {city} : {exc}"
                errors.append(error_msg)
                self.logger.error("  ✗ %s", error_msg)
                continue

            # Pause de politesse entre les appels
            self.rate_limit_pause()

        # Vérifier qu'au moins une ville a été collectée
        if not all_frames:
            self.logger.error(
                "Aucune donnée météo collectée. Erreurs : %s", errors
            )
            return pd.DataFrame()

        # Concaténer toutes les villes
        result = pd.concat(all_frames, ignore_index=True)

        # Calculer les HDD et CDD (base 18°C)
        # HDD = demande chauffage : plus il fait froid, plus le HDD est élevé
        # CDD = demande clim : plus il fait chaud, plus le CDD est élevé
        if "temperature_2m_mean" in result.columns:
            result["hdd"] = (18.0 - result["temperature_2m_mean"]).clip(lower=0)
            result["cdd"] = (result["temperature_2m_mean"] - 18.0).clip(lower=0)
            self.logger.info(
                "HDD/CDD calculés (base 18°C) — "
                "HDD moyen=%.1f, CDD moyen=%.1f",
                result["hdd"].mean(), result["cdd"].mean(),
            )

        # Log du bilan si collecte partielle
        if errors:
            self.logger.warning(
                "⚠ Collecte partielle : %d/%d villes réussies. "
                "Villes en erreur : %s",
                len(all_frames), len(cities), errors,
            )

        return result

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure et la qualité des données météo.

        Vérifications :
        1. Présence des colonnes obligatoires
        2. Conversion des types (dates, numériques)
        3. Détection des valeurs nulles anormales (seuil 5%)
        4. Résumé statistique pour diagnostic visuel

        Args:
            df: DataFrame brut issu de collect().

        Returns:
            DataFrame validé avec types corrects.

        Raises:
            ValueError: Si des colonnes critiques sont manquantes.
        """
        # 1. Vérifier les colonnes obligatoires
        required_cols = {"time", "city", "dept", "temperature_2m_mean"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes obligatoires manquantes dans les données météo : {missing}"
            )

        # 2. Conversion des types
        df["time"] = pd.to_datetime(df["time"])

        # 3. Détection des valeurs nulles
        null_pct = df.isnull().mean()
        for col in null_pct[null_pct > 0.05].index:
            self.logger.warning(
                "⚠ Colonne '%s' : %.1f%% de valeurs nulles",
                col, null_pct[col] * 100,
            )

        # 4. Log du résumé de validation
        self.logger.info(
            "Validation OK : %d lignes | %d villes | %s → %s | "
            "T° moy=%.1f°C [%.1f, %.1f]",
            len(df),
            df["city"].nunique(),
            df["time"].min().strftime("%Y-%m-%d"),
            df["time"].max().strftime("%Y-%m-%d"),
            df["temperature_2m_mean"].mean(),
            df["temperature_2m_mean"].min(),
            df["temperature_2m_mean"].max(),
        )

        return df
