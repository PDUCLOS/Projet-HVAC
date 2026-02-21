"""
Dependances partagees pour l'API HVAC.

Charge les modeles ML, le scaler et l'imputer une seule fois au demarrage,
et expose les objets via un singleton AppState accessible par tous les endpoints.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_VERSION: str = "1.0.0"
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = DATA_DIR / "models"
FEATURES_DIR: Path = DATA_DIR / "features"
RAW_DIR: Path = DATA_DIR / "raw"

FEATURES_CSV: Path = FEATURES_DIR / "hvac_features_dataset.csv"
RIDGE_PKL: Path = MODELS_DIR / "ridge_model.pkl"
SCALER_PKL: Path = MODELS_DIR / "scaler.pkl"
IMPUTER_PKL: Path = MODELS_DIR / "imputer.pkl"
TRAINING_RESULTS_CSV: Path = MODELS_DIR / "training_results.csv"
EVALUATION_REPORT: Path = MODELS_DIR / "evaluation_report.txt"

TARGET_COL: str = "nb_installations_pac"

# Colonnes non-features (a exclure avant la prediction)
NON_FEATURE_COLS: set[str] = {
    "date_id", "dept", "nb_dpe_total", TARGET_COL,
    "nb_installations_clim", "nb_dpe_classe_ab",
    "dept_name", "city_ref", "latitude", "longitude",
    "n_valid_features", "pct_valid_features",
    "_outlier_iqr", "_outlier_zscore", "_outlier_iforest",
    "_outlier_consensus",
}

# Dictionnaire complet des 96 departements metropolitains + 2A/2B
DEPARTEMENTS: dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardeche", "08": "Ardennes",
    "09": "Ariege", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhone", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Correze", "2A": "Corse-du-Sud",
    "2B": "Haute-Corse", "21": "Cote-d'Or", "22": "Cotes-d'Armor", "23": "Creuse",
    "24": "Dordogne", "25": "Doubs", "26": "Drome", "27": "Eure",
    "28": "Eure-et-Loir", "29": "Finistere", "30": "Gard", "31": "Haute-Garonne",
    "32": "Gers", "33": "Gironde", "34": "Herault", "35": "Ille-et-Vilaine",
    "36": "Indre", "37": "Indre-et-Loire", "38": "Isere", "39": "Jura",
    "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozere", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse",
    "56": "Morbihan", "57": "Moselle", "58": "Nievre", "59": "Nord",
    "60": "Oise", "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dome",
    "64": "Pyrenees-Atlantiques", "65": "Hautes-Pyrenees", "66": "Pyrenees-Orientales",
    "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhone", "70": "Haute-Saone",
    "71": "Saone-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne",
    "78": "Yvelines", "79": "Deux-Sevres", "80": "Somme", "81": "Tarn",
    "82": "Tarn-et-Garonne", "83": "Var", "84": "Vaucluse", "85": "Vendee",
    "86": "Vienne", "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne",
    "90": "Territoire de Belfort", "91": "Essonne", "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis", "94": "Val-de-Marne", "95": "Val-d'Oise",
}


# ---------------------------------------------------------------------------
# Etat global de l'application (singleton)
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    """Etat global charge une seule fois au demarrage de l'API."""

    ridge_model: Any = None
    scaler: Any = None
    imputer: Any = None
    features_df: pd.DataFrame | None = None
    feature_names: list[str] = field(default_factory=list)
    training_results: pd.DataFrame | None = None
    start_time: float = field(default_factory=time.time)
    model_date: str | None = None

    # -- chargement ---------------------------------------------------------

    def load(self) -> None:
        """Charge tous les artefacts depuis le disque."""
        self._load_model()
        self._load_features()
        self._load_training_results()

    def _load_model(self) -> None:
        """Charge le modele Ridge, le scaler et l'imputer."""
        with open(RIDGE_PKL, "rb") as f:
            self.ridge_model = pickle.load(f)
        with open(SCALER_PKL, "rb") as f:
            self.scaler = pickle.load(f)
        with open(IMPUTER_PKL, "rb") as f:
            self.imputer = pickle.load(f)

        self.feature_names = list(self.ridge_model.feature_names_in_)
        # Date de derniere modification du fichier modele
        mtime = os.path.getmtime(RIDGE_PKL)
        self.model_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    def _load_features(self) -> None:
        """Charge le dataset de features."""
        self.features_df = pd.read_csv(FEATURES_CSV, dtype={"dept": str})

    def _load_training_results(self) -> None:
        """Charge les resultats d'entrainement."""
        if TRAINING_RESULTS_CSV.exists():
            self.training_results = pd.read_csv(TRAINING_RESULTS_CSV)

    # -- prediction ---------------------------------------------------------

    def predict(
        self,
        dept: str,
        horizon: int = 1,
        extra_features: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Genere des predictions pour un departement sur N mois.

        Utilise la derniere ligne connue du departement comme point de depart,
        puis projette iterativement en reinjectant la prediction precedente
        dans les features lag.

        Retourne une liste de dicts {date, valeur_predite, intervalle_bas, intervalle_haut}.
        """
        df_dept = self.features_df[self.features_df["dept"] == dept].copy()
        if df_dept.empty:
            return []

        df_dept = df_dept.sort_values("date_id")
        last_row = df_dept.iloc[-1].copy()
        last_date_str = str(int(last_row["date_id"]))

        # Calcul du RMSE du modele pour les intervalles de confiance
        rmse = self._get_model_rmse()

        results: list[dict[str, Any]] = []
        current_row = last_row.copy()

        for step in range(1, horizon + 1):
            # Avancer la date d'un mois
            next_date = self._next_month(last_date_str, step)

            # Mettre a jour les features temporelles
            current_row = self._advance_features(current_row, next_date)

            # Injecter les features supplementaires
            if extra_features:
                for feat, val in extra_features.items():
                    if feat in current_row.index:
                        current_row[feat] = val

            # Extraire les features dans le bon ordre
            X = current_row[self.feature_names].values.reshape(1, -1).astype(float)
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
            pred = float(self.ridge_model.predict(X)[0])

            # Intervalle de confiance a ~95 % (1.96 * RMSE)
            margin = 1.96 * rmse
            results.append({
                "date": f"{next_date[:4]}-{next_date[4:]}",
                "valeur_predite": round(pred, 2),
                "intervalle_bas": round(pred - margin, 2),
                "intervalle_haut": round(pred + margin, 2),
            })

            # Reinjecter la prediction dans les lags pour le pas suivant
            current_row["nb_installations_pac_lag_1m"] = pred

        return results

    def _get_model_rmse(self) -> float:
        """Recupere le RMSE test du modele Ridge depuis les resultats."""
        if self.training_results is not None:
            ridge_row = self.training_results[
                self.training_results["model"] == "ridge"
            ]
            if not ridge_row.empty:
                val = ridge_row.iloc[0].get("test_rmse", np.nan)
                if not np.isnan(val):
                    return float(val)
        return 1.0  # fallback

    @staticmethod
    def _next_month(base_yyyymm: str, offset: int) -> str:
        """Calcule la date YYYYMM apres 'offset' mois."""
        year = int(base_yyyymm[:4])
        month = int(base_yyyymm[4:])
        total = (year * 12 + month - 1) + offset
        new_year = total // 12
        new_month = total % 12 + 1
        return f"{new_year}{new_month:02d}"

    @staticmethod
    def _advance_features(row: pd.Series, new_date: str) -> pd.Series:
        """Met a jour les colonnes temporelles pour la nouvelle date."""
        row = row.copy()
        year = int(new_date[:4])
        month = int(new_date[4:])
        row["year"] = year
        row["month"] = month
        row["quarter"] = (month - 1) // 3 + 1
        row["is_heating"] = int(month in (1, 2, 3, 10, 11, 12))
        row["is_cooling"] = int(month in (6, 7, 8))
        row["month_sin"] = round(np.sin(2 * np.pi * month / 12), 3)
        row["month_cos"] = round(np.cos(2 * np.pi * month / 12), 3)
        row["year_trend"] = (year - 2020) + month / 12
        return row


# Instance globale
state = AppState()
