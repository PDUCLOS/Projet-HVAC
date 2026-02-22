# -*- coding: utf-8 -*-
"""
Demo data generator for the HVAC project.
===========================================

Generates realistic data for the 96 French departments
to test the full pipeline without network access.

Usage:
    python scripts/generate_demo_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from config.settings import FRANCE_DEPARTMENTS

np.random.seed(42)

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def generate_weather():
    """Generate weather data for 96 prefectures, 2019-2025."""
    print("Generating weather data...")
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    rows = []

    for city_name, info in FRANCE_DEPARTMENTS.items():
        dept_code = info["dept"]
        lat = info["lat"]
        for date in dates:
            day_of_year = date.day_of_year
            # Realistic seasonal temperature (based on latitude)
            base_temp = 15 - (lat - 43) * 1.2
            seasonal = 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            noise = np.random.normal(0, 3)
            temp_mean = base_temp + seasonal + noise

            rows.append({
                "time": date.strftime("%Y-%m-%d"),
                "city": city_name,
                "dept": dept_code,
                "latitude": lat,
                "longitude": info["lon"],
                "temperature_2m_max": round(temp_mean + np.random.uniform(3, 8), 1),
                "temperature_2m_min": round(temp_mean - np.random.uniform(3, 8), 1),
                "temperature_2m_mean": round(temp_mean, 1),
                "precipitation_sum": round(max(0, np.random.exponential(3) - 1), 1),
                "wind_speed_10m_max": round(max(0, np.random.normal(20, 10)), 1),
                "hdd": round(max(0, 18 - temp_mean), 2),
                "cdd": round(max(0, temp_mean - 18), 2),
            })

    df = pd.DataFrame(rows)
    out_dir = RAW_DIR / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "weather_france.csv", index=False)
    print(f"  Weather: {len(df):,} rows -> {out_dir / 'weather_france.csv'}")
    return df


def generate_dpe():
    """Generate realistic DPE data for 96 departments."""
    print("Generating DPE data...")
    rows = []
    dpe_id = 1000000

    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")

    etiquettes = ["A", "B", "C", "D", "E", "F", "G"]
    etiquette_weights = [0.03, 0.07, 0.15, 0.30, 0.25, 0.12, 0.08]

    types_chauffage = [
        "PAC air/eau", "PAC air/air", "Chaudiere gaz condensation",
        "Chaudiere gaz", "Chaudiere fioul", "Radiateur electrique",
        "Poele bois", "Chauffage collectif", "Pompe a chaleur geothermique",
    ]
    types_froid = ["", "", "", "Climatisation", "PAC reversible", ""]

    for city_name, info in FRANCE_DEPARTMENTS.items():
        dept_code = info["dept"]
        lat = info["lat"]
        # Number of DPE per day (proportional to estimated population)
        # More DPE in large departments
        daily_rate = np.random.uniform(2, 8)

        # Sample 1 out of every 3 days to speed up
        sampled_dates = dates[::3]
        for date in sampled_dates:
            n_dpe = max(1, int(np.random.poisson(daily_rate)))
            for _ in range(n_dpe):
                dpe_id += 1
                etiquette = np.random.choice(etiquettes, p=etiquette_weights)
                chauffage = np.random.choice(types_chauffage)
                froid = np.random.choice(types_froid)
                surface = max(15, np.random.normal(75, 30))

                rows.append({
                    "numero_dpe": f"DPE-{dpe_id}",
                    "date_etablissement_dpe": date.strftime("%Y-%m-%d"),
                    "code_postal_ban": f"{dept_code}000",
                    "code_departement_ban": dept_code,
                    "code_insee_ban": f"{dept_code}001",
                    "nom_commune_ban": city_name,
                    "etiquette_dpe": etiquette,
                    "etiquette_ges": np.random.choice(etiquettes, p=etiquette_weights),
                    "type_batiment": np.random.choice(["maison", "appartement"], p=[0.45, 0.55]),
                    "annee_construction": int(np.random.choice(range(1950, 2025))),
                    "surface_habitable_logement": round(surface, 1),
                    "type_installation_chauffage": chauffage,
                    "type_generateur_chauffage_principal": chauffage,
                    "type_energie_principale_chauffage": "Electricite" if "PAC" in chauffage or "electrique" in chauffage else "Gaz naturel",
                    "type_generateur_froid": froid,
                    "conso_5_usages_par_m2_ep": round(np.random.uniform(30, 500), 1),
                    "emission_ges_5_usages_par_m2": round(np.random.uniform(5, 80), 1),
                    "cout_total_5_usages": round(np.random.uniform(500, 4000), 0),
                })

            # Limit to avoid running out of memory
            if len(rows) > 500000:
                break
        if len(rows) > 500000:
            break

    df = pd.DataFrame(rows)
    out_dir = RAW_DIR / "dpe"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "dpe_france_all.csv", index=False)
    print(f"  DPE: {len(df):,} rows -> {out_dir / 'dpe_france_all.csv'}")
    return df


def generate_insee():
    """Generate INSEE economic indicators."""
    print("Generating INSEE data...")
    periods = pd.date_range("2019-01", "2025-12", freq="MS")
    rows = []

    confiance_base = 100
    for period in periods:
        confiance_base += np.random.normal(0, 1.5)
        rows.append({
            "period": period.strftime("%Y-%m"),
            "confiance_menages": round(max(70, min(130, confiance_base + np.random.normal(0, 2))), 1),
            "climat_affaires_industrie": round(max(70, min(130, 100 + np.random.normal(0, 8))), 1),
            "climat_affaires_batiment": round(max(70, min(130, 95 + np.random.normal(0, 6))), 1),
            "opinion_achats_importants": round(np.random.normal(-15, 8), 1),
            "situation_financiere_future": round(np.random.normal(-5, 5), 1),
            "ipi_industrie_manuf": round(max(70, 100 + np.random.normal(0, 5)), 1),
        })

    df = pd.DataFrame(rows)
    out_dir = RAW_DIR / "insee"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "indicateurs_economiques.csv", index=False)
    print(f"  INSEE: {len(df)} rows -> {out_dir / 'indicateurs_economiques.csv'}")
    return df


def generate_eurostat():
    """Generate Eurostat IPI data."""
    print("Generating Eurostat data...")
    periods = pd.date_range("2019-01", "2025-12", freq="MS")
    nace_codes = ["C25", "C28", "C2825"]
    rows = []

    for period in periods:
        for nace in nace_codes:
            base = {"C25": 105, "C28": 98, "C2825": 92}[nace]
            rows.append({
                "period": period.strftime("%Y-%m"),
                "nace_r2": nace,
                "geo": "FR",
                "ipi_value": round(base + np.random.normal(0, 8), 1),
                "unit": "I21",
                "s_adj": "SCA",
            })

    df = pd.DataFrame(rows)
    out_dir = RAW_DIR / "eurostat"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ipi_hvac_france.csv", index=False)
    print(f"  Eurostat: {len(df)} rows -> {out_dir / 'ipi_hvac_france.csv'}")
    return df


def generate_sitadel():
    """Generate SITADEL building permits data."""
    print("Generating SITADEL data...")
    periods = pd.date_range("2019-01", "2025-12", freq="MS")
    rows = []

    for city_name, dept_info in FRANCE_DEPARTMENTS.items():
        dept_code = dept_info["dept"]
        base_rate = np.random.uniform(50, 500)
        for period in periods:
            # Seasonality
            month_factor = 1 + 0.3 * np.sin(2 * np.pi * (period.month - 3) / 12)
            n_permis = max(1, int(np.random.poisson(base_rate * month_factor)))

            rows.append({
                "DEP": dept_code,
                "REG": dept_info.get("region", ""),
                "DATE_PRISE_EN_COMPTE": period.strftime("%Y-%m"),
                "NB_LGT_TOT_CREES": n_permis,
                "NB_LGT_IND_CREES": int(n_permis * 0.4),
                "NB_LGT_COLL_CREES": int(n_permis * 0.6),
                "SURF_TOT_M2": int(n_permis * np.random.uniform(60, 100)),
            })

    df = pd.DataFrame(rows)
    out_dir = RAW_DIR / "sitadel"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "permis_construire_france.csv", index=False)
    print(f"  SITADEL: {len(df):,} rows -> {out_dir / 'permis_construire_france.csv'}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  DEMO DATA GENERATION")
    print("  96 departments x 2019-2025")
    print("=" * 60)

    generate_weather()
    generate_insee()
    generate_eurostat()
    generate_sitadel()
    generate_dpe()

    print("=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print("\nTo run the full pipeline:")
    print("  python -m src.pipeline process")
    print("  python -m src.pipeline train")
    print("  streamlit run app/app.py")
