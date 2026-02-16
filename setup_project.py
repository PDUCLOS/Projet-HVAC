# -*- coding: utf-8 -*-
"""
Script de setup du projet HVAC Market Analysis.
=================================================

Script tout-en-un pour initialiser le projet sur une nouvelle machine.
Verifie les prerequis, cree les repertoires, copie le .env si absent,
initialise la BDD SQLite et affiche l'etat du projet.

Usage:
    python setup_project.py

Ce script est idempotent : il peut etre relance sans risque.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Repertoire racine du projet (ou se trouve ce script)
PROJECT_ROOT = Path(__file__).parent.resolve()


def print_header(title: str) -> None:
    """Affiche un titre formate."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_python_version() -> bool:
    """Verifie que Python 3.10+ est installe."""
    v = sys.version_info
    print(f"  Python : {v.major}.{v.minor}.{v.micro}", end="")
    if v.major >= 3 and v.minor >= 10:
        print(" [OK]")
        return True
    print(" [ERREUR] Python 3.10+ requis")
    return False


def check_pip_packages() -> dict:
    """Verifie les packages Python requis."""
    required = [
        "pandas", "numpy", "requests", "sqlalchemy",
        "lxml", "dotenv", "tqdm",
    ]
    optional = [
        "eurostat", "matplotlib", "seaborn", "plotly",
        "scikit-learn", "lightgbm", "prophet", "pyodbc",
    ]

    status = {}
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
            status[pkg] = "OK"
        except ImportError:
            status[pkg] = "MANQUANT"

    for pkg in optional:
        try:
            __import__(pkg.replace("-", "_"))
            status[pkg] = "OK"
        except ImportError:
            status[pkg] = "optionnel"

    return status


def create_directories() -> None:
    """Cree les repertoires de donnees s'ils n'existent pas."""
    dirs = [
        PROJECT_ROOT / "data" / "raw" / "weather",
        PROJECT_ROOT / "data" / "raw" / "insee",
        PROJECT_ROOT / "data" / "raw" / "eurostat",
        PROJECT_ROOT / "data" / "raw" / "sitadel",
        PROJECT_ROOT / "data" / "raw" / "dpe",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "features",
        PROJECT_ROOT / "data" / "models",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  {d.relative_to(PROJECT_ROOT)} [OK]")


def setup_env_file() -> None:
    """Copie .env.example vers .env si .env n'existe pas."""
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print(f"  .env existe deja [OK]")
    elif env_example.exists():
        shutil.copy2(env_example, env_file)
        print(f"  .env cree depuis .env.example [OK]")
    else:
        print(f"  .env.example introuvable [ATTENTION]")


def init_database() -> bool:
    """Initialise la BDD SQLite via le pipeline."""
    db_path = PROJECT_ROOT / "data" / "hvac_market.db"
    print(f"  Chemin BDD : {db_path}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.pipeline", "init_db"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            print(f"  BDD initialisee [OK]")
            return True
        else:
            print(f"  Erreur init BDD : {result.stderr[:200]}")
            return False
    except Exception as exc:
        print(f"  Erreur : {exc}")
        return False


def check_collected_data() -> dict:
    """Verifie quelles donnees ont deja ete collectees."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    files = {
        "weather": raw_dir / "weather" / "weather_aura.csv",
        "insee": raw_dir / "insee" / "indicateurs_economiques.csv",
        "eurostat": raw_dir / "eurostat" / "ipi_hvac_france.csv",
        "sitadel": raw_dir / "sitadel" / "permis_construire_aura.csv",
        "dpe": raw_dir / "dpe" / "dpe_aura_all.csv",
    }
    status = {}
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            status[name] = f"OK ({size_mb:.1f} Mo)"
        else:
            status[name] = "Non collecte"
    return status


def main() -> None:
    """Point d'entree principal du setup."""
    print_header("HVAC Market Analysis - Setup du projet")

    # 1. Verifier Python
    print_header("1. Verification Python")
    if not check_python_version():
        sys.exit(1)

    # 2. Verifier les packages
    print_header("2. Verification des packages")
    pkg_status = check_pip_packages()
    missing = []
    for pkg, status in pkg_status.items():
        icon = "[OK]" if status == "OK" else f"[{status}]"
        print(f"  {pkg:<20} {icon}")
        if status == "MANQUANT":
            missing.append(pkg)

    if missing:
        print(f"\n  ATTENTION: Packages manquants : {', '.join(missing)}")
        print(f"  Installer avec : pip install -r requirements.txt")

    # 3. Creer les repertoires
    print_header("3. Creation des repertoires")
    create_directories()

    # 4. Configurer .env
    print_header("4. Configuration .env")
    setup_env_file()

    # 5. Initialiser la BDD
    print_header("5. Initialisation de la base de donnees")
    if not missing:
        init_database()
    else:
        print("  Passe (packages manquants)")

    # 6. Verifier les donnees collectees
    print_header("6. Etat des donnees collectees")
    data_status = check_collected_data()
    for source, status in data_status.items():
        print(f"  {source:<15} : {status}")

    # 7. Resume
    print_header("Setup termine !")
    print("""
  Prochaines etapes :
  1. Collecter les donnees :  python -m src.pipeline collect
  2. Importer dans la BDD :  python -m src.pipeline import_data
  3. Explorer les donnees :   jupyter notebook

  Pour basculer vers SQL Server :
  1. pip install pyodbc
  2. Modifier DB_TYPE=mssql dans .env
  3. Relancer : python -m src.pipeline init_db
  4. Relancer : python -m src.pipeline import_data
""")


if __name__ == "__main__":
    main()
