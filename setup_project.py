# -*- coding: utf-8 -*-
"""
HVAC Market Analysis project setup script.
============================================

All-in-one script to initialize the project on a new machine.
Checks prerequisites, creates directories, copies .env if missing,
initializes the SQLite database, and displays the project status.

Usage:
    python setup_project.py

This script is idempotent: it can be rerun safely.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).parent.resolve()


def print_header(title: str) -> None:
    """Display a formatted title."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_python_version() -> bool:
    """Check that Python 3.10+ is installed."""
    v = sys.version_info
    print(f"  Python : {v.major}.{v.minor}.{v.micro}", end="")
    if v.major >= 3 and v.minor >= 10:
        print(" [OK]")
        return True
    print(" [ERROR] Python 3.10+ required")
    return False


def check_pip_packages() -> dict:
    """Check required Python packages."""
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
            status[pkg] = "MISSING"

    for pkg in optional:
        try:
            __import__(pkg.replace("-", "_"))
            status[pkg] = "OK"
        except ImportError:
            status[pkg] = "optional"

    return status


def create_directories() -> None:
    """Create data directories if they do not exist."""
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
    """Copy .env.example to .env if .env does not exist."""
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        print(f"  .env already exists [OK]")
    elif env_example.exists():
        shutil.copy2(env_example, env_file)
        print(f"  .env created from .env.example [OK]")
    else:
        print(f"  .env.example not found [WARNING]")


def init_database() -> bool:
    """Initialize the SQLite database via the pipeline."""
    db_path = PROJECT_ROOT / "data" / "hvac_market.db"
    print(f"  DB path: {db_path}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.pipeline", "init_db"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            print(f"  DB initialized [OK]")
            return True
        else:
            print(f"  DB init error: {result.stderr[:200]}")
            return False
    except Exception as exc:
        print(f"  Error: {exc}")
        return False


def check_collected_data() -> dict:
    """Check which data has already been collected."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    files = {
        "weather": raw_dir / "weather" / "weather_france.csv",
        "insee": raw_dir / "insee" / "indicateurs_economiques.csv",
        "eurostat": raw_dir / "eurostat" / "ipi_hvac_france.csv",
        "sitadel": raw_dir / "sitadel" / "permis_construire_france.csv",
        "dpe": raw_dir / "dpe" / "dpe_france_all.csv",
    }
    status = {}
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            status[name] = f"OK ({size_mb:.1f} MB)"
        else:
            status[name] = "Not collected"
    return status


def main() -> None:
    """Main entry point for the setup."""
    print_header("HVAC Market Analysis - Project Setup")

    # 1. Check Python
    print_header("1. Python Check")
    if not check_python_version():
        sys.exit(1)

    # 2. Check packages
    print_header("2. Package Check")
    pkg_status = check_pip_packages()
    missing = []
    for pkg, status in pkg_status.items():
        icon = "[OK]" if status == "OK" else f"[{status}]"
        print(f"  {pkg:<20} {icon}")
        if status == "MISSING":
            missing.append(pkg)

    if missing:
        print(f"\n  WARNING: Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install -r requirements.txt")

    # 3. Create directories
    print_header("3. Directory Creation")
    create_directories()

    # 4. Configure .env
    print_header("4. .env Configuration")
    setup_env_file()

    # 5. Initialize the database
    print_header("5. Database Initialization")
    if not missing:
        init_database()
    else:
        print("  Skipped (missing packages)")

    # 6. Check collected data
    print_header("6. Collected Data Status")
    data_status = check_collected_data()
    for source, status in data_status.items():
        print(f"  {source:<15} : {status}")

    # 7. Summary
    print_header("Setup complete!")
    print("""
  Next steps:
  1. Collect data:          python -m src.pipeline collect
  2. Import into DB:        python -m src.pipeline import_data
  3. Explore data:          jupyter notebook

  To switch to SQL Server:
  1. pip install pyodbc
  2. Set DB_TYPE=mssql in .env
  3. Rerun: python -m src.pipeline init_db
  4. Rerun: python -m src.pipeline import_data
""")


if __name__ == "__main__":
    main()
