# -*- coding: utf-8 -*-
"""
Create a lightweight sample dataset in data/sample/.
======================================================

Extracts the first 100 lines from each raw data file
to allow a user to test the project without downloading
all the data.

Usage:
    python scripts/create_sample_data.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

SAMPLE_FILES = {
    "weather": "weather/weather_france.csv",
    "dpe": "dpe/dpe_france_all.csv",
    "insee": "insee/indicateurs_economiques.csv",
    "eurostat": "eurostat/ipi_hvac_france.csv",
    "sitadel": "sitadel/permis_construire_france.csv",
}

SAMPLE_ROWS = 200


def main():
    """Entry point."""
    print("Creating sample data in data/sample/")
    print("=" * 50)

    for source, rel_path in SAMPLE_FILES.items():
        src = RAW_DIR / rel_path
        dst = SAMPLE_DIR / rel_path

        if not src.exists():
            print(f"  {source:<10} : ABSENT ({src})")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Read the first N lines
        with open(src, "r", encoding="utf-8") as f_in:
            lines = []
            for i, line in enumerate(f_in):
                lines.append(line)
                if i >= SAMPLE_ROWS:
                    break

        with open(dst, "w", encoding="utf-8") as f_out:
            f_out.writelines(lines)

        size_kb = dst.stat().st_size / 1024
        print(f"  {source:<10} : {len(lines)-1} rows ({size_kb:.1f} KB)")

    # Add a README in sample/
    readme = SAMPLE_DIR / "README.md"
    readme.write_text(
        "# Sample Data\n\n"
        "This folder contains a 200-row extract from each source.\n"
        "Use to test the pipeline without downloading the full datasets.\n\n"
        "```bash\n"
        "# Copy samples to data/raw/ for testing\n"
        "cp -r data/sample/* data/raw/\n"
        "python -m src.pipeline process\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"\n  Sample data created in {SAMPLE_DIR}")
    print("  To test: cp -r data/sample/* data/raw/")


if __name__ == "__main__":
    main()
