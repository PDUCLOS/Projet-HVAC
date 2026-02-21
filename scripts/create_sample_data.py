# -*- coding: utf-8 -*-
"""
Cree un dataset d'exemple leger dans data/sample/.
====================================================

Extrait les 100 premieres lignes de chaque fichier de donnees brutes
pour permettre a un utilisateur de tester le projet sans telecharger
toutes les donnees.

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
    """Point d'entree."""
    print("Creation des donnees d'exemple dans data/sample/")
    print("=" * 50)

    for source, rel_path in SAMPLE_FILES.items():
        src = RAW_DIR / rel_path
        dst = SAMPLE_DIR / rel_path

        if not src.exists():
            print(f"  {source:<10} : ABSENT ({src})")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        # Lire les N premieres lignes
        with open(src, "r", encoding="utf-8") as f_in:
            lines = []
            for i, line in enumerate(f_in):
                lines.append(line)
                if i >= SAMPLE_ROWS:
                    break

        with open(dst, "w", encoding="utf-8") as f_out:
            f_out.writelines(lines)

        size_kb = dst.stat().st_size / 1024
        print(f"  {source:<10} : {len(lines)-1} lignes ({size_kb:.1f} Ko)")

    # Ajouter un README dans sample/
    readme = SAMPLE_DIR / "README.md"
    readme.write_text(
        "# Donnees d'exemple\n\n"
        "Ce dossier contient un extrait de 200 lignes de chaque source.\n"
        "Utiliser pour tester le pipeline sans telecharger les donnees completes.\n\n"
        "```bash\n"
        "# Copier les exemples dans data/raw/ pour tester\n"
        "cp -r data/sample/* data/raw/\n"
        "python -m src.pipeline process\n"
        "```\n",
        encoding="utf-8",
    )

    print(f"\n  Donnees d'exemple creees dans {SAMPLE_DIR}")
    print("  Pour tester : cp -r data/sample/* data/raw/")


if __name__ == "__main__":
    main()
