# -*- coding: utf-8 -*-
"""
Pipeline Orchestrator — Point d'entrée pour exécuter le projet.
================================================================

Orchestre les différentes étapes du projet HVAC Market Analysis :
    1. collect     — Collecte des données depuis les sources externes
    2. init_db     — Initialisation de la base de données
    3. import_data — Import des CSV collectés dans la BDD
    4. clean       — Nettoyage et normalisation des données brutes
    5. merge       — Fusion multi-sources → dataset ML-ready
    6. features    — Feature engineering (lags, rolling, interactions)
    7. process     — Exécute clean + merge + features en séquence
    8. train       — Entraînement des modèles (TODO Phase 4)
    9. evaluate    — Évaluation et comparaison (TODO Phase 4)

Usage CLI :
    # Exécuter une étape spécifique
    python -m src.pipeline collect
    python -m src.pipeline collect --sources weather,insee
    python -m src.pipeline init_db
    python -m src.pipeline import_data
    python -m src.pipeline clean
    python -m src.pipeline merge
    python -m src.pipeline features
    python -m src.pipeline process          # clean + merge + features

    # Exécuter toutes les étapes
    python -m src.pipeline all

    # Lister les collecteurs disponibles
    python -m src.pipeline list

Extensibilité :
    Pour ajouter une nouvelle étape au pipeline :
    1. Créer une fonction dans le module approprié
    2. L'enregistrer dans le dictionnaire STAGES ci-dessous
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from config.settings import config
from src.collectors.base import CollectorConfig, CollectorRegistry


def setup_logging(level: str = "INFO") -> None:
    """Configure le logging global du pipeline.

    Args:
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_collect(sources: Optional[List[str]] = None) -> None:
    """Exécute la collecte de données.

    Args:
        sources: Liste des sources à collecter. Si None, collecte tout.
                 Noms valides : weather, insee, eurostat, sitadel, dpe.
    """
    logger = logging.getLogger("pipeline")

    # Importer tous les collecteurs pour déclencher l'auto-enregistrement
    import src.collectors  # noqa: F401

    # Construire la configuration des collecteurs depuis l'environnement
    collector_config = CollectorConfig.from_env()

    if sources:
        # Collecter uniquement les sources demandées
        logger.info("Collecte ciblée : %s", sources)
        for source in sources:
            try:
                result = CollectorRegistry.run(source, collector_config)
                logger.info("%s", result)
            except KeyError:
                logger.error(
                    "Source inconnue : '%s'. Disponibles : %s",
                    source, CollectorRegistry.available(),
                )
    else:
        # Collecter toutes les sources dans l'ordre recommandé
        order = ["weather", "insee", "eurostat", "sitadel", "dpe"]
        logger.info("Collecte complète dans l'ordre : %s", order)
        results = CollectorRegistry.run_all(collector_config, order=order)
        for result in results:
            logger.info("%s", result)


def run_init_db() -> None:
    """Initialise la base de données (tables, index, données de référence)."""
    from src.database.db_manager import DatabaseManager

    logger = logging.getLogger("pipeline")
    logger.info("Initialisation de la base de données...")

    db = DatabaseManager(config.database.connection_string)
    db.init_database()

    # Afficher le résumé des tables
    table_info = db.get_table_info()
    logger.info("Tables créées :\n%s", table_info.to_string(index=False))


def run_import_data() -> None:
    """Importe les CSV collectés dans la base de données.

    Lit les fichiers CSV depuis data/raw/ et les charge dans les tables
    appropriées de la BDD (fact_hvac_installations, fact_economic_context).
    La BDD doit avoir été initialisée au préalable (init_db).
    """
    from src.database.db_manager import DatabaseManager

    logger = logging.getLogger("pipeline")
    logger.info("Import des donnees collectees dans la BDD...")

    db = DatabaseManager(config.database.connection_string)
    results = db.import_collected_data(raw_data_dir=config.raw_data_dir)

    # Afficher le résumé des tables après import
    table_info = db.get_table_info()
    logger.info("Etat des tables apres import :\n%s", table_info.to_string(index=False))


def run_clean() -> None:
    """Nettoie les données brutes de toutes les sources.

    Lit les fichiers depuis data/raw/, applique les règles de nettoyage
    (doublons, valeurs aberrantes, types, NaN) et sauvegarde dans
    data/processed/.

    Peut être relancé sans risque (idempotent).
    """
    from src.processing.clean_data import DataCleaner

    logger = logging.getLogger("pipeline")
    logger.info("Nettoyage des données brutes...")

    cleaner = DataCleaner(config)
    stats = cleaner.clean_all()

    logger.info("Nettoyage terminé : %d sources traitées", len(stats))


def run_merge() -> None:
    """Fusionne les données nettoyées en dataset ML-ready.

    Agrège les DPE par mois × département, joint la météo et les
    indicateurs économiques, et sauvegarde le résultat dans
    data/features/hvac_ml_dataset.csv.

    Prérequis : avoir exécuté 'clean' (ou au minimum avoir des données
    dans data/raw/).
    """
    from src.processing.merge_datasets import DatasetMerger

    logger = logging.getLogger("pipeline")
    logger.info("Fusion multi-sources → dataset ML...")

    merger = DatasetMerger(config)
    df_ml = merger.build_ml_dataset()

    if df_ml.empty:
        logger.error("Dataset ML vide ! Vérifier les données sources.")
    else:
        logger.info(
            "Dataset ML construit : %d lignes × %d colonnes",
            len(df_ml), len(df_ml.columns),
        )


def run_features() -> None:
    """Applique le feature engineering sur le dataset ML.

    Ajoute les features avancées (lags, rolling windows, interactions,
    tendances) et sauvegarde dans data/features/hvac_features_dataset.csv.

    Prérequis : avoir exécuté 'merge' (dataset ML dans data/features/).
    """
    from src.processing.feature_engineering import FeatureEngineer

    logger = logging.getLogger("pipeline")
    logger.info("Feature engineering...")

    fe = FeatureEngineer(config)
    df_features = fe.engineer_from_file()

    logger.info(
        "Features dataset : %d lignes × %d colonnes",
        len(df_features), len(df_features.columns),
    )


def run_process() -> None:
    """Exécute le pipeline de traitement complet (clean → merge → features).

    Enchaîne les trois étapes de traitement en séquence.
    C'est la commande recommandée pour traiter les données en une fois.
    """
    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Pipeline de traitement complet")
    logger.info("  clean → merge → features")
    logger.info("=" * 60)

    run_clean()
    run_merge()
    run_features()

    logger.info("Pipeline de traitement terminé.")


def run_list() -> None:
    """Liste les collecteurs disponibles."""
    # Importer pour déclencher l'enregistrement
    import src.collectors  # noqa: F401

    print("\nCollecteurs disponibles :")
    print("-" * 40)
    for name in CollectorRegistry.available():
        collector_class = CollectorRegistry.get(name)
        print(f"  {name:<15} → {collector_class.__name__}")
    print()


def main() -> None:
    """Point d'entrée CLI du pipeline."""
    parser = argparse.ArgumentParser(
        description="HVAC Market Analysis — Pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python -m src.pipeline collect                  # Collecter toutes les sources
  python -m src.pipeline collect --sources weather,insee  # Sources specifiques
  python -m src.pipeline init_db                  # Initialiser la BDD
  python -m src.pipeline import_data              # Importer les CSV dans la BDD
  python -m src.pipeline clean                    # Nettoyer les donnees brutes
  python -m src.pipeline merge                    # Fusionner en dataset ML
  python -m src.pipeline features                 # Feature engineering
  python -m src.pipeline process                  # clean + merge + features
  python -m src.pipeline list                     # Lister les collecteurs
        """,
    )

    parser.add_argument(
        "stage",
        choices=[
            "collect", "init_db", "import_data",
            "clean", "merge", "features", "process",
            "list", "all",
        ],
        help="Étape du pipeline à exécuter",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Sources à collecter (séparées par des virgules). "
             "Ex: weather,insee,eurostat",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de logging (défaut: INFO)",
    )

    args = parser.parse_args()

    # Configurer le logging
    setup_logging(args.log_level)
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("  HVAC Market Analysis — Pipeline")
    logger.info("  Étape : %s", args.stage)
    logger.info("=" * 60)

    # Dispatcher vers l'étape demandée
    if args.stage == "collect":
        sources = args.sources.split(",") if args.sources else None
        run_collect(sources)

    elif args.stage == "init_db":
        run_init_db()

    elif args.stage == "import_data":
        run_import_data()

    elif args.stage == "clean":
        run_clean()

    elif args.stage == "merge":
        run_merge()

    elif args.stage == "features":
        run_features()

    elif args.stage == "process":
        run_process()

    elif args.stage == "list":
        run_list()

    elif args.stage == "all":
        # Exécuter toutes les étapes disponibles
        run_init_db()
        run_collect()
        run_import_data()
        run_process()   # clean + merge + features

    logger.info("Pipeline terminé.")


if __name__ == "__main__":
    main()
