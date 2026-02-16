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
    8. eda         — Analyse exploratoire + corrélations (Phase 3)
    9. train       — Entraînement des modèles (Phase 4)
   10. evaluate    — Évaluation et comparaison (Phase 4)

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
    python -m src.pipeline eda              # EDA + corrélations

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


def run_eda() -> None:
    """Exécute l'analyse exploratoire complète (EDA + corrélations).

    Génère les graphiques dans data/analysis/figures/ et les rapports
    textuels dans data/analysis/.

    Prérequis : avoir exécuté 'features' (dataset dans data/features/).
    """
    from src.analysis.eda import EDAAnalyzer
    from src.analysis.correlation import CorrelationAnalyzer

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 3 — Analyse Exploratoire")
    logger.info("  EDA + Corrélations")
    logger.info("=" * 60)

    # 1. EDA
    eda = EDAAnalyzer(config)
    eda_results = eda.run_full_eda()

    # 2. Corrélations
    corr = CorrelationAnalyzer(config)
    corr_results = corr.run_full_correlation()

    total_figs = len(eda_results.get("figures", [])) + len(corr_results.get("figures", []))
    logger.info("Phase 3 terminée : %d graphiques générés.", total_figs)
    logger.info("  Figures : data/analysis/figures/")
    logger.info("  Rapports : data/analysis/")


def run_train(target: str = "nb_installations_pac") -> None:
    """Entraîne tous les modèles ML sur le dataset features (Phase 4).

    Modèles entraînés :
    - Ridge Regression (baseline robuste)
    - LightGBM (gradient boosting régularisé)
    - Prophet (séries temporelles avec régresseurs)
    - LSTM (exploratoire, si PyTorch disponible)

    Les modèles et métriques sont sauvegardés dans data/models/.

    Prérequis : avoir exécuté 'features' (dataset dans data/features/).

    Args:
        target: Variable cible à prédire.
    """
    from src.models.train import ModelTrainer

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 4 — Entraînement ML")
    logger.info("  Cible : %s", target)
    logger.info("=" * 60)

    trainer = ModelTrainer(config, target=target)
    results = trainer.train_all()

    logger.info("Phase 4 (train) terminée : %d modèles entraînés.", len(results))
    logger.info("  Modèles : data/models/")
    logger.info("  Résultats : data/models/training_results.csv")


def run_evaluate(target: str = "nb_installations_pac") -> None:
    """Évalue et compare les modèles entraînés (Phase 4.3-4.4).

    Génère :
    - Tableau comparatif des métriques
    - Graphiques predictions vs réel
    - Graphique comparaison des modèles
    - Analyse des résidus
    - Feature importance
    - Analyse SHAP (si shap installé)
    - Rapport textuel complet

    Prérequis : avoir exécuté 'train'.

    Args:
        target: Variable cible utilisée à l'entraînement.
    """
    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 4 — Évaluation et comparaison")
    logger.info("=" * 60)

    # Re-entraîner pour obtenir les résultats en mémoire
    trainer = ModelTrainer(config, target=target)
    results = trainer.train_all()

    evaluator = ModelEvaluator(config)

    # Préparer les données pour les plots
    df = trainer.load_dataset()
    _, df_val, df_test = trainer.temporal_split(df)
    _, y_val = trainer.prepare_features(df_val)
    _, y_test = trainer.prepare_features(df_test)

    # Générer les visualisations
    evaluator.plot_predictions_vs_actual(
        results, y_val.values, y_test.values, target,
    )
    evaluator.plot_model_comparison(results)
    evaluator.plot_residuals(results, y_val.values, y_test.values)
    evaluator.plot_feature_importance(results)

    # Preparer imputer sur les donnees train (reutilise pour SHAP + diagnostic)
    import pandas as pd
    from pathlib import Path
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X_train_raw, y_train_raw = trainer.prepare_features(
        df[df["date_id"] <= trainer.train_end]
    )
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)

    # SHAP sur LightGBM
    if "lightgbm" in results:
        X_val_raw, _ = trainer.prepare_features(df_val)
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val_raw),
            columns=X_val_raw.columns, index=X_val_raw.index,
        )
        evaluator.shap_analysis(
            results["lightgbm"]["model"], X_val_imp, "lightgbm",
        )

    # Diagnostic surapprentissage
    X_train_imp_eval = pd.DataFrame(
        imputer.transform(X_train_raw),
        columns=X_train_raw.columns, index=X_train_raw.index,
    )
    scaler_eval = StandardScaler()
    X_train_scaled_eval = pd.DataFrame(
        scaler_eval.fit_transform(X_train_imp_eval),
        columns=X_train_imp_eval.columns, index=X_train_imp_eval.index,
    )
    df_diag = evaluator.diagnose_overfitting(
        results, X_train_scaled_eval, y_train_raw,
    )
    if not df_diag.empty:
        diag_path = Path("data/models") / "overfitting_diagnostic.csv"
        df_diag.to_csv(diag_path, index=False)
        logger.info("  Diagnostic surapprentissage → %s", diag_path)

    # Rapport
    evaluator.generate_full_report(results, target)

    logger.info("Phase 4 (evaluate) terminée.")
    logger.info("  Figures : data/models/figures/")
    logger.info("  Rapport : data/models/evaluation_report.txt")


def run_predict(
    target: str = "nb_installations_pac",
    dept: Optional[str] = None,
) -> None:
    """Genere les predictions par departement et le classement des potentiels.

    Utilise le meilleur modele (Ridge) pour predire la cible sur chaque
    departement de la periode de test, puis classe les departements par
    potentiel (total predit, tendance).

    Si --dept est specifie, affiche le detail pour ce departement.

    Args:
        target: Variable cible.
        dept: Code departement pour filtrer (ex: '69', '38').
    """
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Predictions par departement")
    logger.info("  Cible : %s", target)
    if dept:
        logger.info("  Departement filtre : %s", dept)
    logger.info("=" * 60)

    trainer = ModelTrainer(config, target=target)
    evaluator = ModelEvaluator(config)

    # Charger le dataset
    df = trainer.load_dataset()

    # Split temporel
    df_train, df_val, df_test = trainer.temporal_split(df)

    # Preparer les features
    X_train, y_train = trainer.prepare_features(df_train)
    X_test, y_test = trainer.prepare_features(df_test)

    # Imputer et scaler
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imp),
        columns=X_train_imp.columns, index=X_train_imp.index,
    )

    # Entrainer Ridge (le meilleur modele)
    from src.models.baseline import BaselineModels

    X_val, y_val = trainer.prepare_features(df_val)
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns, index=X_val.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imp),
        columns=X_val_imp.columns, index=X_val_imp.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp),
        columns=X_test_imp.columns, index=X_test_imp.index,
    )

    baseline = BaselineModels(config, target)
    ridge_result = baseline.train_ridge(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
    )
    ridge_model = ridge_result["model"]

    # Preparer le df_test avec features scalees pour la prediction
    df_test_pred = df_test.copy()
    for col in X_test_scaled.columns:
        df_test_pred.loc[X_test_scaled.index, col] = X_test_scaled[col].values

    # Predictions par departement
    feature_cols = list(X_test_scaled.columns)
    df_ranking = evaluator.predict_by_department(
        ridge_model, df_test_pred, feature_cols, target, "ridge",
    )

    # Filtrer par departement si demande
    if dept:
        df_dept = df_ranking[df_ranking["dept"] == dept]
        if df_dept.empty:
            logger.warning("Departement '%s' non trouve dans les donnees.", dept)
            logger.info("Departements disponibles : %s",
                        sorted(df_ranking["dept"].unique()))
        else:
            logger.info("\n  Detail departement %s :", dept)
            for _, row in df_dept.iterrows():
                logger.info("    Total predit  : %.0f", row["total_predit"])
                logger.info("    Moy. mensuelle: %.1f", row["moyenne_mensuelle"])
                logger.info("    Tendance      : %+.1f%%", row["tendance_pct"])
                if "rmse" in row:
                    logger.info("    RMSE          : %.2f", row["rmse"])
                if "r2" in row:
                    logger.info("    R2            : %.3f", row["r2"])

    # Sauvegarder et tracer
    evaluator.save_department_ranking(df_ranking, "ridge")
    evaluator.plot_department_ranking(df_ranking, "ridge", target)

    # Afficher le classement complet
    logger.info("\nClassement des departements (potentiel PAC) :")
    logger.info("-" * 60)
    for _, row in df_ranking.iterrows():
        trend_symbol = "+" if row["tendance_pct"] > 0 else ""
        logger.info(
            "  #%d  Dept %-4s | Total: %6.0f | Moy: %5.1f/mois | Tendance: %s%.1f%%",
            row["rang"], row["dept"], row["total_predit"],
            row["moyenne_mensuelle"], trend_symbol, row["tendance_pct"],
        )
    logger.info("-" * 60)
    logger.info("  Fichier CSV  : data/models/dept_ranking_ridge.csv")
    logger.info("  Figure       : data/models/figures/dept_ranking_ridge.png")


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
  python -m src.pipeline eda                      # EDA + correlations
  python -m src.pipeline train                    # Entrainer les modeles ML
  python -m src.pipeline train --target nb_dpe_total  # Autre cible
  python -m src.pipeline evaluate                 # Evaluer et comparer
  python -m src.pipeline predict                  # Predictions par departement
  python -m src.pipeline predict --dept 69        # Detail pour le Rhone
  python -m src.pipeline list                     # Lister les collecteurs
        """,
    )

    parser.add_argument(
        "stage",
        choices=[
            "collect", "init_db", "import_data",
            "clean", "merge", "features", "process",
            "eda", "train", "evaluate", "predict", "list", "all",
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
        "--target",
        type=str,
        default="nb_installations_pac",
        help="Variable cible pour l'entrainement ML. "
             "Ex: nb_installations_pac, nb_dpe_total",
    )
    parser.add_argument(
        "--dept",
        type=str,
        default=None,
        help="Code departement pour filtrer les predictions. "
             "Ex: 69 (Rhone), 38 (Isere). Utilise avec 'predict'.",
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

    elif args.stage == "eda":
        run_eda()

    elif args.stage == "train":
        run_train(target=args.target)

    elif args.stage == "evaluate":
        run_evaluate(target=args.target)

    elif args.stage == "predict":
        run_predict(target=args.target, dept=args.dept)

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
