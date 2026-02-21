# -*- coding: utf-8 -*-
"""
Pipeline Orchestrator — Entry point for running the project.
==============================================================

Orchestrates the different stages of the HVAC Market Analysis project:
    1. collect     — Collect data from external sources
    2. init_db     — Initialize the database
    3. import_data — Import collected CSVs into the database
    4. clean       — Clean and normalize raw data
    5. merge       — Multi-source merge → ML-ready dataset
    6. features    — Feature engineering (lags, rolling, interactions)
    7. process     — Execute clean + merge + features in sequence
    8. eda         — Exploratory analysis + correlations (Phase 3)
    9. train       — Model training (Phase 4)
   10. evaluate    — Evaluation and comparison (Phase 4)

CLI usage:
    # Run a specific stage
    python -m src.pipeline collect
    python -m src.pipeline collect --sources weather,insee
    python -m src.pipeline init_db
    python -m src.pipeline import_data
    python -m src.pipeline clean
    python -m src.pipeline merge
    python -m src.pipeline features
    python -m src.pipeline process          # clean + merge + features
    python -m src.pipeline eda              # EDA + correlations

    # Run all stages
    python -m src.pipeline all

    # List available collectors
    python -m src.pipeline list

Extensibility:
    To add a new stage to the pipeline:
    1. Create a function in the appropriate module
    2. Register it in the STAGES dictionary below
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from config.settings import config
from src.collectors.base import CollectorConfig, CollectorRegistry


def setup_logging(level: str = "INFO") -> None:
    """Configure global logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_collect(sources: Optional[List[str]] = None) -> None:
    """Execute data collection.

    Args:
        sources: List of sources to collect. If None, collect all.
                 Valid names: weather, insee, eurostat, sitadel, dpe.
    """
    logger = logging.getLogger("pipeline")

    # Import all collectors to trigger auto-registration
    import src.collectors  # noqa: F401

    # Build collector configuration from the environment
    collector_config = CollectorConfig.from_env()

    if sources:
        # Collect only the requested sources
        logger.info("Targeted collection: %s", sources)
        for source in sources:
            try:
                result = CollectorRegistry.run(source, collector_config)
                logger.info("%s", result)
            except KeyError:
                logger.error(
                    "Unknown source: '%s'. Available: %s",
                    source, CollectorRegistry.available(),
                )
    else:
        # Collect all sources in the recommended order
        order = ["weather", "insee", "eurostat", "sitadel", "dpe"]
        logger.info("Full collection in order: %s", order)
        results = CollectorRegistry.run_all(collector_config, order=order)
        for result in results:
            logger.info("%s", result)


def run_init_db() -> None:
    """Initialize the database (tables, indexes, reference data)."""
    from src.database.db_manager import DatabaseManager

    logger = logging.getLogger("pipeline")
    logger.info("Initializing the database...")

    db = DatabaseManager(config.database.connection_string)
    db.init_database()

    # Display table summary
    table_info = db.get_table_info()
    logger.info("Tables created:\n%s", table_info.to_string(index=False))


def run_import_data() -> None:
    """Import collected CSVs into the database.

    Reads CSV files from data/raw/ and loads them into the appropriate
    database tables (fact_hvac_installations, fact_economic_context).
    The database must have been initialized beforehand (init_db).
    """
    from src.database.db_manager import DatabaseManager

    logger = logging.getLogger("pipeline")
    logger.info("Importing collected data into the database...")

    db = DatabaseManager(config.database.connection_string)
    results = db.import_collected_data(raw_data_dir=config.raw_data_dir)

    # Display table summary after import
    table_info = db.get_table_info()
    logger.info("Table status after import:\n%s", table_info.to_string(index=False))


def run_clean() -> None:
    """Clean raw data from all sources.

    Reads files from data/raw/, applies cleaning rules
    (duplicates, outliers, types, NaN) and saves to
    data/processed/.

    Can be rerun safely (idempotent).
    """
    from src.processing.clean_data import DataCleaner

    logger = logging.getLogger("pipeline")
    logger.info("Cleaning raw data...")

    cleaner = DataCleaner(config)
    stats = cleaner.clean_all()

    logger.info("Cleaning complete: %d sources processed", len(stats))


def run_merge() -> None:
    """Merge cleaned data into an ML-ready dataset.

    Aggregates DPEs by month x department, joins weather and
    economic indicators, and saves the result to
    data/features/hvac_ml_dataset.csv.

    Prerequisite: having run 'clean' (or at minimum having data
    in data/raw/).
    """
    from src.processing.merge_datasets import DatasetMerger

    logger = logging.getLogger("pipeline")
    logger.info("Multi-source merge → ML dataset...")

    merger = DatasetMerger(config)
    df_ml = merger.build_ml_dataset()

    if df_ml.empty:
        logger.error("ML dataset empty! Check source data.")
    else:
        logger.info(
            "Dataset ML construit : %d lignes × %d colonnes",
            len(df_ml), len(df_ml.columns),
        )


def run_features() -> None:
    """Apply feature engineering on the ML dataset.

    Adds advanced features (lags, rolling windows, interactions,
    trends) and saves to data/features/hvac_features_dataset.csv.

    Prerequisite: having run 'merge' (ML dataset in data/features/).
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


def run_outliers(strategy: str = "clip") -> None:
    """Detect and handle outliers in the features dataset.

    Runs multi-method analysis (IQR + modified Z-score + Isolation Forest)
    and applies the chosen strategy. Generates a detailed report.

    Prerequisite: having run 'features' (dataset in data/features/).

    Args:
        strategy: Treatment strategy ("flag", "clip", "remove").
    """
    from src.processing.outlier_detection import OutlierDetector

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 2.2 — Outlier Detection")
    logger.info("  Strategy: %s", strategy)
    logger.info("=" * 60)

    detector = OutlierDetector(config)

    # Load the features dataset
    features_path = config.features_data_dir / "hvac_features_dataset.csv"
    if not features_path.exists():
        logger.error("Features dataset not found: %s", features_path)
        logger.error("Run 'python -m src.pipeline features' first.")
        return

    import pandas as pd
    df = pd.read_csv(features_path)
    logger.info("Dataset loaded: %d rows x %d columns", len(df), len(df.columns))

    # Analysis and treatment
    df_treated, report_path = detector.run_full_analysis(df, strategy=strategy)

    # Save the treated dataset
    output_path = config.features_data_dir / "hvac_features_dataset.csv"
    df_treated.to_csv(output_path, index=False)
    logger.info("Treated dataset saved → %s", output_path)
    logger.info("Outliers report → %s", report_path)


def run_process() -> None:
    """Execute the full processing pipeline (clean → merge → features → outliers).

    Chains the four processing stages in sequence.
    This is the recommended command for processing data in one go.
    """
    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Full processing pipeline")
    logger.info("  clean → merge → features → outliers")
    logger.info("=" * 60)

    run_clean()
    run_merge()
    run_features()
    run_outliers()

    logger.info("Processing pipeline complete.")


def run_eda() -> None:
    """Execute the full exploratory analysis (EDA + correlations).

    Generates charts in data/analysis/figures/ and text reports
    in data/analysis/.

    Prerequisite: having run 'features' (dataset in data/features/).
    """
    from src.analysis.eda import EDAAnalyzer
    from src.analysis.correlation import CorrelationAnalyzer

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 3 — Exploratory Analysis")
    logger.info("  EDA + Correlations")
    logger.info("=" * 60)

    # 1. EDA (Exploratory Data Analysis)
    eda = EDAAnalyzer(config)
    eda_results = eda.run_full_eda()

    # 2. Correlations
    corr = CorrelationAnalyzer(config)
    corr_results = corr.run_full_correlation()

    total_figs = len(eda_results.get("figures", [])) + len(corr_results.get("figures", []))
    logger.info("Phase 3 complete: %d charts generated.", total_figs)
    logger.info("  Figures: data/analysis/figures/")
    logger.info("  Reports: data/analysis/")


def run_train(target: str = "nb_installations_pac") -> None:
    """Train all ML models on the features dataset (Phase 4).

    Trained models:
    - Ridge Regression (robust baseline)
    - LightGBM (regularized gradient boosting)
    - Prophet (time series with regressors)
    - LSTM (exploratory, if PyTorch is available)

    Models and metrics are saved in data/models/.

    Prerequisite: having run 'features' (dataset in data/features/).

    Args:
        target: Target variable to predict.
    """
    from src.models.train import ModelTrainer

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 4 — ML Training")
    logger.info("  Target: %s", target)
    logger.info("=" * 60)

    trainer = ModelTrainer(config, target=target)
    results = trainer.train_all()

    logger.info("Phase 4 (train) complete: %d models trained.", len(results))
    logger.info("  Models: data/models/")
    logger.info("  Results: data/models/training_results.csv")


def run_evaluate(target: str = "nb_installations_pac") -> None:
    """Evaluate and compare trained models (Phase 4.3-4.4).

    Generates:
    - Comparative metrics table
    - Predictions vs actual charts
    - Model comparison chart
    - Residual analysis
    - Feature importance
    - SHAP analysis (if shap is installed)
    - Full text report

    Prerequisite: having run 'train'.

    Args:
        target: Target variable used during training.
    """
    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Phase 4 — Evaluation and Comparison")
    logger.info("=" * 60)

    # Re-train to get results in memory
    trainer = ModelTrainer(config, target=target)
    results = trainer.train_all()

    evaluator = ModelEvaluator(config)

    # Prepare data for plots
    df = trainer.load_dataset()
    _, df_val, df_test = trainer.temporal_split(df)
    _, y_val = trainer.prepare_features(df_val)
    _, y_test = trainer.prepare_features(df_test)

    # Generate visualizations
    evaluator.plot_predictions_vs_actual(
        results, y_val.values, y_test.values, target,
    )
    evaluator.plot_model_comparison(results)
    evaluator.plot_residuals(results, y_val.values, y_test.values)
    evaluator.plot_feature_importance(results)

    # SHAP on LightGBM
    if "lightgbm" in results:
        import pandas as pd
        from sklearn.impute import SimpleImputer

        X_val_raw, _ = trainer.prepare_features(df_val)
        imputer = SimpleImputer(strategy="median")
        X_train_raw, _ = trainer.prepare_features(
            df[df["date_id"] <= trainer.train_end]
        )
        imputer.fit(X_train_raw)
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val_raw),
            columns=X_val_raw.columns, index=X_val_raw.index,
        )
        evaluator.shap_analysis(
            results["lightgbm"]["model"], X_val_imp, "lightgbm",
        )

    # Report
    evaluator.generate_full_report(results, target)

    logger.info("Phase 4 (evaluate) complete.")
    logger.info("  Figures: data/models/figures/")
    logger.info("  Report: data/models/evaluation_report.txt")


def run_sync_pcloud(force: bool = False) -> None:
    """Synchronize data from pCloud and update the database.

    Checks for updates on the pCloud public link, downloads
    new/modified files, then triggers the full import pipeline
    (import → clean → merge → features → outliers).

    Prerequisite: PCLOUD_PUBLIC_CODE variable in .env (or default value).

    Args:
        force: If True, re-download everything even if nothing has changed.
    """
    from src.collectors.pcloud_sync import PCloudSync

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  pCloud Synchronization")
    logger.info("=" * 60)

    sync = PCloudSync(config)
    result = sync.sync_and_update(force=force)

    logger.info(
        "Synchronization complete: %d files downloaded.",
        result["files_downloaded"],
    )


def run_upload_pcloud(folder_id: int = 0) -> None:
    """Upload collected data to pCloud.

    Traverses files in data/raw/ and uploads them to the
    specified pCloud folder. Requires PCLOUD_ACCESS_TOKEN in .env.

    Args:
        folder_id: ID of the destination pCloud folder (0 = root).
    """
    from src.collectors.pcloud_sync import PCloudSync

    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Upload to pCloud")
    logger.info("=" * 60)

    sync = PCloudSync(config)
    result = sync.upload_collected_data(remote_folder_id=folder_id)

    logger.info(
        "Upload complete: %d files uploaded, %d failures.",
        result["files_uploaded"], result["files_failed"],
    )


def run_update_all(target: str = "nb_installations_pac") -> None:
    """Full update: collect → process → train → upload pCloud.

    Chains all pipeline stages end to end:
    1. Collect data from APIs (96 departments)
    2. Initialize the database
    3. Import collected CSVs
    4. Cleaning + Merging + Features + Outliers
    5. Exploratory analysis
    6. ML training and evaluation
    7. Upload results to pCloud

    This is the command to use for a complete update
    of the database and models.

    Args:
        target: ML target variable.
    """
    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  FULL UPDATE")
    logger.info("  collect → process → train → upload")
    logger.info("=" * 60)

    # 1. Collect from APIs
    logger.info("-" * 40)
    logger.info("  STEP 1/7 — Data Collection")
    logger.info("-" * 40)
    run_collect()

    # 2. Database initialization
    logger.info("-" * 40)
    logger.info("  STEP 2/7 — DB Initialization")
    logger.info("-" * 40)
    run_init_db()

    # 3. Import
    logger.info("-" * 40)
    logger.info("  STEP 3/7 — CSV Import")
    logger.info("-" * 40)
    run_import_data()

    # 4. Processing (clean + merge + features + outliers)
    logger.info("-" * 40)
    logger.info("  STEP 4/7 — Full Processing")
    logger.info("-" * 40)
    run_process()

    # 5. EDA
    logger.info("-" * 40)
    logger.info("  STEP 5/7 — Exploratory Analysis")
    logger.info("-" * 40)
    run_eda()

    # 6. Train + Evaluate
    logger.info("-" * 40)
    logger.info("  STEP 6/7 — ML Training and Evaluation")
    logger.info("-" * 40)
    run_train(target=target)
    run_evaluate(target=target)

    # 7. Upload to pCloud
    logger.info("-" * 40)
    logger.info("  STEP 7/7 — Upload to pCloud")
    logger.info("-" * 40)
    run_upload_pcloud()

    logger.info("=" * 60)
    logger.info("  FULL UPDATE COMPLETE")
    logger.info("=" * 60)


def run_list() -> None:
    """List available collectors."""
    # Import to trigger registration
    import src.collectors  # noqa: F401

    print("\nAvailable collectors:")
    print("-" * 40)
    for name in CollectorRegistry.available():
        collector_class = CollectorRegistry.get(name)
        print(f"  {name:<15} → {collector_class.__name__}")
    print()


def main() -> None:
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="HVAC Market Analysis — Pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline collect                  # Collect all sources
  python -m src.pipeline collect --sources weather,insee  # Specific sources
  python -m src.pipeline init_db                  # Initialize the database
  python -m src.pipeline import_data              # Import CSVs into the database
  python -m src.pipeline clean                    # Clean raw data
  python -m src.pipeline merge                    # Merge into ML dataset
  python -m src.pipeline features                 # Feature engineering
  python -m src.pipeline process                  # clean + merge + features
  python -m src.pipeline eda                      # EDA + correlations
  python -m src.pipeline train                    # Train ML models
  python -m src.pipeline train --target nb_dpe_total  # Different target
  python -m src.pipeline evaluate                 # Evaluate and compare
  python -m src.pipeline upload_pcloud            # Upload data to pCloud
  python -m src.pipeline update_all               # Collect + process + train + upload
  python -m src.pipeline list                     # List collectors
        """,
    )

    parser.add_argument(
        "stage",
        choices=[
            "collect", "init_db", "import_data",
            "clean", "merge", "features", "outliers", "process",
            "eda", "train", "evaluate",
            "sync_pcloud", "upload_pcloud", "update_all",
            "list", "all",
        ],
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Sources to collect (comma-separated). "
             "Ex: weather,insee,eurostat",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nb_installations_pac",
        help="Target variable for ML training. "
             "Ex: nb_installations_pac, nb_dpe_total",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.log_level)
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("  HVAC Market Analysis — Pipeline")
    logger.info("  Stage: %s", args.stage)
    logger.info("=" * 60)

    # Dispatch to the requested stage
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

    elif args.stage == "outliers":
        run_outliers()

    elif args.stage == "process":
        run_process()

    elif args.stage == "eda":
        run_eda()

    elif args.stage == "train":
        run_train(target=args.target)

    elif args.stage == "evaluate":
        run_evaluate(target=args.target)

    elif args.stage == "sync_pcloud":
        run_sync_pcloud()

    elif args.stage == "upload_pcloud":
        run_upload_pcloud()

    elif args.stage == "update_all":
        run_update_all(target=args.target)

    elif args.stage == "list":
        run_list()

    elif args.stage == "all":
        # Execute all pipeline stages end to end
        run_init_db()
        run_collect()
        run_import_data()
        run_process()   # clean + merge + features + outliers
        run_eda()       # EDA + correlations
        run_train(target=args.target)
        run_evaluate(target=args.target)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
