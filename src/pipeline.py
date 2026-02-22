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


def run_import_data(interactive: bool = False) -> None:
    """Import collected CSVs into the database.

    Reads CSV files from data/raw/ and loads them into the appropriate
    database tables (fact_hvac_installations, fact_economic_context).
    The database must have been initialized beforehand (init_db).

    Args:
        interactive: If True, show an interactive menu to choose
            which sources to import.
    """
    from src.database.db_manager import DatabaseManager

    logger = logging.getLogger("pipeline")

    db = DatabaseManager(config.database.connection_string)

    if interactive:
        selected = _interactive_import_menu(db, config.raw_data_dir)
        if not selected:
            logger.info("No sources selected. Import cancelled.")
            return
        results = db.import_collected_data(
            raw_data_dir=config.raw_data_dir, sources=selected,
        )
    else:
        logger.info("Importing collected data into the database...")
        results = db.import_collected_data(raw_data_dir=config.raw_data_dir)

    # Display table summary after import
    table_info = db.get_table_info()
    logger.info("Table status after import:\n%s", table_info.to_string(index=False))


def _interactive_import_menu(db, raw_data_dir) -> list:
    """Display an interactive menu for database import.

    Shows each available data source with file info (size, rows,
    last modified) and lets the user pick which ones to import.

    Args:
        db: DatabaseManager instance (for IMPORT_SOURCES metadata).
        raw_data_dir: Path to raw data directory.

    Returns:
        List of selected source names, or empty list if cancelled.
    """
    from datetime import datetime

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║           DATABASE IMPORT — Source Selection            ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    sources_info = []
    for name, meta in db.IMPORT_SOURCES.items():
        filepath = raw_data_dir / meta["file"]
        info = {
            "name": name,
            "description": meta["description"],
            "table": meta["table"],
            "exists": filepath.exists(),
            "rows": 0,
            "size_mb": 0.0,
            "modified": "N/A",
        }

        if filepath.exists():
            stat = filepath.stat()
            info["size_mb"] = stat.st_size / (1024 * 1024)
            info["modified"] = datetime.fromtimestamp(
                stat.st_mtime
            ).strftime("%Y-%m-%d %H:%M")

            # Count rows (fast line count)
            try:
                with open(filepath) as f:
                    info["rows"] = sum(1 for _ in f) - 1
            except Exception:
                pass

        sources_info.append(info)

    # Display each source
    for i, info in enumerate(sources_info, 1):
        status = "READY" if info["exists"] else "MISSING"
        status_icon = "+" if info["exists"] else "x"

        print(f"  [{i}] {status_icon} {info['name'].upper()}")
        print(f"      {info['description']}")
        print(f"      Table: {info['table']}")

        if info["exists"]:
            print(
                f"      File: {info['rows']:,} rows"
                f" | {info['size_mb']:.1f} MB"
                f" | Modified: {info['modified']}"
            )
        else:
            print(f"      File: NOT FOUND")
        print()

    # Count available sources
    available = [s for s in sources_info if s["exists"]]
    n_available = len(available)

    if n_available == 0:
        print("  No data files found. Run collection first:")
        print("  python -m src.pipeline collect")
        return []

    print(f"  ─────────────────────────────────────────────────────")
    print(f"  {n_available} source(s) available for import.")
    print()
    print(f"  Options:")
    print(f"    a     = Import ALL available sources")
    print(f"    1,3,5 = Import specific sources (comma-separated)")
    print(f"    q     = Cancel")
    print()

    try:
        choice = input("  Select > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return []

    if choice in ("q", "quit", "cancel", ""):
        print("  Cancelled.")
        return []

    if choice in ("a", "all"):
        selected = [s["name"] for s in available]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = []
            for idx in indices:
                if 0 <= idx < len(sources_info) and sources_info[idx]["exists"]:
                    selected.append(sources_info[idx]["name"])
                elif 0 <= idx < len(sources_info):
                    print(f"    Skipping {sources_info[idx]['name']} (file missing)")
        except ValueError:
            print("  Invalid input. Cancelled.")
            return []

    if not selected:
        print("  No valid sources selected.")
        return []

    # Confirmation
    print()
    print(f"  ┌───────────────────────────────────────────────────┐")
    print(f"  │  Will import {len(selected)} source(s):".ljust(54) + "│")
    for name in selected:
        info = next(s for s in sources_info if s["name"] == name)
        line = f"  │    {name.upper()} ({info['rows']:,} rows)"
        print(line.ljust(54) + "│")
    print(f"  └───────────────────────────────────────────────────┘")
    print()

    try:
        confirm = input("  Proceed? [Y/n] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "y"

    if confirm in ("n", "no"):
        print("  Import cancelled.")
        return []

    return selected


def run_clean(interactive: bool = False) -> None:
    """Clean raw data from all sources.

    Reads files from data/raw/, applies cleaning rules
    (duplicates, outliers, types, NaN) and saves to
    data/processed/.

    Can be rerun safely (idempotent).

    Args:
        interactive: If True, show a preview of what will be cleaned
            and let the user choose which rules to skip.
    """
    from src.processing.clean_data import CLEANING_RULES, DataCleaner

    logger = logging.getLogger("pipeline")

    if interactive:
        skip_rules = _interactive_cleaning_menu(config)
        cleaner = DataCleaner(config, skip_rules=skip_rules)
    else:
        cleaner = DataCleaner(config)

    logger.info("Cleaning raw data...")
    stats = cleaner.clean_all()
    logger.info("Cleaning complete: %d sources processed", len(stats))


def _interactive_cleaning_menu(cfg) -> dict:
    """Display an interactive menu for the cleaning pipeline.

    Shows a preview of each cleaning rule's impact and lets the user
    choose which rules to skip. Useful when data is expensive to
    re-collect (e.g. DPE from ADEME API).

    Args:
        cfg: Project configuration.

    Returns:
        Dictionary of rules to skip: {source: {rule1, rule2, ...}}.
    """
    from src.processing.clean_data import CLEANING_RULES, DataCleaner

    print("\n" + "=" * 65)
    print("  INTERACTIVE CLEANING MENU")
    print("  Preview the impact of each rule before cleaning.")
    print("=" * 65)

    # Run preview
    previewer = DataCleaner(cfg)
    print("\n  Analyzing data (this may take a moment for DPE)...\n")
    preview = previewer.preview_all()

    skip_rules: dict = {}

    for source, impacts in preview.items():
        rules_def = CLEANING_RULES.get(source, {})
        if not rules_def:
            continue

        # Count total raw rows
        raw_paths = {
            "weather": cfg.raw_data_dir / "weather" / "weather_france.csv",
            "insee": cfg.raw_data_dir / "insee" / "indicateurs_economiques.csv",
            "eurostat": cfg.raw_data_dir / "eurostat" / "ipi_hvac_france.csv",
            "dpe": cfg.raw_data_dir / "dpe" / "dpe_france_all.csv",
        }
        raw_path = raw_paths.get(source)
        total_str = ""
        if raw_path and raw_path.exists():
            with open(raw_path) as f:
                total = sum(1 for _ in f) - 1
            total_str = f" ({total:,} rows)"

        print(f"\n{'─' * 65}")
        print(f"  [{source.upper()}]{total_str}")
        print(f"{'─' * 65}")

        for i, impact in enumerate(impacts):
            rule = impact.get("rule", "?")
            desc = impact.get("description", rule)
            affected = impact.get("rows_affected", 0)
            pct = impact.get("pct", 0)
            note = impact.get("note", "")
            note_str = f" ({note})" if note else ""

            status = "DELETE" if not note else "MODIFY"
            color_prefix = "***" if affected > 0 and "clip" not in rule else "   "

            print(
                f"  [{i + 1}] {desc}"
                f"\n      Impact: {affected:,} rows ({pct}%)"
                f" — {status}{note_str}"
            )

        # Ask user which rules to skip
        print(f"\n  Enter rule numbers to SKIP (comma-separated), or press Enter to keep all:")
        print(f"  Example: 2,3 to skip rules 2 and 3")

        try:
            user_input = input(f"  Skip [{source}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Using all rules (non-interactive).")
            user_input = ""

        if user_input:
            try:
                skip_indices = [int(x.strip()) - 1 for x in user_input.split(",")]
                rule_names = list(rules_def.keys())
                skipped = set()
                for idx in skip_indices:
                    if 0 <= idx < len(rule_names):
                        skipped.add(rule_names[idx])
                        print(f"    → Skipping: {rule_names[idx]}")
                if skipped:
                    skip_rules[source] = skipped
            except ValueError:
                print("    → Invalid input, keeping all rules.")

    # Summary
    print(f"\n{'=' * 65}")
    print("  CLEANING CONFIGURATION SUMMARY")
    print(f"{'=' * 65}")
    if skip_rules:
        for source, rules in skip_rules.items():
            for rule in rules:
                print(f"  [SKIP] {source}.{rule}")
    else:
        print("  All rules will be applied (no skips).")
    print(f"{'=' * 65}")

    try:
        confirm = input("\n  Proceed with cleaning? [Y/n] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirm = "y"

    if confirm in ("n", "no"):
        print("  Cleaning cancelled.")
        sys.exit(0)

    return skip_rules


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
            "ML dataset built: %d rows × %d columns",
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
        "Features dataset: %d rows × %d columns",
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


def run_process(interactive: bool = False) -> None:
    """Execute the full processing pipeline (clean → merge → features → outliers).

    Chains the four processing stages in sequence.
    This is the recommended command for processing data in one go.

    Args:
        interactive: If True, show interactive cleaning menu.
    """
    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("  Full processing pipeline")
    logger.info("  clean → merge → features → outliers")
    logger.info("=" * 60)

    run_clean(interactive=interactive)
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
  python -m src.pipeline import_data -i           # Interactive: choose sources to import
  python -m src.pipeline clean                    # Clean raw data
  python -m src.pipeline merge                    # Merge into ML dataset
  python -m src.pipeline features                 # Feature engineering
  python -m src.pipeline process                  # clean + merge + features
  python -m src.pipeline clean --interactive      # Interactive: preview before cleaning
  python -m src.pipeline process -i               # Interactive processing
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
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="Interactive mode: preview cleaning impact and choose "
             "which rules to skip before deleting data.",
    )
    parser.add_argument(
        "--skip-rules",
        type=str,
        default=None,
        help="Rules to skip (format: source.rule,source.rule). "
             "Ex: dpe.filter_invalid_dates,dpe.filter_invalid_labels",
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
        run_import_data(interactive=args.interactive)

    elif args.stage == "clean":
        run_clean(interactive=args.interactive)

    elif args.stage == "merge":
        run_merge()

    elif args.stage == "features":
        run_features()

    elif args.stage == "outliers":
        run_outliers()

    elif args.stage == "process":
        run_process(interactive=args.interactive)

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
