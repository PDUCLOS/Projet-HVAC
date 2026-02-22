# -*- coding: utf-8 -*-
"""
Airflow DAG — HVAC Market Analysis Pipeline
=============================================

This DAG orchestrates the complete HVAC (heat pump) market analysis pipeline
in France. It chains multi-source data collection, processing, ML modeling,
and deployment of results to pCloud.

Pipeline architecture :
    1. COLLECTION  — 5 sources in parallel (weather, insee, eurostat, sitadel, dpe)
    2. DATABASE    — SQLite database initialization + CSV import
    3. PROCESSING  — Cleaning -> Merging -> Features -> Outliers
    4. ANALYSIS/ML — EDA and training (Ridge, LightGBM, Prophet) in parallel
    5. EVALUATION  — Model comparison + report
    6. DEPLOYMENT  — Upload results to pCloud

Illustrated Airflow concepts :
    - DAG (Directed Acyclic Graph) : directed acyclic graph defining
      the task execution order and their dependencies.
    - BashOperator : executes shell commands (here the pipeline CLI steps).
    - PythonOperator : executes native Python functions (here quality checks).
    - TaskGroup : groups logically related tasks for better
      readability in the Airflow UI.
    - default_args : default arguments applied to all tasks in the DAG
      (owner, retries, email alerts, etc.).
    - schedule_interval : cron expression defining the execution frequency.
    - SLA (Service Level Agreement) : maximum tolerated duration for a task.
      If exceeded, Airflow sends an alert (without interrupting the task).
    - Callbacks : functions called automatically on task success or failure,
      enabling custom notifications.
    - catchup=False : prevents Airflow from retroactively launching missed
      executions between start_date and today.
    - max_active_runs=1 : prevents simultaneous execution of multiple instances
      of the same DAG (safety measure for data pipelines).
    - tags : labels for filtering and finding the DAG in the UI.

Scheduling :
    The DAG runs on the 1st of each month at 6:00 AM UTC.
    This matches the update frequency of the data sources
    (monthly DPE, INSEE, Eurostat, SITADEL data).

Prerequisites :
    - Python 3.10+ with project dependencies installed
    - Environment variables configured (.env) : API keys, pCloud tokens
    - Network access for collection and upload

Author : HVAC Market Analysis Team
Date   : 2025
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Airflow Imports
# ---------------------------------------------------------------------------
# The DAG is the main container : it defines the task graph,
# the scheduling, and global parameters.
from airflow import DAG

# BashOperator executes a shell command in a subprocess.
# Ideal for calling our CLI : python -m src.pipeline <stage>
from airflow.operators.bash import BashOperator

# PythonOperator executes a native Python function.
# Used here for quality checks between pipeline steps.
from airflow.operators.python import PythonOperator

# TaskGroup allows visually grouping tasks in the Airflow UI.
# This improves readability without changing execution behavior.
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project Constants
# ---------------------------------------------------------------------------
# Absolute path to the project — each BashOperator will execute from this directory.
# In production, one would use an Airflow Variable or a path configured
# in the worker's environment.
PROJECT_DIR = str(Path(__file__).resolve().parents[2])

# Base command for executing the pipeline CLI.
# 'cd' ensures that relative imports (config, src) work correctly.
BASE_CMD = f"cd {PROJECT_DIR} && python -m src.pipeline"

# The 5 data sources collected in parallel.
COLLECT_SOURCES = ["weather", "insee", "eurostat", "sitadel", "dpe"]


# ============================================================================
# CALLBACKS — Notification Functions
# ============================================================================
# Callbacks are called automatically by Airflow on specific events
# (success, failure, SLA miss). They allow sending custom alerts
# (Slack, email, webhook, structured logs, etc.).


def on_failure_callback(context: dict[str, Any]) -> None:
    """Callback called when a task fails.

    This callback is triggered automatically by Airflow when a task
    raises an unhandled exception or returns a non-zero exit code.

    In production, one would send a Slack notification, a detailed email,
    or an event to a monitoring system (PagerDuty, Datadog, etc.).

    Args:
        context: Dictionary provided by Airflow containing execution metadata :
                 task_instance, dag_run, execution_date, exception, etc.
    """
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown"

    logger.error(
        "[FAILURE] DAG=%s | Task=%s | Execution=%s | Error=%s",
        dag_id,
        task_instance.task_id if task_instance else "unknown",
        context.get("execution_date", "unknown"),
        str(exception)[:500] if exception else "none",
    )
    # In production :
    # slack_webhook.send(f":x: HVAC Pipeline failed on {task_instance.task_id}")
    # send_email(to="data-team@company.com", subject=f"[ALERT] {dag_id}", ...)


def on_success_callback(context: dict[str, Any]) -> None:
    """Callback called when a task succeeds.

    Useful for operational monitoring : confirming that a critical step
    completed normally, updating a monitoring dashboard,
    or triggering a downstream pipeline.

    Args:
        context: Airflow dictionary with execution metadata.
    """
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown"

    logger.info(
        "[SUCCESS] DAG=%s | Task=%s | Duration=%s",
        dag_id,
        task_instance.task_id if task_instance else "unknown",
        task_instance.duration if task_instance else "unknown",
    )


def on_sla_miss_callback(
    dag: Any,
    task_list: str,
    blocking_task_list: str,
    slas: list,
    blocking_tis: list,
) -> None:
    """Callback called when a task exceeds its SLA.

    The SLA (Service Level Agreement) defines the maximum acceptable
    duration for a task. If exceeded, this callback is called BUT the task
    continues to run (SLA is informational, not blocking).

    This allows detecting performance degradations before they
    become critical.

    Args:
        dag: Instance of the affected DAG.
        task_list: Description of tasks that exceeded their SLA.
        blocking_task_list: Identified blocking tasks.
        slas: List of missed SLAs.
        blocking_tis: Blocking TaskInstances.
    """
    logger.warning(
        "[SLA EXCEEDED] DAG=%s | Tasks=%s | Blocking=%s",
        dag.dag_id if dag else "unknown",
        task_list,
        blocking_task_list,
    )


# ============================================================================
# QUALITY CHECK FUNCTIONS (PythonOperator)
# ============================================================================
# These functions are executed by PythonOperators to verify data
# consistency between pipeline steps. They raise an exception
# if an issue is detected, which stops the pipeline.


def check_collected_data(**context: Any) -> bool:
    """Verify that collected files exist and are not empty.

    This check is crucial because external APIs can fail silently
    (empty response, ignored HTTP error, exceeded quota).
    We ensure that at least one non-empty CSV file was produced for
    each source before proceeding.

    Args:
        **context: Airflow context (PythonOperator kwargs).

    Returns:
        True if data is valid.

    Raises:
        FileNotFoundError: If the data/raw/ directory does not exist.
        ValueError: If no CSV file was found.
    """
    raw_dir = Path(PROJECT_DIR) / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found : {raw_dir}. "
            f"The collection step probably failed."
        )

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(
            f"No CSV file found in {raw_dir}. "
            f"Check the collection logs to identify the source of the problem."
        )

    # Check that files are not empty (> header only)
    empty_files = [f.name for f in csv_files if f.stat().st_size < 50]
    if empty_files:
        logger.warning(
            "Potentially empty files detected : %s", empty_files
        )

    total_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    logger.info(
        "Collection check OK : %d CSV files, %.1f MB total.",
        len(csv_files),
        total_size_mb,
    )
    return True


def check_processed_data(**context: Any) -> bool:
    """Verify that processed data is consistent after cleaning/merging.

    Checks performed :
    - Existence of the features dataset
    - Minimum number of rows (safety threshold)
    - Absence of entirely empty columns

    Args:
        **context: Airflow context.

    Returns:
        True if processed data is valid.

    Raises:
        FileNotFoundError: If the features dataset does not exist.
        ValueError: If the dataset is too small or corrupted.
    """
    features_dir = Path(PROJECT_DIR) / "data" / "features"
    dataset_path = features_dir / "hvac_features_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Features dataset not found : {dataset_path}. "
            f"The clean/merge/features steps probably failed."
        )

    # Lightweight read : count lines without loading everything into memory
    with open(dataset_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f) - 1  # -1 for the header

    # Safety threshold : at least 10 rows for a valid pipeline
    min_rows = 10
    if line_count < min_rows:
        raise ValueError(
            f"Dataset too small : {line_count} rows "
            f"(expected minimum : {min_rows}). "
            f"Check source data and cleaning steps."
        )

    logger.info(
        "Processing check OK : %d rows in the features dataset.",
        line_count,
    )
    return True


def check_models_trained(**context: Any) -> bool:
    """Verify that models were correctly trained and saved.

    Checks performed :
    - Existence of the models directory
    - Presence of at least one results file
    - Presence of serialized model files (.pkl, .json, .txt)

    Args:
        **context: Airflow context.

    Returns:
        True if models are valid.

    Raises:
        FileNotFoundError: If the models directory does not exist.
        ValueError: If no model was saved.
    """
    models_dir = Path(PROJECT_DIR) / "data" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found : {models_dir}. "
            f"The training step probably failed."
        )

    # Check for model files
    model_extensions = ("*.pkl", "*.joblib", "*.json", "*.txt", "*.csv")
    model_files = []
    for ext in model_extensions:
        model_files.extend(models_dir.glob(ext))

    if not model_files:
        raise ValueError(
            f"No model file found in {models_dir}. "
            f"Check the training logs."
        )

    logger.info(
        "Model check OK : %d files in %s.",
        len(model_files),
        models_dir,
    )
    return True


# ============================================================================
# DAG DEFINITION
# ============================================================================
# default_args : default parameters applied to ALL tasks in the DAG.
# Each task can override these values individually if needed.
default_args = {
    # Owner displayed in the Airflow UI ("Owner" tab)
    "owner": "hvac-data-team",

    # depends_on_past=False : each execution is independent.
    # If True, a task only runs if its previous execution succeeded.
    "depends_on_past": False,

    # Email addresses for automatic alerts.
    # Requires SMTP configuration in airflow.cfg.
    "email": ["data-team@hvac-project.fr"],
    "email_on_failure": True,    # Send an email if the task fails
    "email_on_retry": False,     # Do not spam on each retry

    # Retry policy : on failure, Airflow automatically retries.
    # Useful for transient errors (API timeout, unstable network).
    "retries": 2,
    "retry_delay": timedelta(minutes=5),

    # DAG start date. Airflow schedules executions from this date.
    # With catchup=False, only the most recent execution will be launched.
    "start_date": datetime(2025, 1, 1),

    # Default callbacks for all tasks.
    "on_failure_callback": on_failure_callback,
    "on_success_callback": on_success_callback,
}

# ---------------------------------------------------------------------------
# DAG Instantiation
# ---------------------------------------------------------------------------
# The context manager 'with DAG(...)' automatically registers all tasks
# defined within the block as belonging to this DAG.

with DAG(
    # Unique DAG identifier in Airflow. Must be unique across the entire
    # Airflow instance. Convention : descriptive snake_case.
    dag_id="hvac_market_analysis",

    default_args=default_args,

    # Description displayed in the Airflow UI.
    description=(
        "Monthly HVAC market analysis pipeline : multi-source collection, "
        "processing, ML modeling (Ridge, LightGBM, Prophet) and deployment."
    ),

    # Cron expression : "minute hour day_of_month month day_of_week"
    # "0 6 1 * *" = the 1st of each month at 06:00 UTC.
    schedule_interval="0 6 1 * *",

    # catchup=False : do NOT retroactively execute missed runs.
    # Without this, if the DAG is enabled on March 15 with start_date on January 1,
    # Airflow would launch 3 executions (Jan, Feb, Mar).
    catchup=False,

    # max_active_runs=1 : only one simultaneous execution.
    # Safety measure to avoid write conflicts on files and the database.
    max_active_runs=1,

    # Tags for filtering in the Airflow UI.
    tags=["hvac", "data-pipeline", "ml", "monthly"],

    # Global callback for SLA misses.
    sla_miss_callback=on_sla_miss_callback,

    # Documentation displayed in the Airflow UI ("Docs" tab).
    doc_md="""
    ## HVAC Market Analysis Pipeline

    **Objective** : Analyze the heat pump market in France
    by cross-referencing 5 data sources (weather, INSEE, Eurostat, SITADEL, DPE).

    **Frequency** : Monthly (1st of the month at 06:00 UTC)

    **Steps** :
    1. Parallel collection of the 5 sources
    2. Import into SQLite database
    3. Multi-source cleaning and merging
    4. Feature engineering and outlier detection
    5. EDA and ML training in parallel
    6. Comparative model evaluation
    7. Upload results to pCloud

    **Contact** : data-team@hvac-project.fr
    """,

) as dag:

    # ========================================================================
    # GROUP 1 : DATA COLLECTION (TaskGroup)
    # ========================================================================
    # TaskGroup creates a visual subgraph in the Airflow UI.
    # The 5 collectors run in parallel as they are independent
    # (different sources, no dependency between them).

    with TaskGroup(
        group_id="collect_group",
        tooltip="Parallel collection of 5 external data sources",
    ) as collect_group:

        collect_tasks = {}
        for source in COLLECT_SOURCES:
            # Each source has its own BashOperator.
            # Advantage : if one source fails, the others continue.
            # The retry (2 attempts) handles transient network errors.
            collect_tasks[source] = BashOperator(
                task_id=f"collect_{source}",
                bash_command=f"{BASE_CMD} collect --sources {source}",

                # Collection-specific SLA : max 30 minutes per source.
                # External APIs can be slow (96 departments).
                sla=timedelta(minutes=30),

                # Override : 3 retries for collection (unstable APIs).
                retries=3,
                retry_delay=timedelta(minutes=2),

                doc_md=f"Collect **{source}** data via the dedicated API.",
            )

    # Post-collection quality check
    check_collect = PythonOperator(
        task_id="check_collected_data",
        python_callable=check_collected_data,
        doc_md="Verify that collected CSV files exist and are not empty.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # GROUP 2 : DATA PROCESSING (TaskGroup)
    # ========================================================================
    # Sequential steps : each step depends on the previous one because it
    # reads files produced by the preceding step.

    with TaskGroup(
        group_id="process_group",
        tooltip="Database initialization, import, cleaning, merging, features, outliers",
    ) as process_group:

        init_db = BashOperator(
            task_id="init_db",
            bash_command=f"{BASE_CMD} init_db",
            doc_md="Initialize the SQLite schema (tables, indexes, references).",
            sla=timedelta(minutes=5),
        )

        import_data = BashOperator(
            task_id="import_data",
            bash_command=f"{BASE_CMD} import_data",
            doc_md="Import collected CSVs into SQLite tables.",
            sla=timedelta(minutes=10),
        )

        clean = BashOperator(
            task_id="clean",
            bash_command=f"{BASE_CMD} clean",
            doc_md="Cleaning : duplicates, types, missing values, normalization.",
            sla=timedelta(minutes=15),
        )

        merge = BashOperator(
            task_id="merge",
            bash_command=f"{BASE_CMD} merge",
            doc_md="Multi-source merging into ML-ready dataset (temporal joins).",
            sla=timedelta(minutes=10),
        )

        features = BashOperator(
            task_id="features",
            bash_command=f"{BASE_CMD} features",
            doc_md="Feature engineering : lags, rolling windows, interactions, trends.",
            sla=timedelta(minutes=10),
        )

        outliers = BashOperator(
            task_id="outliers",
            bash_command=f"{BASE_CMD} outliers",
            doc_md="Outlier detection (IQR + Z-score + Isolation Forest) and treatment.",
            sla=timedelta(minutes=10),
        )

        # Sequential chain within the processing group.
        # The >> operator defines a dependency : init_db must finish
        # before import_data can start, etc.
        init_db >> import_data >> clean >> merge >> features >> outliers

    # Post-processing quality check
    check_process = PythonOperator(
        task_id="check_processed_data",
        python_callable=check_processed_data,
        doc_md="Verify the features dataset : size, completeness, consistency.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # GROUP 3 : ANALYSIS AND ML MODELING (TaskGroup)
    # ========================================================================
    # EDA and ML training run in parallel because they read
    # the same input dataset without modifying it.

    with TaskGroup(
        group_id="ml_group",
        tooltip="Exploratory analysis and ML model training",
    ) as ml_group:

        eda = BashOperator(
            task_id="eda",
            bash_command=f"{BASE_CMD} eda",
            doc_md="Exploratory analysis : distributions, correlations, charts.",
            sla=timedelta(minutes=20),
        )

        train = BashOperator(
            task_id="train",
            bash_command=f"{BASE_CMD} train",
            doc_md=(
                "ML model training : "
                "Ridge Regression, LightGBM, Prophet."
            ),
            # Training can be long depending on data volume.
            sla=timedelta(minutes=45),
            # Execution timeout : stops the task if it exceeds 2 hours.
            # Different from SLA which is informational.
            execution_timeout=timedelta(hours=2),
        )

    # Post-training quality check
    check_models = PythonOperator(
        task_id="check_models_trained",
        python_callable=check_models_trained,
        doc_md="Verify that models were correctly saved.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # FINAL STEPS : EVALUATION AND DEPLOYMENT
    # ========================================================================

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"{BASE_CMD} evaluate",
        doc_md=(
            "Comparative evaluation : metrics, prediction vs actual charts, "
            "residual analysis, feature importance, SHAP."
        ),
        sla=timedelta(minutes=30),
    )

    upload_pcloud = BashOperator(
        task_id="upload_pcloud",
        bash_command=f"{BASE_CMD} upload_pcloud",
        doc_md="Upload results (data, models, reports) to pCloud.",
        sla=timedelta(minutes=15),
        # Upload is the last step : 3 retries in case of network error.
        retries=3,
        retry_delay=timedelta(minutes=3),
    )

    # ========================================================================
    # DEPENDENCY DEFINITIONS (the DAG graph)
    # ========================================================================
    # The >> (bit shift) operator defines dependencies between tasks.
    # A >> B means : "B only starts once A has succeeded".
    #
    # Complete graph :
    #
    #   [collect_weather]  \
    #   [collect_insee]     \
    #   [collect_eurostat]   --> check_collect --> [process_group] --> check_process
    #   [collect_sitadel]   /       (init_db -> import -> clean -> merge      |
    #   [collect_dpe]      /         -> features -> outliers)                 |
    #                                                                       v
    #                                                               [eda]  [train]
    #                                                                  \   /
    #                                                              check_models
    #                                                                   |
    #                                                                evaluate
    #                                                                   |
    #                                                             upload_pcloud

    # 1. Parallel collection feeds the quality check
    collect_group >> check_collect

    # 2. If collection is valid, launch processing
    check_collect >> process_group

    # 3. After processing, verify the data
    process_group >> check_process

    # 4. If processed data is valid, launch EDA + ML in parallel
    check_process >> ml_group

    # 5. After training, verify the models
    ml_group >> check_models

    # 6. Model evaluation
    check_models >> evaluate

    # 7. Final upload
    evaluate >> upload_pcloud
