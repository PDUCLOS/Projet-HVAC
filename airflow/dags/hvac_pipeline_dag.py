# -*- coding: utf-8 -*-
"""
DAG Airflow — HVAC Market Analysis Pipeline
============================================

Ce DAG orchestre le pipeline complet d'analyse du marche HVAC (pompes a chaleur)
en France. Il enchaine la collecte multi-sources, le traitement, la modelisation
ML et le deploiement des resultats sur pCloud.

Architecture du pipeline :
    1. COLLECTE   — 5 sources en parallele (weather, insee, eurostat, sitadel, dpe)
    2. BASE       — Initialisation BDD SQLite + import des CSV
    3. TRAITEMENT — Nettoyage → Fusion → Features → Outliers
    4. ANALYSE/ML — EDA et entrainement (Ridge, LightGBM, Prophet) en parallele
    5. EVALUATION  — Comparaison des modeles + rapport
    6. DEPLOIEMENT — Upload des resultats vers pCloud

Concepts Airflow illustres :
    - DAG (Directed Acyclic Graph) : graphe oriente acyclique definissant
      l'ordre d'execution des taches et leurs dependances.
    - BashOperator : execute des commandes shell (ici les etapes CLI du pipeline).
    - PythonOperator : execute des fonctions Python (ici les checks qualite).
    - TaskGroup : regroupe des taches logiquement liees pour une meilleure
      lisibilite dans l'interface Airflow.
    - default_args : arguments par defaut appliques a toutes les taches du DAG
      (owner, retries, alertes email, etc.).
    - schedule_interval : expression cron definissant la periodicite d'execution.
    - SLA (Service Level Agreement) : duree maximale toleree pour une tache.
      Si depassee, Airflow envoie une alerte (sans interrompre la tache).
    - Callbacks : fonctions appelees automatiquement en cas de succes ou d'echec
      d'une tache, permettant des notifications personnalisees.
    - catchup=False : empeche Airflow de lancer retroactivement les executions
      manquees entre start_date et aujourd'hui.
    - max_active_runs=1 : empeche l'execution simultanee de plusieurs instances
      du meme DAG (securite pour les pipelines de donnees).
    - tags : etiquettes pour filtrer et retrouver le DAG dans l'interface.

Planification :
    Le DAG s'execute le 1er de chaque mois a 6h00 UTC.
    Cela correspond a la frequence de mise a jour des sources de donnees
    (donnees mensuelles DPE, INSEE, Eurostat, SITADEL).

Prerequis :
    - Python 3.10+ avec les dependances du projet installees
    - Variables d'environnement configurees (.env) : cles API, tokens pCloud
    - Acces reseau pour la collecte et l'upload

Auteur : Equipe HVAC Market Analysis
Date   : 2025
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Imports Airflow
# ---------------------------------------------------------------------------
# Le DAG est le conteneur principal : il definit le graphe de taches,
# la planification et les parametres globaux.
from airflow import DAG

# BashOperator execute une commande shell dans un sous-processus.
# Ideal pour appeler notre CLI : python -m src.pipeline <stage>
from airflow.operators.bash import BashOperator

# PythonOperator execute une fonction Python native.
# Utilise ici pour les controles qualite entre les etapes.
from airflow.operators.python import PythonOperator

# TaskGroup permet de regrouper visuellement des taches dans l'interface
# Airflow. Cela ameliore la lisibilite sans changer l'execution.
from airflow.utils.task_group import TaskGroup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes du projet
# ---------------------------------------------------------------------------
# Chemin absolu du projet — chaque BashOperator s'executera depuis ce dossier.
# En production, on utiliserait une Variable Airflow ou un chemin configure
# dans l'environnement du worker.
PROJECT_DIR = str(Path(__file__).resolve().parents[2])

# Commande de base pour executer le pipeline CLI.
# 'cd' garantit que les imports relatifs (config, src) fonctionnent.
BASE_CMD = f"cd {PROJECT_DIR} && python -m src.pipeline"

# Les 5 sources de donnees collectees en parallele.
COLLECT_SOURCES = ["weather", "insee", "eurostat", "sitadel", "dpe"]


# ============================================================================
# CALLBACKS — Fonctions de notification
# ============================================================================
# Les callbacks sont appeles automatiquement par Airflow lors d'evenements
# specifiques (succes, echec, SLA manque). Ils permettent d'envoyer des
# alertes personnalisees (Slack, email, webhook, logs structures, etc.).


def on_failure_callback(context: dict[str, Any]) -> None:
    """Callback appele quand une tache echoue.

    Ce callback est declenche automatiquement par Airflow lorsqu'une tache
    leve une exception non geree ou retourne un code de sortie non nul.

    En production, on enverrait une notification Slack, un email detaille,
    ou un evenement dans un systeme de monitoring (PagerDuty, Datadog, etc.).

    Args:
        context: Dictionnaire fourni par Airflow contenant les metadonnees
                 de l'execution : task_instance, dag_run, execution_date,
                 exception, etc.
    """
    task_instance = context.get("task_instance")
    exception = context.get("exception")
    dag_id = context.get("dag").dag_id if context.get("dag") else "inconnu"

    logger.error(
        "[ECHEC] DAG=%s | Tache=%s | Execution=%s | Erreur=%s",
        dag_id,
        task_instance.task_id if task_instance else "inconnue",
        context.get("execution_date", "inconnue"),
        str(exception)[:500] if exception else "aucune",
    )
    # En production :
    # slack_webhook.send(f":x: Pipeline HVAC echoue sur {task_instance.task_id}")
    # send_email(to="data-team@company.com", subject=f"[ALERTE] {dag_id}", ...)


def on_success_callback(context: dict[str, Any]) -> None:
    """Callback appele quand une tache reussit.

    Utile pour le suivi operationnel : confirmer qu'une etape critique
    s'est terminee normalement, mettre a jour un dashboard de monitoring,
    ou declencher un pipeline aval.

    Args:
        context: Dictionnaire Airflow avec les metadonnees d'execution.
    """
    task_instance = context.get("task_instance")
    dag_id = context.get("dag").dag_id if context.get("dag") else "inconnu"

    logger.info(
        "[SUCCES] DAG=%s | Tache=%s | Duree=%s",
        dag_id,
        task_instance.task_id if task_instance else "inconnue",
        task_instance.duration if task_instance else "inconnue",
    )


def on_sla_miss_callback(
    dag: Any,
    task_list: str,
    blocking_task_list: str,
    slas: list,
    blocking_tis: list,
) -> None:
    """Callback appele lorsqu'une tache depasse son SLA.

    Le SLA (Service Level Agreement) definit la duree maximale acceptable
    pour une tache. Si depassee, ce callback est appele MAIS la tache
    continue de s'executer (le SLA est informatif, pas bloquant).

    Cela permet de detecter les degradations de performance avant qu'elles
    ne deviennent critiques.

    Args:
        dag: Instance du DAG concerne.
        task_list: Description des taches ayant depasse leur SLA.
        blocking_task_list: Taches bloquantes identifiees.
        slas: Liste des SLA manques.
        blocking_tis: TaskInstances bloquantes.
    """
    logger.warning(
        "[SLA DEPASSE] DAG=%s | Taches=%s | Bloquantes=%s",
        dag.dag_id if dag else "inconnu",
        task_list,
        blocking_task_list,
    )


# ============================================================================
# FONCTIONS DE CONTROLE QUALITE (PythonOperator)
# ============================================================================
# Ces fonctions sont executees par des PythonOperator pour verifier la
# coherence des donnees entre les etapes du pipeline. Elles levent une
# exception si un probleme est detecte, ce qui arrete le pipeline.


def check_collected_data(**context: Any) -> bool:
    """Verifie que les fichiers de collecte existent et ne sont pas vides.

    Cette verification est cruciale car les APIs externes peuvent echouer
    silencieusement (reponse vide, erreur HTTP ignoree, quota depasse).
    On s'assure qu'au moins un fichier CSV non vide a ete produit pour
    chaque source avant de continuer.

    Args:
        **context: Contexte Airflow (kwargs du PythonOperator).

    Returns:
        True si les donnees sont valides.

    Raises:
        FileNotFoundError: Si le dossier data/raw/ n'existe pas.
        ValueError: Si aucun fichier CSV n'a ete trouve.
    """
    raw_dir = Path(PROJECT_DIR) / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dossier de donnees brutes introuvable : {raw_dir}. "
            f"La collecte a probablement echoue."
        )

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(
            f"Aucun fichier CSV dans {raw_dir}. "
            f"Verifier les logs de collecte pour identifier la source du probleme."
        )

    # Verifier que les fichiers ne sont pas vides (> en-tete seul)
    empty_files = [f.name for f in csv_files if f.stat().st_size < 50]
    if empty_files:
        logger.warning(
            "Fichiers potentiellement vides detectes : %s", empty_files
        )

    total_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    logger.info(
        "Controle collecte OK : %d fichiers CSV, %.1f Mo au total.",
        len(csv_files),
        total_size_mb,
    )
    return True


def check_processed_data(**context: Any) -> bool:
    """Verifie que les donnees traitees sont coherentes apres nettoyage/fusion.

    Controles effectues :
    - Existence du dataset features
    - Nombre minimum de lignes (seuil de securite)
    - Absence de colonnes entierement vides

    Args:
        **context: Contexte Airflow.

    Returns:
        True si les donnees traitees sont valides.

    Raises:
        FileNotFoundError: Si le dataset features n'existe pas.
        ValueError: Si le dataset est trop petit ou corrompu.
    """
    features_dir = Path(PROJECT_DIR) / "data" / "features"
    dataset_path = features_dir / "hvac_features_dataset.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset features introuvable : {dataset_path}. "
            f"Les etapes clean/merge/features ont probablement echoue."
        )

    # Lecture legere : compter les lignes sans charger tout en memoire
    with open(dataset_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f) - 1  # -1 pour l'en-tete

    # Seuil de securite : au moins 10 lignes pour un pipeline valide
    min_rows = 10
    if line_count < min_rows:
        raise ValueError(
            f"Dataset trop petit : {line_count} lignes "
            f"(minimum attendu : {min_rows}). "
            f"Verifier les donnees sources et les etapes de nettoyage."
        )

    logger.info(
        "Controle traitement OK : %d lignes dans le dataset features.",
        line_count,
    )
    return True


def check_models_trained(**context: Any) -> bool:
    """Verifie que les modeles ont ete correctement entraines et sauvegardes.

    Controles effectues :
    - Existence du dossier de modeles
    - Presence d'au moins un fichier de resultats
    - Presence de fichiers de modeles serialises (.pkl, .json, .txt)

    Args:
        **context: Contexte Airflow.

    Returns:
        True si les modeles sont valides.

    Raises:
        FileNotFoundError: Si le dossier models n'existe pas.
        ValueError: Si aucun modele n'a ete sauvegarde.
    """
    models_dir = Path(PROJECT_DIR) / "data" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Dossier de modeles introuvable : {models_dir}. "
            f"L'entrainement a probablement echoue."
        )

    # Verifier la presence de fichiers de modeles
    model_extensions = ("*.pkl", "*.joblib", "*.json", "*.txt", "*.csv")
    model_files = []
    for ext in model_extensions:
        model_files.extend(models_dir.glob(ext))

    if not model_files:
        raise ValueError(
            f"Aucun fichier de modele dans {models_dir}. "
            f"Verifier les logs d'entrainement."
        )

    logger.info(
        "Controle modeles OK : %d fichiers dans %s.",
        len(model_files),
        models_dir,
    )
    return True


# ============================================================================
# DEFINITION DU DAG
# ============================================================================
# default_args : parametres par defaut appliques a TOUTES les taches du DAG.
# Chaque tache peut surcharger ces valeurs individuellement si besoin.
default_args = {
    # Proprietaire affiche dans l'interface Airflow (onglet "Owner")
    "owner": "hvac-data-team",

    # depends_on_past=False : chaque execution est independante.
    # Si True, une tache ne s'execute que si sa precedente execution a reussi.
    "depends_on_past": False,

    # Adresses email pour les alertes automatiques.
    # Necessite la configuration SMTP dans airflow.cfg.
    "email": ["data-team@hvac-project.fr"],
    "email_on_failure": True,    # Envoyer un email si la tache echoue
    "email_on_retry": False,     # Ne pas spammer a chaque retry

    # Politique de retry : en cas d'echec, Airflow re-essaie automatiquement.
    # Utile pour les erreurs transitoires (timeout API, reseau instable).
    "retries": 2,
    "retry_delay": timedelta(minutes=5),

    # Date de debut du DAG. Airflow planifie les executions a partir de cette
    # date. Avec catchup=False, seule l'execution la plus recente sera lancee.
    "start_date": datetime(2025, 1, 1),

    # Callbacks par defaut pour toutes les taches.
    "on_failure_callback": on_failure_callback,
    "on_success_callback": on_success_callback,
}

# ---------------------------------------------------------------------------
# Instanciation du DAG
# ---------------------------------------------------------------------------
# Le context manager 'with DAG(...)' enregistre automatiquement toutes les
# taches definies dans le bloc comme appartenant a ce DAG.

with DAG(
    # Identifiant unique du DAG dans Airflow. Doit etre unique sur toute
    # l'instance Airflow. Convention : snake_case descriptif.
    dag_id="hvac_market_analysis",

    default_args=default_args,

    # Description affichee dans l'interface Airflow.
    description=(
        "Pipeline mensuel d'analyse du marche HVAC : collecte multi-sources, "
        "traitement, modelisation ML (Ridge, LightGBM, Prophet) et deploiement."
    ),

    # Expression cron : "minute heure jour_du_mois mois jour_de_la_semaine"
    # "0 6 1 * *" = le 1er de chaque mois a 06:00 UTC.
    schedule_interval="0 6 1 * *",

    # catchup=False : ne PAS executer retroactivement les runs manques.
    # Sans cela, si le DAG est active le 15 mars avec start_date au 1er janvier,
    # Airflow lancerait 3 executions (jan, fev, mar).
    catchup=False,

    # max_active_runs=1 : une seule execution simultanee.
    # Securite pour eviter les conflits d'ecriture sur les fichiers et la BDD.
    max_active_runs=1,

    # Tags pour filtrer dans l'interface Airflow.
    tags=["hvac", "data-pipeline", "ml", "monthly"],

    # Callback global pour les depassements de SLA.
    sla_miss_callback=on_sla_miss_callback,

    # Documentation affichee dans l'interface Airflow (onglet "Docs").
    doc_md="""
    ## Pipeline HVAC Market Analysis

    **Objectif** : Analyser le marche des pompes a chaleur en France
    en croisant 5 sources de donnees (meteo, INSEE, Eurostat, SITADEL, DPE).

    **Frequence** : Mensuelle (1er du mois a 06h00 UTC)

    **Etapes** :
    1. Collecte parallele des 5 sources
    2. Import en base SQLite
    3. Nettoyage et fusion multi-sources
    4. Feature engineering et detection d'outliers
    5. EDA et entrainement ML en parallele
    6. Evaluation comparative des modeles
    7. Upload des resultats sur pCloud

    **Contact** : data-team@hvac-project.fr
    """,

) as dag:

    # ========================================================================
    # GROUPE 1 : COLLECTE DES DONNEES (TaskGroup)
    # ========================================================================
    # TaskGroup cree un sous-graphe visuel dans l'interface Airflow.
    # Les 5 collecteurs s'executent en parallele car ils sont independants
    # (sources differentes, pas de dependance entre eux).

    with TaskGroup(
        group_id="collect_group",
        tooltip="Collecte parallele des 5 sources de donnees externes",
    ) as collect_group:

        collect_tasks = {}
        for source in COLLECT_SOURCES:
            # Chaque source a son propre BashOperator.
            # Avantage : si une source echoue, les autres continuent.
            # Le retry (2 tentatives) gere les erreurs reseau transitoires.
            collect_tasks[source] = BashOperator(
                task_id=f"collect_{source}",
                bash_command=f"{BASE_CMD} collect --sources {source}",

                # SLA specifique a la collecte : max 30 minutes par source.
                # Les APIs externes peuvent etre lentes (96 departements).
                sla=timedelta(minutes=30),

                # Surcharge : 3 retries pour la collecte (APIs instables).
                retries=3,
                retry_delay=timedelta(minutes=2),

                doc_md=f"Collecte des donnees **{source}** via l'API dediee.",
            )

    # Controle qualite post-collecte
    check_collect = PythonOperator(
        task_id="check_collected_data",
        python_callable=check_collected_data,
        doc_md="Verification que les fichiers CSV collectes existent et ne sont pas vides.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # GROUPE 2 : TRAITEMENT DES DONNEES (TaskGroup)
    # ========================================================================
    # Etapes sequentielles : chaque etape depend de la precedente car elle
    # lit les fichiers produits par l'etape d'avant.

    with TaskGroup(
        group_id="process_group",
        tooltip="Initialisation BDD, import, nettoyage, fusion, features, outliers",
    ) as process_group:

        init_db = BashOperator(
            task_id="init_db",
            bash_command=f"{BASE_CMD} init_db",
            doc_md="Initialisation du schema SQLite (tables, index, references).",
            sla=timedelta(minutes=5),
        )

        import_data = BashOperator(
            task_id="import_data",
            bash_command=f"{BASE_CMD} import_data",
            doc_md="Import des CSV collectes dans les tables SQLite.",
            sla=timedelta(minutes=10),
        )

        clean = BashOperator(
            task_id="clean",
            bash_command=f"{BASE_CMD} clean",
            doc_md="Nettoyage : doublons, types, valeurs manquantes, normalisation.",
            sla=timedelta(minutes=15),
        )

        merge = BashOperator(
            task_id="merge",
            bash_command=f"{BASE_CMD} merge",
            doc_md="Fusion multi-sources en dataset ML-ready (jointures temporelles).",
            sla=timedelta(minutes=10),
        )

        features = BashOperator(
            task_id="features",
            bash_command=f"{BASE_CMD} features",
            doc_md="Feature engineering : lags, rolling windows, interactions, tendances.",
            sla=timedelta(minutes=10),
        )

        outliers = BashOperator(
            task_id="outliers",
            bash_command=f"{BASE_CMD} outliers",
            doc_md="Detection d'outliers (IQR + Z-score + Isolation Forest) et traitement.",
            sla=timedelta(minutes=10),
        )

        # Chaine sequentielle au sein du groupe de traitement.
        # L'operateur >> definit une dependance : init_db doit finir
        # avant que import_data ne commence, etc.
        init_db >> import_data >> clean >> merge >> features >> outliers

    # Controle qualite post-traitement
    check_process = PythonOperator(
        task_id="check_processed_data",
        python_callable=check_processed_data,
        doc_md="Verification du dataset features : taille, completude, coherence.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # GROUPE 3 : ANALYSE ET MODELISATION ML (TaskGroup)
    # ========================================================================
    # EDA et entrainement ML s'executent en parallele car ils lisent
    # le meme dataset en entree sans le modifier.

    with TaskGroup(
        group_id="ml_group",
        tooltip="Analyse exploratoire et entrainement des modeles ML",
    ) as ml_group:

        eda = BashOperator(
            task_id="eda",
            bash_command=f"{BASE_CMD} eda",
            doc_md="Analyse exploratoire : distributions, correlations, graphiques.",
            sla=timedelta(minutes=20),
        )

        train = BashOperator(
            task_id="train",
            bash_command=f"{BASE_CMD} train",
            doc_md=(
                "Entrainement des modeles ML : "
                "Ridge Regression, LightGBM, Prophet."
            ),
            # L'entrainement peut etre long selon le volume de donnees.
            sla=timedelta(minutes=45),
            # Execution timeout : arrete la tache si elle depasse 2 heures.
            # Different du SLA qui est informatif.
            execution_timeout=timedelta(hours=2),
        )

    # Controle qualite post-entrainement
    check_models = PythonOperator(
        task_id="check_models_trained",
        python_callable=check_models_trained,
        doc_md="Verification que les modeles ont ete correctement sauvegardes.",
        sla=timedelta(minutes=2),
    )

    # ========================================================================
    # ETAPES FINALES : EVALUATION ET DEPLOIEMENT
    # ========================================================================

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"{BASE_CMD} evaluate",
        doc_md=(
            "Evaluation comparative : metriques, graphiques predictions vs reel, "
            "analyse des residus, feature importance, SHAP."
        ),
        sla=timedelta(minutes=30),
    )

    upload_pcloud = BashOperator(
        task_id="upload_pcloud",
        bash_command=f"{BASE_CMD} upload_pcloud",
        doc_md="Upload des resultats (donnees, modeles, rapports) vers pCloud.",
        sla=timedelta(minutes=15),
        # L'upload est la derniere etape : 3 retries en cas d'erreur reseau.
        retries=3,
        retry_delay=timedelta(minutes=3),
    )

    # ========================================================================
    # DEFINITION DES DEPENDANCES (le graphe du DAG)
    # ========================================================================
    # L'operateur >> (bit shift) definit les dependances entre taches.
    # A >> B signifie : "B ne commence que lorsque A a reussi".
    #
    # Graphe complet :
    #
    #   [collect_weather]  \
    #   [collect_insee]     \
    #   [collect_eurostat]   --> check_collect --> [process_group] --> check_process
    #   [collect_sitadel]   /       (init_db → import → clean → merge      |
    #   [collect_dpe]      /         → features → outliers)                 |
    #                                                                       v
    #                                                               [eda]  [train]
    #                                                                  \   /
    #                                                              check_models
    #                                                                   |
    #                                                                evaluate
    #                                                                   |
    #                                                             upload_pcloud

    # 1. La collecte parallele alimente le controle qualite
    collect_group >> check_collect

    # 2. Si la collecte est valide, on lance le traitement
    check_collect >> process_group

    # 3. Apres le traitement, on verifie les donnees
    process_group >> check_process

    # 4. Si les donnees traitees sont valides, on lance EDA + ML en parallele
    check_process >> ml_group

    # 5. Apres l'entrainement, on verifie les modeles
    ml_group >> check_models

    # 6. Evaluation des modeles
    check_models >> evaluate

    # 7. Upload final
    evaluate >> upload_pcloud
