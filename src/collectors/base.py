# -*- coding: utf-8 -*-
"""
BaseCollector & CollectorRegistry — Fondation du système de collecte.
=====================================================================

Ce module définit l'architecture extensible du système de collecte de données.
Deux composants principaux :

1. **BaseCollector** (classe abstraite) :
   Fournit le squelette commun à tous les collecteurs : gestion HTTP avec
   retry automatique, logging structuré, validation des données, et
   sauvegarde standardisée. Chaque source de données n'a qu'à implémenter
   `collect()` et `validate()`.

2. **CollectorRegistry** (système de plugins) :
   Enregistre automatiquement chaque sous-classe concrète de BaseCollector
   grâce au hook `__init_subclass__`. Permet de découvrir, lister et
   exécuter les collecteurs sans configuration manuelle.

Architecture :
    BaseCollector (ABC)
    ├── collect()     → Récupère les données brutes (à implémenter)
    ├── validate()    → Vérifie la qualité des données (à implémenter)
    ├── save()        → Persiste sur disque (CSV par défaut, overridable)
    ├── run()         → Orchestre le cycle complet collect→validate→save
    ├── fetch_json()  → GET HTTP avec retry + parsing JSON
    ├── fetch_xml()   → GET HTTP avec retry + parsing XML
    └── fetch_bytes() → GET HTTP avec retry + contenu binaire (ZIP, etc.)

Extensibilité :
    Pour ajouter une nouvelle source de données :
    1. Créer un fichier dans src/collectors/ (ex: src/collectors/my_source.py)
    2. Définir une classe héritant de BaseCollector avec source_name
    3. Implémenter collect() et validate()
    4. C'est tout ! La source est auto-enregistrée et disponible dans le registry.

    Exemple minimal :
        class MySourceCollector(BaseCollector):
            source_name = "my_source"

            def collect(self) -> pd.DataFrame:
                data = self.fetch_json("https://api.example.com/data")
                return pd.DataFrame(data)

            def validate(self, df: pd.DataFrame) -> pd.DataFrame:
                assert "id" in df.columns, "Colonne 'id' manquante"
                return df

Usage:
    >>> from src.collectors.base import CollectorRegistry, CollectorConfig
    >>> config = CollectorConfig.from_env()
    >>> # Lancer un seul collecteur
    >>> result = CollectorRegistry.run("weather", config)
    >>> print(f"{result.name}: {result.rows_collected} lignes")
    >>> # Lancer tous les collecteurs
    >>> results = CollectorRegistry.run_all(config)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

import pandas as pd
import requests
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# Énumération des statuts de collecte
# =============================================================================

class CollectorStatus(Enum):
    """États possibles d'une exécution de collecteur.

    Values:
        SUCCESS: Collecte complète, toutes les données récupérées.
        PARTIAL: Collecte partielle (certains éléments ont échoué).
        FAILED: Échec total de la collecte.
        SKIPPED: Collecteur ignoré (données déjà à jour, etc.).
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Configuration des collecteurs
# =============================================================================

@dataclass(frozen=True)
class CollectorConfig:
    """Configuration partagée par tous les collecteurs.

    Peut être construite manuellement (tests) ou depuis les variables
    d'environnement via `from_env()`.

    Attributes:
        raw_data_dir: Répertoire racine pour les données brutes.
        processed_data_dir: Répertoire racine pour les données structurées.
        start_date: Date de début de collecte (ISO YYYY-MM-DD).
        end_date: Date de fin de collecte.
        departments: Liste des codes départements cibles.
        region_code: Code INSEE de la région (84 = AURA).
        request_timeout: Timeout HTTP en secondes.
        max_retries: Nombre max de retries sur erreur transitoire.
        retry_backoff_factor: Facteur exponentiel entre retries.
        rate_limit_delay: Pause minimale entre appels API (secondes).
        log_level: Niveau de logging (DEBUG, INFO, WARNING, ERROR).
    """
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    start_date: str = "2019-01-01"
    end_date: str = "2026-02-28"
    departments: List[str] = field(default_factory=lambda: [
        "01", "07", "26", "38", "42", "69", "73", "74"
    ])
    region_code: str = "84"
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    rate_limit_delay: float = 0.5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> CollectorConfig:
        """Construit la configuration depuis les variables d'environnement.

        Charge le fichier .env via python-dotenv puis mappe les variables
        sur les attributs. Valeurs par défaut utilisées si absent.

        Returns:
            CollectorConfig initialisée depuis l'environnement.
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
            start_date=os.getenv("DATA_START_DATE", "2019-01-01"),
            end_date=os.getenv("DATA_END_DATE", "2026-02-28"),
            departments=os.getenv(
                "TARGET_DEPARTMENTS", "01,07,26,38,42,69,73,74"
            ).split(","),
            region_code=os.getenv("TARGET_REGION", "84"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("RETRY_BACKOFF", "1.0")),
            rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "0.5")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# =============================================================================
# Résultat d'une collecte
# =============================================================================

@dataclass
class CollectorResult:
    """Résultat d'une exécution de collecteur.

    Fournit un rapport structuré de ce qui s'est passé : nombre de lignes
    collectées, chemin de sauvegarde, durée, erreurs rencontrées.

    Attributes:
        name: Nom de la source collectée.
        status: Statut final (SUCCESS, PARTIAL, FAILED, SKIPPED).
        rows_collected: Nombre de lignes dans le DataFrame final.
        output_path: Chemin du fichier sauvegardé (si applicable).
        started_at: Horodatage de début de collecte.
        finished_at: Horodatage de fin de collecte.
        errors: Liste des messages d'erreur rencontrés.
    """
    name: str
    status: CollectorStatus
    rows_collected: int = 0
    output_path: Optional[Path] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Durée totale de la collecte en secondes."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def __str__(self) -> str:
        """Résumé lisible du résultat."""
        duration = f"{self.duration_seconds:.1f}s" if self.duration_seconds else "N/A"
        return (
            f"[{self.status.value.upper():>7}] {self.name:<15} "
            f"| {self.rows_collected:>6} lignes | {duration}"
        )


# =============================================================================
# Classe abstraite BaseCollector
# =============================================================================

class BaseCollector(ABC):
    """Classe abstraite de base pour tous les collecteurs de données.

    Fournit :
    - Session HTTP avec retry automatique (429, 500, 502, 503, 504)
    - Helpers `fetch_json()`, `fetch_xml()`, `fetch_bytes()`
    - Logging structuré avec horodatage
    - Cycle de vie orchestré par `run()` : collect → validate → save
    - Auto-enregistrement dans le CollectorRegistry

    Sous-classes DOIVENT implémenter :
        - `source_name` (ClassVar[str]) : identifiant unique de la source
        - `collect()` : logique de récupération des données → DataFrame
        - `validate(df)` : vérifications de qualité → DataFrame nettoyé

    Sous-classes PEUVENT surcharger :
        - `output_subdir` : sous-répertoire de sortie (défaut = source_name)
        - `output_filename` : nom du fichier (défaut = "{source_name}.csv")
        - `save(df, path)` : logique de persistance custom (Parquet, etc.)
    """

    # --- Contrat de classe (à définir par les sous-classes) ---------------

    source_name: ClassVar[str]                       # Identifiant unique
    output_subdir: ClassVar[Optional[str]] = None    # Sous-dossier output
    output_filename: ClassVar[Optional[str]] = None  # Nom du fichier output

    # --- Hook d'auto-enregistrement ---------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enregistre automatiquement chaque sous-classe concrète.

        Ce hook Python est appelé à chaque fois qu'une classe hérite de
        BaseCollector. Si la classe est concrète (pas de méthodes abstraites
        non implémentées), elle est ajoutée au CollectorRegistry.
        """
        super().__init_subclass__(**kwargs)
        # N'enregistrer que les classes concrètes (pas les ABC intermédiaires)
        if hasattr(cls, "source_name") and not getattr(cls, "__abstractmethods__", None):
            CollectorRegistry._register(cls)

    # --- Initialisation ---------------------------------------------------

    def __init__(self, config: CollectorConfig) -> None:
        """Initialise le collecteur avec sa configuration.

        Args:
            config: Configuration partagée (chemins, dates, réseau, etc.).
        """
        self.config = config
        self.logger = logging.getLogger(f"collectors.{self.source_name}")
        self._setup_logging()
        self._session: Optional[requests.Session] = None

    # --- Point d'entrée principal -----------------------------------------

    def run(self) -> CollectorResult:
        """Exécute le cycle complet de collecte.

        Orchestre les étapes :
        1. Log de démarrage avec les paramètres
        2. `collect()` — récupération des données brutes
        3. `validate()` — vérification de la qualité
        4. `save()` — persistance sur disque
        5. Log de fin avec le bilan

        En cas d'exception, le statut passe à FAILED et l'erreur est loguée.
        La collecte ne s'interrompt jamais brutalement (toujours un résultat).

        Returns:
            CollectorResult avec le bilan complet de la collecte.
        """
        result = CollectorResult(
            name=self.source_name,
            status=CollectorStatus.FAILED,
            started_at=datetime.now(),
        )

        self.logger.info(
            "="*60 + "\n"
            "  COLLECTE : %s\n"
            "  Période  : %s → %s\n"
            "  Départements : %s\n"
            + "="*60,
            self.source_name.upper(),
            self.config.start_date,
            self.config.end_date,
            ", ".join(self.config.departments),
        )

        try:
            # Étape 1 : Collecte des données brutes
            self.logger.info("Étape 1/3 — Collecte des données...")
            df = self.collect()

            if df.empty:
                self.logger.warning("⚠ La collecte a retourné un DataFrame vide.")
                result.status = CollectorStatus.PARTIAL
                result.errors.append("DataFrame vide retourné par collect().")
            else:
                # Étape 2 : Validation et nettoyage léger
                self.logger.info("Étape 2/3 — Validation des données...")
                df = self.validate(df)

                # Étape 3 : Sauvegarde sur disque
                self.logger.info("Étape 3/3 — Sauvegarde...")
                output_path = self._resolve_output_path()
                self.save(df, output_path)

                result.rows_collected = len(df)
                result.output_path = output_path
                result.status = CollectorStatus.SUCCESS

                self.logger.info(
                    "✓ Collecte réussie : %d lignes sauvegardées → %s",
                    len(df), output_path,
                )

        except Exception as exc:
            self.logger.exception(
                "✗ ÉCHEC de la collecte '%s' : %s", self.source_name, exc
            )
            result.status = CollectorStatus.FAILED
            result.errors.append(str(exc))

        result.finished_at = datetime.now()
        self.logger.info(
            "Terminé '%s' en %.1fs — statut : %s",
            self.source_name,
            result.duration_seconds or 0,
            result.status.value,
        )
        return result

    # --- Méthodes abstraites (contrat des sous-classes) -------------------

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Récupère les données depuis la source externe.

        Cette méthode contient toute la logique spécifique à la source :
        appels API, téléchargement de fichiers, parsing, etc.

        Returns:
            DataFrame pandas avec les données brutes collectées.

        Raises:
            requests.RequestException: En cas d'erreur réseau.
            ValueError: Si le format de données est inattendu.
        """
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie légèrement le DataFrame collecté.

        Vérifications attendues :
        - Présence des colonnes obligatoires
        - Types de données corrects
        - Pas de valeurs nulles critiques
        - Nombre de lignes dans les bornes attendues

        Args:
            df: DataFrame brut issu de `collect()`.

        Returns:
            DataFrame validé (éventuellement nettoyé).

        Raises:
            ValueError: Si la validation échoue de manière critique.
        """
        ...

    # --- Sauvegarde (overridable) -----------------------------------------

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Persiste le DataFrame sur disque.

        Implémentation par défaut : sauvegarde en CSV.
        Les sous-classes peuvent surcharger pour utiliser Parquet, etc.

        Args:
            df: DataFrame validé à sauvegarder.
            path: Chemin complet du fichier de sortie.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        structured_path = self._resolve_output_path(self.config.processed_data_dir)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        if structured_path.exists():
            existing = pd.read_csv(structured_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(keep="last")
            combined.to_csv(structured_path, index=False)
        else:
            df.drop_duplicates(keep="last").to_csv(structured_path, index=False)
        self.logger.debug("Sauvegardé %d lignes → %s", len(df), path)

    # --- Helpers HTTP (partagés par tous les collecteurs) ------------------

    @property
    def session(self) -> requests.Session:
        """Session HTTP avec retry automatique (lazy-initialisée).

        Configure des retries automatiques sur les codes HTTP transitoires :
        - 429 : Too Many Requests (rate limiting)
        - 500, 502, 503, 504 : Erreurs serveur

        Le délai entre retries suit une progression exponentielle :
        retry_backoff_factor * (2 ** (retry_number - 1))

        Returns:
            Session requests réutilisable avec retry intégré.
        """
        if self._session is None:
            self._session = requests.Session()
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
        return self._session

    def fetch_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET HTTP → JSON avec gestion d'erreurs complète.

        Args:
            url: URL de l'endpoint.
            params: Paramètres de query string (optionnel).

        Returns:
            Objet Python parsé depuis le JSON (dict, list, etc.).

        Raises:
            requests.HTTPError: Sur code 4xx/5xx après retries.
            ValueError: Si le corps de réponse n'est pas du JSON valide.
        """
        self.logger.debug("GET %s | params=%s", url, params)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return response.json()

    def fetch_xml(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> etree._Element:
        """GET HTTP → XML parsé avec gestion d'erreurs.

        Utilisé principalement pour l'API SDMX de l'INSEE.

        Args:
            url: URL de l'endpoint.
            params: Paramètres de query string (optionnel).

        Returns:
            Élément racine de l'arbre XML parsé (lxml).

        Raises:
            requests.HTTPError: Sur code 4xx/5xx.
            etree.XMLSyntaxError: Si le contenu n'est pas du XML valide.
        """
        self.logger.debug("GET (XML) %s | params=%s", url, params)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return etree.fromstring(response.content)

    def fetch_bytes(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """GET HTTP → contenu binaire brut (ZIP, fichiers volumineux).

        Args:
            url: URL de l'endpoint.
            params: Paramètres de query string (optionnel).

        Returns:
            Contenu brut de la réponse en bytes.

        Raises:
            requests.HTTPError: Sur code 4xx/5xx.
        """
        self.logger.debug("GET (bytes) %s", url)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return response.content

    def rate_limit_pause(self) -> None:
        """Pause de politesse entre deux appels API.

        Respecte le délai configuré dans `rate_limit_delay`.
        Évite de surcharger les APIs gratuites.
        """
        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)

    # --- Méthodes privées -------------------------------------------------

    def _resolve_output_path(self, base_dir: Optional[Path] = None) -> Path:
        """Calcule le chemin complet du fichier de sortie.

        Utilise les overrides de classe (output_subdir, output_filename)
        ou les valeurs par défaut basées sur source_name.

        Returns:
            Path absolue ou relative du fichier de sortie.
        """
        subdir = self.output_subdir or self.source_name
        filename = self.output_filename or f"{self.source_name}.csv"
        root = base_dir or self.config.raw_data_dir
        return root / subdir / filename

    def _setup_logging(self) -> None:
        """Configure le logger pour cette instance de collecteur.

        Format : YYYY-MM-DD HH:MM:SS | collectors.source_name | LEVEL | message
        """
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        # Éviter les handlers dupliqués si le module est rechargé
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


# =============================================================================
# Registry des collecteurs (système de plugins)
# =============================================================================

class CollectorRegistry:
    """Registre central de tous les collecteurs disponibles.

    Les collecteurs sont enregistrés automatiquement via le hook
    `__init_subclass__` de BaseCollector. Il suffit d'importer le module
    contenant un collecteur pour qu'il soit disponible.

    Ce pattern "plugin" permet d'ajouter de nouvelles sources sans
    modifier aucun fichier existant — juste créer un nouveau module.

    Usage:
        >>> CollectorRegistry.available()
        ['dpe', 'eurostat', 'insee', 'sitadel', 'weather']

        >>> result = CollectorRegistry.run("weather", config)
        >>> print(result)
        [SUCCESS] weather | 12345 lignes | 5.2s

        >>> results = CollectorRegistry.run_all(config)
    """

    # Stockage des classes de collecteurs (pas des instances)
    _collectors: ClassVar[Dict[str, Type[BaseCollector]]] = {}

    @classmethod
    def _register(cls, collector_class: Type[BaseCollector]) -> None:
        """Enregistre une classe de collecteur (appelé automatiquement).

        Args:
            collector_class: Sous-classe concrète de BaseCollector.
        """
        name = collector_class.source_name
        if name in cls._collectors:
            logging.getLogger("collectors.registry").warning(
                "Écrasement du collecteur existant '%s' par %s",
                name, collector_class.__name__,
            )
        cls._collectors[name] = collector_class
        logging.getLogger("collectors.registry").debug(
            "Collecteur '%s' enregistré (%s)", name, collector_class.__name__
        )

    @classmethod
    def available(cls) -> List[str]:
        """Liste les noms de tous les collecteurs enregistrés.

        Returns:
            Liste triée des source_name disponibles.
        """
        return sorted(cls._collectors.keys())

    @classmethod
    def get(cls, name: str) -> Type[BaseCollector]:
        """Récupère une classe de collecteur par son nom.

        Args:
            name: Le source_name du collecteur souhaité.

        Returns:
            La classe du collecteur (pas une instance).

        Raises:
            KeyError: Si aucun collecteur n'est enregistré avec ce nom.
        """
        if name not in cls._collectors:
            raise KeyError(
                f"Collecteur inconnu : '{name}'. "
                f"Disponibles : {cls.available()}"
            )
        return cls._collectors[name]

    @classmethod
    def run(cls, name: str, config: CollectorConfig) -> CollectorResult:
        """Instancie et exécute un collecteur par son nom.

        Args:
            name: Le source_name du collecteur.
            config: Configuration partagée.

        Returns:
            CollectorResult avec le bilan de la collecte.
        """
        collector_class = cls.get(name)
        collector = collector_class(config)
        return collector.run()

    @classmethod
    def run_all(
        cls,
        config: CollectorConfig,
        order: Optional[List[str]] = None,
    ) -> List[CollectorResult]:
        """Exécute plusieurs collecteurs en séquence.

        Args:
            config: Configuration partagée.
            order: Ordre d'exécution explicite. Si None, exécute
                   tous les collecteurs par ordre alphabétique.

        Returns:
            Liste des CollectorResult, un par collecteur.
        """
        logger = logging.getLogger("collectors.registry")
        names = order or cls.available()
        results: List[CollectorResult] = []

        logger.info(
            "Lancement de %d collecteurs : %s", len(names), names
        )

        for name in names:
            try:
                result = cls.run(name, config)
            except KeyError as exc:
                logger.error("Collecteur '%s' introuvable : %s", name, exc)
                result = CollectorResult(
                    name=name,
                    status=CollectorStatus.FAILED,
                    errors=[str(exc)],
                )
            results.append(result)

            # Log d'avancement si un collecteur échoue
            if result.status == CollectorStatus.FAILED:
                logger.warning(
                    "⚠ Collecteur '%s' ÉCHOUÉ — passage au suivant.", name
                )

        # Bilan final
        succeeded = sum(1 for r in results if r.status == CollectorStatus.SUCCESS)
        logger.info(
            "\n" + "="*60 + "\n"
            "  BILAN COLLECTE : %d/%d réussis\n" +
            "="*60,
            succeeded, len(results),
        )
        for result in results:
            logger.info("  %s", result)

        return results
