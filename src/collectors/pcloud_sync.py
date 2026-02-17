# -*- coding: utf-8 -*-
"""
Synchronisation pCloud — Mise a jour automatisee des donnees.
==============================================================

Ce module gere la synchronisation des donnees entre pCloud et le projet
local, avec detection des nouveaux fichiers et mise a jour automatique
de la base de donnees.

Fonctionnalites :
    1. Connexion a l'API pCloud (authentification par token)
    2. Listing des fichiers disponibles dans le dossier partage
    3. Telechargement des fichiers nouveaux/modifies
    4. Detection des mises a jour (comparaison hash/date)
    5. Declenchement automatique du pipeline d'import

Prerequis :
    - Token d'acces pCloud (variable PCLOUD_ACCESS_TOKEN dans .env)
    - Lien public pCloud (variable PCLOUD_PUBLIC_LINK dans .env)

Usage :
    >>> from src.collectors.pcloud_sync import PCloudSync
    >>> sync = PCloudSync(config)
    >>> sync.sync_and_update()

    # Ou via CLI
    python -m src.pipeline sync_pcloud
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config.settings import ProjectConfig


class PCloudSync:
    """Gestionnaire de synchronisation pCloud pour les donnees HVAC.

    Gere le telechargement, la detection de mises a jour et
    l'integration automatique des nouvelles donnees dans le pipeline.

    Attributes:
        config: Configuration du projet.
        access_token: Token d'acces pCloud (OAuth2).
        public_code: Code du lien public pCloud.
        api_base: URL de base de l'API pCloud.
        sync_state_file: Fichier JSON tracking l'etat de synchronisation.
    """

    API_BASE = "https://eapi.pcloud.com"

    # Mapping des fichiers pCloud vers les dossiers locaux
    FILE_MAPPING = {
        "weather_france.csv": "weather",
        "indicateurs_economiques.csv": "insee",
        "ipi_hvac_france.csv": "eurostat",
        "permis_construire_france.csv": "sitadel",
        "dpe_france_all.csv": "dpe",
        "hvac_market.db": "_database",
        # Retrocompatibilite : anciens noms AURA
        "weather_aura.csv": "weather",
        "permis_construire_aura.csv": "sitadel",
        "dpe_aura_all.csv": "dpe",
    }

    def __init__(
        self,
        config: ProjectConfig,
        access_token: Optional[str] = None,
        public_code: Optional[str] = None,
    ) -> None:
        import os

        self.config = config
        self.logger = logging.getLogger("collectors.pcloud")

        self.access_token = access_token or os.getenv("PCLOUD_ACCESS_TOKEN", "")
        self.public_code = public_code or os.getenv(
            "PCLOUD_PUBLIC_CODE",
            "kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy",
        )

        self.session = requests.Session()
        self.session.timeout = 60

        # Fichier d'etat pour tracker les fichiers deja synchronises
        self.sync_state_file = Path("data") / ".pcloud_sync_state.json"
        self.sync_state = self._load_sync_state()

    # ==================================================================
    # API pCloud
    # ==================================================================

    def list_public_folder(self) -> List[Dict[str, Any]]:
        """Liste les fichiers disponibles dans le dossier public pCloud.

        Utilise l'endpoint showpublink pour acceder au contenu
        du lien public sans authentification.

        Returns:
            Liste de dictionnaires avec les infos fichier
            (name, size, hash, modified).
        """
        self.logger.info("Listing du dossier public pCloud...")

        try:
            resp = self.session.get(
                f"{self.API_BASE}/showpublink",
                params={"code": self.public_code},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("result") != 0:
                self.logger.error(
                    "Erreur API pCloud : %s", data.get("error", "inconnue")
                )
                return []

            metadata = data.get("metadata", {})
            contents = metadata.get("contents", [])

            files = []
            for item in contents:
                if not item.get("isfolder", False):
                    files.append({
                        "name": item.get("name", ""),
                        "size": item.get("size", 0),
                        "hash": item.get("hash", 0),
                        "modified": item.get("modified", ""),
                        "fileid": item.get("fileid", 0),
                    })

            # Lister aussi les sous-dossiers (recursif un niveau)
            for item in contents:
                if item.get("isfolder", False):
                    sub_contents = item.get("contents", [])
                    folder_name = item.get("name", "")
                    for sub_item in sub_contents:
                        if not sub_item.get("isfolder", False):
                            files.append({
                                "name": sub_item.get("name", ""),
                                "size": sub_item.get("size", 0),
                                "hash": sub_item.get("hash", 0),
                                "modified": sub_item.get("modified", ""),
                                "fileid": sub_item.get("fileid", 0),
                                "folder": folder_name,
                            })

            self.logger.info("  %d fichiers trouves sur pCloud", len(files))
            return files

        except requests.RequestException as e:
            self.logger.error("Erreur connexion pCloud : %s", e)
            return []

    def download_public_file(
        self,
        file_info: Dict[str, Any],
        dest_path: Path,
    ) -> bool:
        """Telecharge un fichier depuis le lien public pCloud.

        Args:
            file_info: Infos du fichier (depuis list_public_folder).
            dest_path: Chemin local de destination.

        Returns:
            True si le telechargement a reussi.
        """
        filename = file_info["name"]
        self.logger.info("  Telechargement : %s...", filename)

        try:
            # Obtenir le lien de telechargement
            resp = self.session.get(
                f"{self.API_BASE}/getpublinkdownload",
                params={
                    "code": self.public_code,
                    "fileid": file_info["fileid"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("result") != 0:
                self.logger.error(
                    "Erreur getpublinkdownload : %s",
                    data.get("error", "inconnue"),
                )
                return False

            # Construire l'URL de telechargement
            hosts = data.get("hosts", [])
            path = data.get("path", "")
            if not hosts or not path:
                self.logger.error("Pas de lien de telechargement disponible")
                return False

            download_url = f"https://{hosts[0]}{path}"

            # Telecharger en streaming (gros fichiers possibles)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session.get(download_url, stream=True, timeout=300) as dl:
                dl.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=8192):
                        f.write(chunk)

            size_mb = dest_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                "  ✓ %s telecharge (%.1f Mo)", filename, size_mb,
            )
            return True

        except requests.RequestException as e:
            self.logger.error("Erreur telechargement %s : %s", filename, e)
            return False

    # ==================================================================
    # Upload vers pCloud (necessite access_token)
    # ==================================================================

    def upload_file(self, local_path: Path, remote_folder_id: int = 0) -> bool:
        """Upload un fichier local vers pCloud.

        Necessite un access_token valide (pas le lien public).

        Args:
            local_path: Chemin du fichier local a uploader.
            remote_folder_id: ID du dossier pCloud de destination (0 = racine).

        Returns:
            True si l'upload a reussi.
        """
        if not self.access_token:
            self.logger.error(
                "PCLOUD_ACCESS_TOKEN requis pour l'upload. "
                "Definir la variable dans .env"
            )
            return False

        if not local_path.exists():
            self.logger.error("Fichier local introuvable : %s", local_path)
            return False

        self.logger.info("  Upload : %s...", local_path.name)

        try:
            with open(local_path, "rb") as f:
                resp = self.session.post(
                    f"{self.API_BASE}/uploadfile",
                    params={
                        "auth": self.access_token,
                        "folderid": remote_folder_id,
                        "filename": local_path.name,
                        "nopartial": 1,
                    },
                    files={"file": (local_path.name, f)},
                    timeout=600,
                )

            resp.raise_for_status()
            data = resp.json()

            if data.get("result") != 0:
                self.logger.error(
                    "Erreur upload pCloud : %s", data.get("error", "inconnue")
                )
                return False

            # Extraire les metadonnees du fichier uploade
            uploaded = data.get("metadata", [{}])
            if uploaded:
                size_mb = uploaded[0].get("size", 0) / (1024 * 1024)
                self.logger.info(
                    "  OK %s uploade (%.1f Mo)", local_path.name, size_mb,
                )

            return True

        except requests.RequestException as e:
            self.logger.error("Erreur upload %s : %s", local_path.name, e)
            return False

    def upload_collected_data(self, remote_folder_id: int = 0) -> Dict[str, Any]:
        """Upload tous les fichiers de donnees collectees vers pCloud.

        Parcourt les dossiers data/raw/ et uploade chaque fichier CSV
        correspondant au FILE_MAPPING.

        Args:
            remote_folder_id: ID du dossier pCloud de destination.

        Returns:
            Dictionnaire resume de l'upload.
        """
        self.logger.info("=" * 60)
        self.logger.info("  UPLOAD vers pCloud")
        self.logger.info("=" * 60)

        result = {
            "timestamp": datetime.now().isoformat(),
            "files_uploaded": 0,
            "files_failed": 0,
            "files_skipped": 0,
        }

        # Fichiers principaux a uploader (noms France)
        upload_targets = {
            "weather_france.csv": self.config.raw_data_dir / "weather" / "weather_france.csv",
            "indicateurs_economiques.csv": self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv",
            "ipi_hvac_france.csv": self.config.raw_data_dir / "eurostat" / "ipi_hvac_france.csv",
            "permis_construire_france.csv": self.config.raw_data_dir / "sitadel" / "permis_construire_france.csv",
            "dpe_france_all.csv": self.config.raw_data_dir / "dpe" / "dpe_france_all.csv",
        }

        # Ajouter la base de donnees si elle existe
        db_path = Path(self.config.database.db_path)
        if db_path.exists():
            upload_targets["hvac_market.db"] = db_path

        for filename, filepath in upload_targets.items():
            if not filepath.exists():
                self.logger.warning("  Fichier absent, skip : %s", filename)
                result["files_skipped"] += 1
                continue

            success = self.upload_file(filepath, remote_folder_id)
            if success:
                result["files_uploaded"] += 1
            else:
                result["files_failed"] += 1

        self.logger.info("=" * 60)
        self.logger.info("  RESUME UPLOAD")
        self.logger.info("  Fichiers uploades : %d", result["files_uploaded"])
        self.logger.info("  Echecs            : %d", result["files_failed"])
        self.logger.info("  Absents (skip)    : %d", result["files_skipped"])
        self.logger.info("=" * 60)

        return result

    # ==================================================================
    # Detection des mises a jour
    # ==================================================================

    def check_for_updates(self) -> List[Dict[str, Any]]:
        """Verifie quels fichiers ont ete modifies depuis la derniere sync.

        Compare les hash/tailles des fichiers pCloud avec l'etat
        sauvegarde localement.

        Returns:
            Liste des fichiers a mettre a jour.
        """
        self.logger.info("Verification des mises a jour pCloud...")

        remote_files = self.list_public_folder()
        if not remote_files:
            return []

        updates = []
        for file_info in remote_files:
            name = file_info["name"]

            # Verifier si le fichier est connu et pertinent
            if not self._is_relevant_file(name):
                continue

            # Comparer avec l'etat sauvegarde
            prev_state = self.sync_state.get(name, {})
            prev_hash = prev_state.get("hash", 0)
            prev_size = prev_state.get("size", 0)

            current_hash = file_info.get("hash", 0)
            current_size = file_info.get("size", 0)

            if current_hash != prev_hash or current_size != prev_size:
                file_info["update_reason"] = (
                    "nouveau" if not prev_state else "modifie"
                )
                updates.append(file_info)
                self.logger.info(
                    "  %s : %s (hash %s → %s)",
                    name, file_info["update_reason"], prev_hash, current_hash,
                )

        if not updates:
            self.logger.info("  Aucune mise a jour detectee.")
        else:
            self.logger.info("  %d fichier(s) a mettre a jour.", len(updates))

        return updates

    # ==================================================================
    # Synchronisation complete
    # ==================================================================

    def sync_and_update(
        self,
        force: bool = False,
        run_pipeline: bool = True,
    ) -> Dict[str, Any]:
        """Synchronise les donnees pCloud et met a jour la base.

        Pipeline complet :
        1. Verifier les mises a jour sur pCloud
        2. Telecharger les fichiers modifies
        3. Copier dans les bons dossiers data/raw/
        4. Declencher le pipeline d'import si necessaire

        Args:
            force: Si True, re-telecharge tout meme si rien n'a change.
            run_pipeline: Si True, lance l'import DB apres telechargement.

        Returns:
            Dictionnaire resume de la synchronisation.
        """
        self.logger.info("=" * 60)
        self.logger.info("  SYNCHRONISATION pCloud")
        self.logger.info("=" * 60)

        result = {
            "timestamp": datetime.now().isoformat(),
            "files_checked": 0,
            "files_downloaded": 0,
            "files_failed": 0,
            "pipeline_triggered": False,
        }

        # 1. Verifier les mises a jour
        if force:
            updates = self.list_public_folder()
            updates = [f for f in updates if self._is_relevant_file(f["name"])]
        else:
            updates = self.check_for_updates()

        result["files_checked"] = len(updates)

        if not updates:
            self.logger.info("Aucun fichier a synchroniser.")
            return result

        # 2. Telecharger chaque fichier
        downloaded = []
        for file_info in updates:
            dest = self._get_local_path(file_info["name"])
            if dest is None:
                continue

            success = self.download_public_file(file_info, dest)
            if success:
                downloaded.append(file_info)
                result["files_downloaded"] += 1

                # Mettre a jour l'etat de sync
                self.sync_state[file_info["name"]] = {
                    "hash": file_info.get("hash", 0),
                    "size": file_info.get("size", 0),
                    "last_sync": datetime.now().isoformat(),
                }
            else:
                result["files_failed"] += 1

        # 3. Sauvegarder l'etat de sync
        self._save_sync_state()

        # 4. Declencher le pipeline si des donnees ont change
        if downloaded and run_pipeline:
            self.logger.info("-" * 40)
            self.logger.info("Declenchement du pipeline d'import...")
            try:
                self._trigger_pipeline_update()
                result["pipeline_triggered"] = True
            except Exception as e:
                self.logger.error("Erreur pipeline : %s", e)

        # Resume
        self.logger.info("=" * 60)
        self.logger.info("  RESUME SYNCHRONISATION")
        self.logger.info("  Fichiers verifies   : %d", result["files_checked"])
        self.logger.info("  Fichiers telecharges : %d", result["files_downloaded"])
        self.logger.info("  Echecs              : %d", result["files_failed"])
        self.logger.info("  Pipeline declenche  : %s", result["pipeline_triggered"])
        self.logger.info("=" * 60)

        return result

    # ==================================================================
    # Pipeline integration
    # ==================================================================

    def _trigger_pipeline_update(self) -> None:
        """Declenche la mise a jour de la base de donnees.

        Enchaine : import_data → clean → merge → features → outliers.
        """
        from src.database.db_manager import DatabaseManager
        from src.processing.clean_data import DataCleaner
        from src.processing.merge_datasets import DatasetMerger
        from src.processing.feature_engineering import FeatureEngineer
        from src.processing.outlier_detection import OutlierDetector

        # 1. Import dans la base
        self.logger.info("  1/5 Import des donnees dans la BDD...")
        db = DatabaseManager(self.config.database.connection_string)
        db.import_collected_data(raw_data_dir=self.config.raw_data_dir)

        # 2. Nettoyage
        self.logger.info("  2/5 Nettoyage des donnees brutes...")
        cleaner = DataCleaner(self.config)
        cleaner.clean_all()

        # 3. Fusion
        self.logger.info("  3/5 Fusion multi-sources...")
        merger = DatasetMerger(self.config)
        merger.build_ml_dataset()

        # 4. Feature engineering
        self.logger.info("  4/5 Feature engineering...")
        fe = FeatureEngineer(self.config)
        fe.engineer_from_file()

        # 5. Detection outliers
        self.logger.info("  5/5 Detection des outliers...")
        detector = OutlierDetector(self.config)
        import pandas as pd
        features_path = self.config.features_data_dir / "hvac_features_dataset.csv"
        if features_path.exists():
            df = pd.read_csv(features_path)
            df_treated, _ = detector.run_full_analysis(df, strategy="clip")
            df_treated.to_csv(features_path, index=False)

        self.logger.info("  Pipeline de mise a jour termine.")

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _is_relevant_file(self, filename: str) -> bool:
        """Verifie si un fichier est pertinent pour le projet."""
        return filename in self.FILE_MAPPING or filename.endswith((".csv", ".db"))

    def _get_local_path(self, filename: str) -> Optional[Path]:
        """Determine le chemin local pour un fichier pCloud."""
        subdir = self.FILE_MAPPING.get(filename)

        if subdir == "_database":
            # Base de donnees directement dans data/
            return Path("data") / filename

        if subdir:
            return self.config.raw_data_dir / subdir / filename

        # Fichier CSV inconnu → mettre dans raw/other/
        if filename.endswith(".csv"):
            return self.config.raw_data_dir / "other" / filename

        return None

    def _load_sync_state(self) -> Dict[str, Any]:
        """Charge l'etat de synchronisation depuis le fichier JSON."""
        if self.sync_state_file.exists():
            try:
                return json.loads(self.sync_state_file.read_text())
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_sync_state(self) -> None:
        """Sauvegarde l'etat de synchronisation."""
        self.sync_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.sync_state_file.write_text(
            json.dumps(self.sync_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info("  Etat de sync sauvegarde → %s", self.sync_state_file)

    def get_sync_status(self) -> Dict[str, Any]:
        """Retourne un resume de l'etat de synchronisation actuel."""
        state = self._load_sync_state()
        return {
            "n_files_tracked": len(state),
            "files": {
                name: info.get("last_sync", "jamais")
                for name, info in state.items()
            },
        }
