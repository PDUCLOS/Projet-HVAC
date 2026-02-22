# -*- coding: utf-8 -*-
"""
pCloud synchronization — Automated data update.
=================================================

This module manages data synchronization between pCloud and the local
project, with detection of new files and automatic database updates.

Features:
    1. Connection to the pCloud API (token authentication)
    2. Listing of available files in the shared folder
    3. Download of new/modified files
    4. Update detection (hash/date comparison)
    5. Automatic triggering of the import pipeline

Prerequisites:
    - pCloud access token (PCLOUD_ACCESS_TOKEN variable in .env)
    - pCloud public link (PCLOUD_PUBLIC_LINK variable in .env)

Usage:
    >>> from src.collectors.pcloud_sync import PCloudSync
    >>> sync = PCloudSync(config)
    >>> sync.sync_and_update()

    # Or via CLI
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
    """pCloud synchronization manager for HVAC data.

    Manages downloading, update detection, and automatic
    integration of new data into the pipeline.

    Attributes:
        config: Project configuration.
        access_token: pCloud access token (OAuth2).
        public_code: pCloud public link code.
        api_base: pCloud API base URL.
        sync_state_file: JSON file tracking the synchronization state.
    """

    API_BASE = "https://eapi.pcloud.com"

    # Mapping of pCloud files to local directories
    FILE_MAPPING = {
        "weather_france.csv": "weather",
        "indicateurs_economiques.csv": "insee",
        "ipi_hvac_france.csv": "eurostat",
        "permis_construire_france.csv": "sitadel",
        "dpe_france_all.csv": "dpe",
        "hvac_market.db": "_database",
        # Backward compatibility: old AURA names
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
        self.public_code = public_code or os.getenv("PCLOUD_PUBLIC_CODE", "")

        self.session = requests.Session()
        self.session.timeout = 60

        # State file to track already synchronized files
        self.sync_state_file = Path("data") / ".pcloud_sync_state.json"
        self.sync_state = self._load_sync_state()

    # ==================================================================
    # pCloud API
    # ==================================================================

    def list_public_folder(self) -> List[Dict[str, Any]]:
        """List available files in the pCloud public folder.

        Uses the showpublink endpoint to access the content
        of the public link without authentication.

        Returns:
            List of dictionaries with file info
            (name, size, hash, modified).
        """
        self.logger.info("Listing pCloud public folder...")

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
                    "pCloud API error: %s", data.get("error", "unknown")
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

            # Also list subfolders (recursive one level)
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

            self.logger.info("  %d files found on pCloud", len(files))
            return files

        except requests.RequestException as e:
            self.logger.error("pCloud connection error: %s", e)
            return []

    def download_public_file(
        self,
        file_info: Dict[str, Any],
        dest_path: Path,
    ) -> bool:
        """Download a file from the pCloud public link.

        Args:
            file_info: File info (from list_public_folder).
            dest_path: Local destination path.

        Returns:
            True if the download succeeded.
        """
        filename = file_info["name"]
        self.logger.info("  Downloading: %s...", filename)

        try:
            # Get the download link
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
                    "getpublinkdownload error: %s",
                    data.get("error", "unknown"),
                )
                return False

            # Build the download URL
            hosts = data.get("hosts", [])
            path = data.get("path", "")
            if not hosts or not path:
                self.logger.error("No download link available")
                return False

            download_url = f"https://{hosts[0]}{path}"

            # Download with streaming (large files possible)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session.get(download_url, stream=True, timeout=300) as dl:
                dl.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=8192):
                        f.write(chunk)

            size_mb = dest_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                "  ✓ %s downloaded (%.1f MB)", filename, size_mb,
            )
            return True

        except requests.RequestException as e:
            self.logger.error("Download error %s: %s", filename, e)
            return False

    # ==================================================================
    # Upload to pCloud (requires access_token)
    # ==================================================================

    def upload_file(self, local_path: Path, remote_folder_id: int = 0) -> bool:
        """Upload a local file to pCloud.

        Requires a valid access_token (not the public link).

        Args:
            local_path: Path of the local file to upload.
            remote_folder_id: pCloud destination folder ID (0 = root).

        Returns:
            True if the upload succeeded.
        """
        if not self.access_token:
            self.logger.error(
                "PCLOUD_ACCESS_TOKEN required for upload. "
                "Define the variable in .env"
            )
            return False

        if not local_path.exists():
            self.logger.error("Local file not found: %s", local_path)
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
                    "pCloud upload error: %s", data.get("error", "unknown")
                )
                return False

            # Extract metadata of the uploaded file
            uploaded = data.get("metadata", [{}])
            if uploaded:
                size_mb = uploaded[0].get("size", 0) / (1024 * 1024)
                self.logger.info(
                    "  OK %s uploaded (%.1f MB)", local_path.name, size_mb,
                )

            return True

        except requests.RequestException as e:
            self.logger.error("Upload error %s: %s", local_path.name, e)
            return False

    def upload_collected_data(self, remote_folder_id: int = 0) -> Dict[str, Any]:
        """Upload all collected data files to pCloud.

        Traverses the data/raw/ directories and uploads each CSV file
        matching the FILE_MAPPING.

        Args:
            remote_folder_id: pCloud destination folder ID.

        Returns:
            Dictionary summarizing the upload.
        """
        self.logger.info("=" * 60)
        self.logger.info("  UPLOAD to pCloud")
        self.logger.info("=" * 60)

        result = {
            "timestamp": datetime.now().isoformat(),
            "files_uploaded": 0,
            "files_failed": 0,
            "files_skipped": 0,
        }

        # Main files to upload (France names)
        upload_targets = {
            "weather_france.csv": self.config.raw_data_dir / "weather" / "weather_france.csv",
            "indicateurs_economiques.csv": self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv",
            "ipi_hvac_france.csv": self.config.raw_data_dir / "eurostat" / "ipi_hvac_france.csv",
            "permis_construire_france.csv": self.config.raw_data_dir / "sitadel" / "permis_construire_france.csv",
            "dpe_france_all.csv": self.config.raw_data_dir / "dpe" / "dpe_france_all.csv",
            "reference_departements.csv": self.config.raw_data_dir / "insee" / "reference_departements.csv",
        }

        # Add features datasets
        features_dir = self.config.features_data_dir
        for name in ["hvac_ml_dataset.csv", "hvac_features_dataset.csv"]:
            path = features_dir / name
            if path.exists():
                upload_targets[name] = path

        # Add the database if it exists
        db_path = Path(self.config.database.db_path)
        if db_path.exists():
            upload_targets["hvac_market.db"] = db_path

        for filename, filepath in upload_targets.items():
            if not filepath.exists():
                self.logger.warning("  File missing, skip: %s", filename)
                result["files_skipped"] += 1
                continue

            success = self.upload_file(filepath, remote_folder_id)
            if success:
                result["files_uploaded"] += 1
            else:
                result["files_failed"] += 1

        self.logger.info("=" * 60)
        self.logger.info("  UPLOAD SUMMARY")
        self.logger.info("  Files uploaded    : %d", result["files_uploaded"])
        self.logger.info("  Failures          : %d", result["files_failed"])
        self.logger.info("  Missing (skip)    : %d", result["files_skipped"])
        self.logger.info("=" * 60)

        return result

    # ==================================================================
    # Update detection
    # ==================================================================

    def check_for_updates(self) -> List[Dict[str, Any]]:
        """Check which files have been modified since the last sync.

        Compares the hashes/sizes of pCloud files with the locally
        saved state.

        Returns:
            List of files to update.
        """
        self.logger.info("Checking for pCloud updates...")

        remote_files = self.list_public_folder()
        if not remote_files:
            return []

        updates = []
        for file_info in remote_files:
            name = file_info["name"]

            # Check if the file is known and relevant
            if not self._is_relevant_file(name):
                continue

            # Compare with the saved state
            prev_state = self.sync_state.get(name, {})
            prev_hash = prev_state.get("hash", 0)
            prev_size = prev_state.get("size", 0)

            current_hash = file_info.get("hash", 0)
            current_size = file_info.get("size", 0)

            if current_hash != prev_hash or current_size != prev_size:
                file_info["update_reason"] = (
                    "new" if not prev_state else "modified"
                )
                updates.append(file_info)
                self.logger.info(
                    "  %s : %s (hash %s → %s)",
                    name, file_info["update_reason"], prev_hash, current_hash,
                )

        if not updates:
            self.logger.info("  No updates detected.")
        else:
            self.logger.info("  %d file(s) to update.", len(updates))

        return updates

    # ==================================================================
    # Full synchronization
    # ==================================================================

    def sync_and_update(
        self,
        force: bool = False,
        run_pipeline: bool = True,
    ) -> Dict[str, Any]:
        """Synchronize pCloud data and update the database.

        Full pipeline:
        1. Check for updates on pCloud
        2. Download modified files
        3. Copy into the correct data/raw/ directories
        4. Trigger the import pipeline if necessary

        Args:
            force: If True, re-download everything even if nothing has changed.
            run_pipeline: If True, launch the DB import after download.

        Returns:
            Dictionary summarizing the synchronization.
        """
        self.logger.info("=" * 60)
        self.logger.info("  pCloud SYNCHRONIZATION")
        self.logger.info("=" * 60)

        result = {
            "timestamp": datetime.now().isoformat(),
            "files_checked": 0,
            "files_downloaded": 0,
            "files_failed": 0,
            "pipeline_triggered": False,
        }

        # 1. Check for updates
        if force:
            updates = self.list_public_folder()
            updates = [f for f in updates if self._is_relevant_file(f["name"])]
        else:
            updates = self.check_for_updates()

        result["files_checked"] = len(updates)

        if not updates:
            self.logger.info("No files to synchronize.")
            return result

        # 2. Download each file
        downloaded = []
        for file_info in updates:
            dest = self._get_local_path(file_info["name"])
            if dest is None:
                continue

            success = self.download_public_file(file_info, dest)
            if success:
                downloaded.append(file_info)
                result["files_downloaded"] += 1

                # Update the sync state
                self.sync_state[file_info["name"]] = {
                    "hash": file_info.get("hash", 0),
                    "size": file_info.get("size", 0),
                    "last_sync": datetime.now().isoformat(),
                }
            else:
                result["files_failed"] += 1

        # 3. Save the sync state
        self._save_sync_state()

        # 4. Trigger the pipeline if data has changed
        if downloaded and run_pipeline:
            self.logger.info("-" * 40)
            self.logger.info("Triggering import pipeline...")
            try:
                self._trigger_pipeline_update()
                result["pipeline_triggered"] = True
            except Exception as e:
                self.logger.error("Pipeline error: %s", e)

        # Summary
        self.logger.info("=" * 60)
        self.logger.info("  SYNCHRONIZATION SUMMARY")
        self.logger.info("  Files checked       : %d", result["files_checked"])
        self.logger.info("  Files downloaded    : %d", result["files_downloaded"])
        self.logger.info("  Failures            : %d", result["files_failed"])
        self.logger.info("  Pipeline triggered  : %s", result["pipeline_triggered"])
        self.logger.info("=" * 60)

        return result

    # ==================================================================
    # Pipeline integration
    # ==================================================================

    def _trigger_pipeline_update(self) -> None:
        """Trigger the database update.

        Chains: import_data -> clean -> merge -> features -> outliers.
        """
        from src.database.db_manager import DatabaseManager
        from src.processing.clean_data import DataCleaner
        from src.processing.merge_datasets import DatasetMerger
        from src.processing.feature_engineering import FeatureEngineer
        from src.processing.outlier_detection import OutlierDetector

        # 1. Import into the database
        self.logger.info("  1/5 Importing data into the database...")
        db = DatabaseManager(self.config.database.connection_string)
        db.import_collected_data(raw_data_dir=self.config.raw_data_dir)

        # 2. Cleaning
        self.logger.info("  2/5 Cleaning raw data...")
        cleaner = DataCleaner(self.config)
        cleaner.clean_all()

        # 3. Merging
        self.logger.info("  3/5 Multi-source merging...")
        merger = DatasetMerger(self.config)
        merger.build_ml_dataset()

        # 4. Feature engineering
        self.logger.info("  4/5 Feature engineering...")
        fe = FeatureEngineer(self.config)
        fe.engineer_from_file()

        # 5. Outlier detection
        self.logger.info("  5/5 Outlier detection...")
        detector = OutlierDetector(self.config)
        import pandas as pd
        features_path = self.config.features_data_dir / "hvac_features_dataset.csv"
        if features_path.exists():
            df = pd.read_csv(features_path)
            df_treated, _ = detector.run_full_analysis(df, strategy="clip")
            df_treated.to_csv(features_path, index=False)

        self.logger.info("  Update pipeline completed.")

    # ==================================================================
    # Utilities
    # ==================================================================

    def _is_relevant_file(self, filename: str) -> bool:
        """Check if a file is relevant to the project."""
        return filename in self.FILE_MAPPING or filename.endswith((".csv", ".db"))

    def _get_local_path(self, filename: str) -> Optional[Path]:
        """Determine the local path for a pCloud file."""
        subdir = self.FILE_MAPPING.get(filename)

        if subdir == "_database":
            # Database directly in data/
            return Path("data") / filename

        if subdir:
            return self.config.raw_data_dir / subdir / filename

        # Unknown CSV file -> put in raw/other/
        if filename.endswith(".csv"):
            return self.config.raw_data_dir / "other" / filename

        return None

    def _load_sync_state(self) -> Dict[str, Any]:
        """Load the synchronization state from the JSON file."""
        if self.sync_state_file.exists():
            try:
                return json.loads(self.sync_state_file.read_text())
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_sync_state(self) -> None:
        """Save the synchronization state."""
        self.sync_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.sync_state_file.write_text(
            json.dumps(self.sync_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info("  Sync state saved → %s", self.sync_state_file)

    def get_sync_status(self) -> Dict[str, Any]:
        """Return a summary of the current synchronization state."""
        state = self._load_sync_state()
        return {
            "n_files_tracked": len(state),
            "files": {
                name: info.get("last_sync", "never")
                for name, info in state.items()
            },
        }
