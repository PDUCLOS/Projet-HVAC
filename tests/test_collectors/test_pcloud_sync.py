# -*- coding: utf-8 -*-
"""Tests pour le module de synchronisation pCloud."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import ProjectConfig
from src.collectors.pcloud_sync import PCloudSync


@pytest.fixture
def pcloud_sync(test_config: ProjectConfig) -> PCloudSync:
    """Instance PCloudSync pour les tests."""
    return PCloudSync(
        test_config,
        access_token="test_token",
        public_code="test_code",
    )


@pytest.fixture
def mock_pcloud_response():
    """Reponse mock de l'API pCloud showpublink."""
    return {
        "result": 0,
        "metadata": {
            "contents": [
                {
                    "name": "weather_france.csv",
                    "size": 1024000,
                    "hash": 12345,
                    "modified": "2026-02-15T10:00:00Z",
                    "fileid": 100,
                    "isfolder": False,
                },
                {
                    "name": "indicateurs_economiques.csv",
                    "size": 50000,
                    "hash": 67890,
                    "modified": "2026-02-14T08:00:00Z",
                    "fileid": 101,
                    "isfolder": False,
                },
                {
                    "name": "raw",
                    "isfolder": True,
                    "contents": [
                        {
                            "name": "dpe_france_all.csv",
                            "size": 500000000,
                            "hash": 11111,
                            "modified": "2026-02-10T12:00:00Z",
                            "fileid": 200,
                            "isfolder": False,
                        },
                    ],
                },
            ],
        },
    }


# ==================================================================
# Tests initialisation
# ==================================================================

class TestInit:
    """Tests pour l'initialisation."""

    def test_default_public_code(self, test_config):
        """Le code public par defaut est celui du README."""
        with patch.dict("os.environ", {}, clear=True):
            sync = PCloudSync(test_config)
            assert sync.public_code == "kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy"

    def test_custom_token(self, test_config):
        """Le token peut etre passe en parametre."""
        sync = PCloudSync(test_config, access_token="my_token")
        assert sync.access_token == "my_token"

    def test_env_variables(self, test_config):
        """Les variables d'environnement sont lues."""
        with patch.dict("os.environ", {
            "PCLOUD_ACCESS_TOKEN": "env_token",
            "PCLOUD_PUBLIC_CODE": "env_code",
        }):
            sync = PCloudSync(test_config)
            assert sync.access_token == "env_token"
            assert sync.public_code == "env_code"


# ==================================================================
# Tests listing
# ==================================================================

class TestListPublicFolder:
    """Tests pour le listing des fichiers."""

    def test_parses_response(self, pcloud_sync, mock_pcloud_response):
        """Parse correctement la reponse API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_pcloud_response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            files = pcloud_sync.list_public_folder()

        assert len(files) >= 2
        names = [f["name"] for f in files]
        assert "weather_france.csv" in names
        assert "indicateurs_economiques.csv" in names

    def test_includes_subfolder_files(self, pcloud_sync, mock_pcloud_response):
        """Les fichiers dans les sous-dossiers sont aussi listes."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_pcloud_response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            files = pcloud_sync.list_public_folder()

        names = [f["name"] for f in files]
        assert "dpe_france_all.csv" in names

    def test_api_error(self, pcloud_sync):
        """Retourne une liste vide sur erreur API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": 2000, "error": "test error"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            files = pcloud_sync.list_public_folder()

        assert files == []

    def test_network_error(self, pcloud_sync):
        """Retourne une liste vide sur erreur reseau."""
        import requests
        with patch.object(
            pcloud_sync.session, "get",
            side_effect=requests.ConnectionError("timeout"),
        ):
            files = pcloud_sync.list_public_folder()
        assert files == []


# ==================================================================
# Tests detection mises a jour
# ==================================================================

class TestCheckForUpdates:
    """Tests pour la detection des mises a jour."""

    def test_new_file_detected(self, pcloud_sync, mock_pcloud_response):
        """Un nouveau fichier est detecte comme mise a jour."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_pcloud_response
        mock_resp.raise_for_status = MagicMock()

        # Etat vide = tout est nouveau
        pcloud_sync.sync_state = {}

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            updates = pcloud_sync.check_for_updates()

        assert len(updates) >= 2
        assert any(u["update_reason"] == "nouveau" for u in updates)

    def test_modified_file_detected(self, pcloud_sync, mock_pcloud_response):
        """Un fichier modifie est detecte."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_pcloud_response
        mock_resp.raise_for_status = MagicMock()

        # Etat avec ancien hash
        pcloud_sync.sync_state = {
            "weather_france.csv": {"hash": 99999, "size": 1024000},
        }

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            updates = pcloud_sync.check_for_updates()

        weather_updates = [u for u in updates if u["name"] == "weather_france.csv"]
        assert len(weather_updates) == 1
        assert weather_updates[0]["update_reason"] == "modifie"

    def test_no_update_same_hash(self, pcloud_sync, mock_pcloud_response):
        """Pas de mise a jour si le hash n'a pas change."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_pcloud_response
        mock_resp.raise_for_status = MagicMock()

        # Meme hash et taille
        pcloud_sync.sync_state = {
            "weather_france.csv": {"hash": 12345, "size": 1024000},
            "indicateurs_economiques.csv": {"hash": 67890, "size": 50000},
        }

        with patch.object(pcloud_sync.session, "get", return_value=mock_resp):
            updates = pcloud_sync.check_for_updates()

        # Seul dpe_france_all.csv (dans sous-dossier) est nouveau
        top_level_updates = [
            u for u in updates
            if u["name"] in ("weather_france.csv", "indicateurs_economiques.csv")
        ]
        assert len(top_level_updates) == 0


# ==================================================================
# Tests file mapping
# ==================================================================

class TestFileMapping:
    """Tests pour le mapping des fichiers."""

    def test_weather_path(self, pcloud_sync):
        """weather_france.csv va dans data/raw/weather/."""
        path = pcloud_sync._get_local_path("weather_france.csv")
        assert path is not None
        assert "weather" in str(path)

    def test_dpe_path(self, pcloud_sync):
        """dpe_france_all.csv va dans data/raw/dpe/."""
        path = pcloud_sync._get_local_path("dpe_france_all.csv")
        assert path is not None
        assert "dpe" in str(path)

    def test_retrocompat_aura_names(self, pcloud_sync):
        """Les anciens noms AURA sont toujours reconnus."""
        path = pcloud_sync._get_local_path("weather_aura.csv")
        assert path is not None
        assert "weather" in str(path)

        path = pcloud_sync._get_local_path("dpe_aura_all.csv")
        assert path is not None
        assert "dpe" in str(path)

    def test_database_path(self, pcloud_sync):
        """hvac_market.db va dans data/."""
        path = pcloud_sync._get_local_path("hvac_market.db")
        assert path is not None
        assert path.name == "hvac_market.db"

    def test_unknown_csv(self, pcloud_sync):
        """Les CSV inconnus vont dans data/raw/other/."""
        path = pcloud_sync._get_local_path("unknown_file.csv")
        assert path is not None
        assert "other" in str(path)

    def test_irrelevant_file(self, pcloud_sync):
        """Les fichiers non-CSV/DB retournent None."""
        assert pcloud_sync._is_relevant_file("readme.txt") is False


# ==================================================================
# Tests sync state
# ==================================================================

class TestSyncState:
    """Tests pour la persistence de l'etat de synchronisation."""

    def test_save_and_load(self, pcloud_sync, tmp_path):
        """L'etat est correctement sauvegarde et recharge."""
        pcloud_sync.sync_state_file = tmp_path / ".sync_state.json"
        pcloud_sync.sync_state = {
            "weather_france.csv": {"hash": 12345, "last_sync": "2026-02-17"},
        }
        pcloud_sync._save_sync_state()

        # Recharger
        loaded = pcloud_sync._load_sync_state()
        assert loaded["weather_france.csv"]["hash"] == 12345

    def test_load_missing_file(self, pcloud_sync, tmp_path):
        """Retourne un dict vide si le fichier n'existe pas."""
        pcloud_sync.sync_state_file = tmp_path / "nonexistent.json"
        state = pcloud_sync._load_sync_state()
        assert state == {}

    def test_get_sync_status(self, pcloud_sync, tmp_path):
        """get_sync_status retourne un resume correct."""
        pcloud_sync.sync_state_file = tmp_path / ".sync_state.json"
        pcloud_sync.sync_state = {
            "a.csv": {"last_sync": "2026-01-01"},
            "b.csv": {"last_sync": "2026-02-01"},
        }
        pcloud_sync._save_sync_state()
        status = pcloud_sync.get_sync_status()
        assert status["n_files_tracked"] == 2


# ==================================================================
# Tests sync_and_update
# ==================================================================

class TestSyncAndUpdate:
    """Tests pour la synchronisation complete."""

    def test_no_updates(self, pcloud_sync, tmp_path):
        """Retourne le bon resultat quand rien a synchroniser."""
        pcloud_sync.sync_state_file = tmp_path / ".sync.json"

        with patch.object(pcloud_sync, "check_for_updates", return_value=[]):
            result = pcloud_sync.sync_and_update()

        assert result["files_checked"] == 0
        assert result["files_downloaded"] == 0
        assert result["pipeline_triggered"] is False

    def test_download_success(self, pcloud_sync, tmp_path):
        """Telecharge et met a jour l'etat."""
        pcloud_sync.sync_state_file = tmp_path / ".sync.json"

        updates = [{
            "name": "weather_france.csv",
            "hash": 12345,
            "size": 1024,
            "fileid": 100,
            "update_reason": "nouveau",
        }]

        with patch.object(pcloud_sync, "check_for_updates", return_value=updates):
            with patch.object(pcloud_sync, "download_public_file", return_value=True):
                with patch.object(pcloud_sync, "_trigger_pipeline_update"):
                    result = pcloud_sync.sync_and_update()

        assert result["files_downloaded"] == 1
        assert "weather_france.csv" in pcloud_sync.sync_state


# ==================================================================
# Tests upload
# ==================================================================

class TestUpload:
    """Tests pour l'upload vers pCloud."""

    def test_upload_requires_token(self, test_config, tmp_path):
        """L'upload echoue sans access_token."""
        sync = PCloudSync(test_config, access_token="", public_code="test")
        result = sync.upload_file(tmp_path / "test.csv")
        assert result is False

    def test_upload_file_missing(self, pcloud_sync, tmp_path):
        """L'upload echoue si le fichier n'existe pas."""
        result = pcloud_sync.upload_file(tmp_path / "nonexistent.csv")
        assert result is False

    def test_upload_success(self, pcloud_sync, tmp_path):
        """L'upload reussit avec un mock API."""
        # Creer un fichier test
        test_file = tmp_path / "test.csv"
        test_file.write_text("a,b\n1,2\n")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": 0,
            "metadata": [{"size": 10, "name": "test.csv"}],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(pcloud_sync.session, "post", return_value=mock_resp):
            result = pcloud_sync.upload_file(test_file)

        assert result is True

    def test_upload_collected_data(self, pcloud_sync, tmp_path):
        """upload_collected_data parcourt les fichiers et uploade."""
        # Creer des fichiers de test
        weather_dir = tmp_path / "raw" / "weather"
        weather_dir.mkdir(parents=True)
        (weather_dir / "weather_france.csv").write_text("a,b\n1,2\n")

        pcloud_sync.config = pcloud_sync.config.__class__(
            raw_data_dir=tmp_path / "raw",
        )

        with patch.object(pcloud_sync, "upload_file", return_value=True) as mock_upload:
            result = pcloud_sync.upload_collected_data()

        # Au moins 1 fichier uploade (weather_france.csv existe)
        assert result["files_uploaded"] >= 1
