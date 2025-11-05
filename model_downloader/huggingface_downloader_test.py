"""
huggingface_downloader_test.py

Unit tests for HuggingFace model downloader functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import os
import tempfile
import shutil
from pathlib import Path

from model_downloader.huggingface_downloader import (
    check_disk_space,
    HuggingFaceDownloader,
    WHISPER_MODELS,
)


class TestCheckDiskSpace(unittest.TestCase):
    """Test the check_disk_space helper function."""

    @patch('model_downloader.huggingface_downloader.shutil.disk_usage')
    def test_sufficient_space_returns_true(self, mock_disk_usage):
        """Test that sufficient disk space returns True."""
        # Mock 100GB free space
        mock_usage = Mock()
        mock_usage.free = 100 * (1024**3)  # 100GB in bytes
        mock_disk_usage.return_value = mock_usage

        result = check_disk_space(10)  # Request 10GB

        self.assertTrue(result)

    @patch('model_downloader.huggingface_downloader.shutil.disk_usage')
    def test_insufficient_space_returns_false(self, mock_disk_usage):
        """Test that insufficient disk space returns False."""
        # Mock 5GB free space
        mock_usage = Mock()
        mock_usage.free = 5 * (1024**3)  # 5GB in bytes
        mock_disk_usage.return_value = mock_usage

        result = check_disk_space(10)  # Request 10GB

        self.assertFalse(result)

    @patch('model_downloader.huggingface_downloader.shutil.disk_usage')
    def test_exception_returns_true(self, mock_disk_usage):
        """Test that exception during check returns True (fail-safe)."""
        mock_disk_usage.side_effect = Exception("Disk check failed")

        result = check_disk_space(10)

        self.assertTrue(result)  # Should fail-safe to True

    @patch('model_downloader.huggingface_downloader.shutil.disk_usage')
    def test_includes_buffer_space(self, mock_disk_usage):
        """Test that function includes 1GB buffer in calculation."""
        # Mock exactly required space + 1GB
        mock_usage = Mock()
        mock_usage.free = 11 * (1024**3)  # 11GB in bytes
        mock_disk_usage.return_value = mock_usage

        result = check_disk_space(10)  # Request 10GB

        self.assertTrue(result)  # 11GB > 10GB + 1GB buffer

        # Now test with exactly 10GB (should fail because of buffer)
        mock_usage.free = 10 * (1024**3)
        result = check_disk_space(10)

        self.assertFalse(result)  # 10GB < 10GB + 1GB buffer


class TestHuggingFaceDownloader(unittest.TestCase):
    """Test the HuggingFaceDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = HuggingFaceDownloader(models_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that downloader initializes correctly."""
        downloader = HuggingFaceDownloader(models_dir=self.temp_dir)

        self.assertIsInstance(downloader, HuggingFaceDownloader)
        self.assertEqual(downloader.models_dir, Path(self.temp_dir))
        self.assertTrue(downloader.models_dir.exists())

    def test_initialization_creates_directory(self):
        """Test that initialization creates models directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "new_models")
        self.assertFalse(os.path.exists(new_dir))

        downloader = HuggingFaceDownloader(models_dir=new_dir)

        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))

    def test_get_available_models(self):
        """Test getting available models."""
        models = self.downloader.get_available_models()

        self.assertIsInstance(models, dict)
        self.assertIn("tiny", models)
        self.assertIn("base", models)
        self.assertIn("large-v3", models)

        # Check model structure
        for model_name, model_info in models.items():
            self.assertIn("repo", model_info)
            self.assertIn("size_gb", model_info)
            self.assertIn("description", model_info)

    def test_get_available_models_returns_copy(self):
        """Test that get_available_models returns a copy, not reference."""
        models1 = self.downloader.get_available_models()
        models2 = self.downloader.get_available_models()

        self.assertIsNot(models1, models2)
        self.assertEqual(models1, models2)

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_single_model_success(self, mock_snapshot):
        """Test successful single model download."""
        mock_snapshot.return_value = str(self.temp_dir)

        result = self.downloader.download_single_model("tiny", force=True)

        self.assertIsNotNone(result)
        mock_snapshot.assert_called_once()

        # Check repo_id argument
        call_args = mock_snapshot.call_args
        self.assertEqual(call_args[1]['repo_id'], "Systran/faster-whisper-tiny")

    def test_download_single_model_unknown_model(self):
        """Test downloading unknown model returns None."""
        result = self.downloader.download_single_model("unknown_model")

        self.assertIsNone(result)

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_single_model_skip_existing(self, mock_snapshot):
        """Test that existing model is skipped when force=False."""
        # Create fake model file
        cache_dir = Path(self.temp_dir) / "models--Systran--faster-whisper-tiny"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "model.bin").touch()

        result = self.downloader.download_single_model("tiny", force=False)

        self.assertIsNotNone(result)
        mock_snapshot.assert_not_called()  # Should not download

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_single_model_force_redownload(self, mock_snapshot):
        """Test that force=True redownloads existing model."""
        # Create fake model file
        cache_dir = Path(self.temp_dir) / "models--Systran--faster-whisper-tiny"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "model.bin").touch()

        mock_snapshot.return_value = str(cache_dir)

        result = self.downloader.download_single_model("tiny", force=True)

        self.assertIsNotNone(result)
        mock_snapshot.assert_called_once()  # Should download despite existing

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_single_model_exception_handling(self, mock_snapshot):
        """Test that exceptions during download are handled gracefully."""
        mock_snapshot.side_effect = Exception("Download failed")

        result = self.downloader.download_single_model("tiny", force=True)

        self.assertIsNone(result)

    @patch('model_downloader.huggingface_downloader.check_disk_space')
    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_multiple_models(self, mock_snapshot, mock_disk_space):
        """Test downloading multiple models."""
        mock_disk_space.return_value = True
        mock_snapshot.return_value = str(self.temp_dir)

        result = self.downloader.download(
            models=["tiny", "base"],
            skip_space_check=True
        )

        self.assertEqual(len(result["successful"]), 2)
        self.assertEqual(len(result["failed"]), 0)
        self.assertIn("tiny", result["successful"])
        self.assertIn("base", result["successful"])

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_invalid_model_name(self, mock_snapshot):
        """Test that invalid model names are handled."""
        result = self.downloader.download(
            models=["invalid_model"],
            skip_space_check=True
        )

        self.assertEqual(len(result["successful"]), 0)
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("invalid_model", result["failed"])
        mock_snapshot.assert_not_called()

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_mixed_valid_invalid(self, mock_snapshot):
        """Test downloading mix of valid and invalid models."""
        mock_snapshot.return_value = str(self.temp_dir)

        result = self.downloader.download(
            models=["tiny", "invalid"],
            skip_space_check=True
        )

        # Should reject entire batch if any model is invalid
        self.assertEqual(len(result["successful"]), 0)
        self.assertEqual(len(result["failed"]), 2)

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_none_downloads_all(self, mock_snapshot):
        """Test that models=None attempts to download all models."""
        mock_snapshot.return_value = str(self.temp_dir)

        result = self.downloader.download(
            models=None,
            skip_space_check=True
        )

        # Should attempt all models defined in WHISPER_MODELS
        total = len(result["successful"]) + len(result["failed"])
        self.assertEqual(total, len(WHISPER_MODELS))

    @patch('model_downloader.huggingface_downloader.check_disk_space')
    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_respects_skip_space_check(self, mock_snapshot, mock_disk_space):
        """Test that skip_space_check parameter works."""
        mock_snapshot.return_value = str(self.temp_dir)

        # With skip_space_check=True
        self.downloader.download(
            models=["tiny"],
            skip_space_check=True
        )

        mock_disk_space.assert_not_called()

    @patch('model_downloader.huggingface_downloader.check_disk_space')
    @patch('model_downloader.huggingface_downloader.Console')
    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_space_check_abort(self, mock_snapshot, mock_console_class, mock_disk_space):
        """Test that user can abort when disk space is insufficient."""
        mock_disk_space.return_value = False
        mock_console = MagicMock()
        mock_console.input.return_value = "n"  # User says no
        mock_console_class.return_value = mock_console

        # Recreate downloader to use mocked console
        downloader = HuggingFaceDownloader(models_dir=self.temp_dir)
        downloader.console = mock_console

        result = downloader.download(
            models=["tiny"],
            skip_space_check=False
        )

        self.assertEqual(len(result["successful"]), 0)
        self.assertEqual(len(result["failed"]), 1)
        mock_snapshot.assert_not_called()

    @patch('model_downloader.huggingface_downloader.snapshot_download')
    def test_download_partial_failure(self, mock_snapshot):
        """Test handling of partial download failures."""
        # First call succeeds, second fails
        mock_snapshot.side_effect = [
            str(self.temp_dir),  # Success for first model
            Exception("Download failed")  # Failure for second model
        ]

        result = self.downloader.download(
            models=["tiny", "base"],
            skip_space_check=True
        )

        self.assertEqual(len(result["successful"]), 1)
        self.assertEqual(len(result["failed"]), 1)
        self.assertIn("tiny", result["successful"])
        self.assertIn("base", result["failed"])

    def test_whisper_models_constant(self):
        """Test that WHISPER_MODELS constant has expected structure."""
        self.assertIsInstance(WHISPER_MODELS, dict)
        self.assertGreater(len(WHISPER_MODELS), 0)

        # Check expected models exist
        expected_models = ["tiny", "base", "small", "medium", "large-v3"]
        for model in expected_models:
            self.assertIn(model, WHISPER_MODELS)

            # Check structure
            model_info = WHISPER_MODELS[model]
            self.assertIn("repo", model_info)
            self.assertIn("size_gb", model_info)
            self.assertIn("description", model_info)

            # Check types
            self.assertIsInstance(model_info["repo"], str)
            self.assertIsInstance(model_info["size_gb"], (int, float))
            self.assertIsInstance(model_info["description"], str)

            # Check values
            self.assertTrue(model_info["repo"].startswith("Systran/"))
            self.assertGreater(model_info["size_gb"], 0)
            self.assertGreater(len(model_info["description"]), 0)


if __name__ == '__main__':
    unittest.main()
