"""
model_downloader_factory_test.py

Unit tests for model downloader factory function.
"""

import unittest
from unittest.mock import patch
import tempfile
import shutil

from model_downloader import get_downloader, WHISPER_MODELS
from model_downloader.model_downloader_factory import ModelDownloader
from model_downloader.huggingface_downloader import HuggingFaceDownloader


class TestModelDownloaderFactory(unittest.TestCase):
    """Test the model downloader factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if shutil.os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_huggingface_downloader(self):
        """Test getting HuggingFace downloader."""
        downloader = get_downloader("huggingface", models_dir=self.temp_dir)

        self.assertIsInstance(downloader, HuggingFaceDownloader)
        self.assertTrue(hasattr(downloader, 'download'))

    def test_default_platform_is_huggingface(self):
        """Test that default platform is HuggingFace."""
        downloader = get_downloader(models_dir=self.temp_dir)

        self.assertIsInstance(downloader, HuggingFaceDownloader)

    def test_unsupported_platform_raises_error(self):
        """Test that unsupported platform raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_downloader("unsupported_platform")  # type: ignore

        self.assertIn("Unsupported platform", str(context.exception))
        self.assertIn("unsupported_platform", str(context.exception))

    def test_downloader_implements_protocol(self):
        """Test that HuggingFaceDownloader implements ModelDownloader protocol."""
        downloader = get_downloader("huggingface", models_dir=self.temp_dir)

        # Check that it has the required download method
        self.assertTrue(hasattr(downloader, 'download'))
        self.assertTrue(callable(downloader.download))

    def test_returned_downloader_has_correct_signature(self):
        """Test that returned downloader has correct method signature."""
        downloader = get_downloader("huggingface", models_dir=self.temp_dir)

        # Get the download method
        download_method = getattr(downloader, 'download')

        # Check it's callable
        self.assertTrue(callable(download_method))

        # Check signature has required parameters
        import inspect
        sig = inspect.signature(download_method)
        params = list(sig.parameters.keys())

        self.assertIn('models', params)
        self.assertIn('force', params)
        self.assertIn('skip_space_check', params)

    def test_factory_returns_new_instance_each_time(self):
        """Test that factory returns new instances."""
        downloader1 = get_downloader("huggingface", models_dir=self.temp_dir)
        downloader2 = get_downloader("huggingface", models_dir=self.temp_dir)

        # Should be different instances
        self.assertIsNot(downloader1, downloader2)

        # But same type
        self.assertEqual(type(downloader1), type(downloader2))

    def test_models_dir_parameter_is_used(self):
        """Test that models_dir parameter is passed to downloader."""
        custom_dir = self.temp_dir + "/custom"
        downloader = get_downloader("huggingface", models_dir=custom_dir)

        self.assertEqual(str(downloader.models_dir), custom_dir)

    def test_default_models_dir(self):
        """Test that default models_dir is 'models'."""
        downloader = get_downloader("huggingface")

        self.assertEqual(str(downloader.models_dir), "models")


class TestModelDownloaderProtocol(unittest.TestCase):
    """Test the ModelDownloader Protocol."""

    def test_protocol_defines_download_method(self):
        """Test that Protocol defines download method."""
        # ModelDownloader is a Protocol, check it has the method defined
        self.assertTrue(hasattr(ModelDownloader, 'download'))

    def test_huggingface_downloader_conforms_to_protocol(self):
        """Test that HuggingFaceDownloader conforms to ModelDownloader protocol."""
        from typing import Protocol, runtime_checkable

        # Create a runtime checkable version for testing
        @runtime_checkable
        class TestProtocol(Protocol):
            def download(self, models=None, force=False, skip_space_check=False):
                ...

        downloader = HuggingFaceDownloader()

        # Check if it conforms to the protocol
        self.assertIsInstance(downloader, TestProtocol)


class TestWhisperModelsConstant(unittest.TestCase):
    """Test the WHISPER_MODELS constant exported from module."""

    def test_whisper_models_exported(self):
        """Test that WHISPER_MODELS is exported from module."""
        self.assertIsNotNone(WHISPER_MODELS)
        self.assertIsInstance(WHISPER_MODELS, dict)

    def test_whisper_models_has_expected_entries(self):
        """Test that WHISPER_MODELS has expected model entries."""
        expected_models = ["tiny", "base", "small", "medium", "large-v3"]

        for model in expected_models:
            self.assertIn(model, WHISPER_MODELS)

    def test_whisper_models_structure(self):
        """Test that each model has required fields."""
        for model_name, model_info in WHISPER_MODELS.items():
            self.assertIn("repo", model_info, f"{model_name} missing 'repo'")
            self.assertIn("size_gb", model_info, f"{model_name} missing 'size_gb'")
            self.assertIn("description", model_info, f"{model_name} missing 'description'")


if __name__ == '__main__':
    unittest.main()
