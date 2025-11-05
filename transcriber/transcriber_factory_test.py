"""
transcriber_factory_test.py

Unit tests for transcriber factory function.
"""

import unittest
from unittest.mock import patch
import tempfile
import shutil

from transcriber import get_transcriber
from transcriber.transcriber_factory import Transcriber
from transcriber.whisper_transcriber import WhisperTranscriber


class TestTranscriberFactory(unittest.TestCase):
    """Test the transcriber factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if shutil.os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_whisper_transcriber(self):
        """Test getting Whisper transcriber."""
        transcriber = get_transcriber("whisper", model_dir=self.temp_dir)

        self.assertIsInstance(transcriber, WhisperTranscriber)
        self.assertTrue(hasattr(transcriber, 'transcribe'))

    def test_default_backend_is_whisper(self):
        """Test that default backend is Whisper."""
        transcriber = get_transcriber(model_dir=self.temp_dir)

        self.assertIsInstance(transcriber, WhisperTranscriber)

    def test_unsupported_backend_raises_error(self):
        """Test that unsupported backend raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_transcriber("unsupported_backend")  # type: ignore

        self.assertIn("Unsupported backend", str(context.exception))
        self.assertIn("unsupported_backend", str(context.exception))

    def test_transcriber_implements_protocol(self):
        """Test that WhisperTranscriber implements Transcriber protocol."""
        transcriber = get_transcriber("whisper", model_dir=self.temp_dir)

        # Check that it has the required transcribe method
        self.assertTrue(hasattr(transcriber, 'transcribe'))
        self.assertTrue(callable(transcriber.transcribe))

    def test_factory_passes_parameters(self):
        """Test that factory passes parameters to transcriber."""
        transcriber = get_transcriber(
            backend="whisper",
            model_name="tiny",
            language="en",
            model_dir=self.temp_dir,
            device="cpu"
        )

        self.assertEqual(transcriber.model_name, "tiny")
        self.assertEqual(transcriber.language, "en")
        self.assertEqual(transcriber.model_dir, self.temp_dir)
        self.assertEqual(transcriber.device, "cpu")

    def test_factory_default_model_name(self):
        """Test that factory uses default model name."""
        transcriber = get_transcriber("whisper", model_dir=self.temp_dir)

        self.assertEqual(transcriber.model_name, "large-v3")

    def test_factory_default_device(self):
        """Test that factory uses default device."""
        transcriber = get_transcriber("whisper", model_dir=self.temp_dir)

        self.assertEqual(transcriber.device, "auto")

    def test_factory_returns_new_instance_each_time(self):
        """Test that factory returns new instances."""
        transcriber1 = get_transcriber("whisper", model_dir=self.temp_dir)
        transcriber2 = get_transcriber("whisper", model_dir=self.temp_dir)

        # Should be different instances
        self.assertIsNot(transcriber1, transcriber2)

        # But same type
        self.assertEqual(type(transcriber1), type(transcriber2))

    def test_returned_transcriber_has_correct_signature(self):
        """Test that returned transcriber has correct method signature."""
        transcriber = get_transcriber("whisper", model_dir=self.temp_dir)

        # Get the transcribe method
        transcribe_method = getattr(transcriber, 'transcribe')

        # Check it's callable
        self.assertTrue(callable(transcribe_method))

        # Check signature has required parameters
        import inspect
        sig = inspect.signature(transcribe_method)
        params = list(sig.parameters.keys())

        self.assertIn('input_file', params)
        self.assertIn('output_dir', params)


class TestTranscriberProtocol(unittest.TestCase):
    """Test the Transcriber Protocol."""

    def test_protocol_defines_transcribe_method(self):
        """Test that Protocol defines transcribe method."""
        # Transcriber is a Protocol, check it has the method defined
        self.assertTrue(hasattr(Transcriber, 'transcribe'))

    def test_whisper_transcriber_conforms_to_protocol(self):
        """Test that WhisperTranscriber conforms to Transcriber protocol."""
        from typing import Protocol, runtime_checkable

        # Create a runtime checkable version for testing
        @runtime_checkable
        class TestProtocol(Protocol):
            def transcribe(self, input_file: str, output_dir: str = "outputs"):
                ...

        temp_dir = tempfile.mkdtemp()
        try:
            transcriber = WhisperTranscriber(model_dir=temp_dir)

            # Check if it conforms to the protocol
            self.assertIsInstance(transcriber, TestProtocol)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
