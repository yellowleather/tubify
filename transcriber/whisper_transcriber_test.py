"""
whisper_transcriber_test.py

Unit tests for Whisper transcriber functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call, mock_open
import os
import tempfile
import shutil
import platform

from transcriber.whisper_transcriber import (
    ts_hhmmssms,
    write_srt,
    write_vtt,
    WhisperTranscriber,
)


class TestTimestampFormatting(unittest.TestCase):
    """Test the ts_hhmmssms timestamp formatting function."""

    def test_zero_timestamp(self):
        """Test formatting of zero timestamp."""
        result = ts_hhmmssms(0.0)
        self.assertEqual(result, "00:00:00,000")

    def test_seconds_only(self):
        """Test formatting of seconds only."""
        result = ts_hhmmssms(5.5)
        self.assertEqual(result, "00:00:05,500")

    def test_minutes_and_seconds(self):
        """Test formatting with minutes and seconds."""
        result = ts_hhmmssms(125.750)
        self.assertEqual(result, "00:02:05,750")

    def test_hours_minutes_seconds(self):
        """Test formatting with hours, minutes, and seconds."""
        result = ts_hhmmssms(3665.123)
        self.assertEqual(result, "01:01:05,123")

    def test_milliseconds_rounding(self):
        """Test that milliseconds are properly truncated."""
        result = ts_hhmmssms(1.9999)
        self.assertEqual(result, "00:00:01,999")

    def test_large_timestamp(self):
        """Test formatting of large timestamp."""
        result = ts_hhmmssms(36000.0)  # 10 hours
        self.assertEqual(result, "10:00:00,000")


class TestWriteSRT(unittest.TestCase):
    """Test the write_srt subtitle writing function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "output.srt")

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_write_empty_segments(self):
        """Test writing empty segments list."""
        write_srt([], self.output_file)

        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, 'r') as f:
            content = f.read()
            self.assertEqual(content, "")

    def test_write_single_segment(self):
        """Test writing a single segment."""
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"}
        ]

        write_srt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            content = f.read()

        self.assertIn("1\n", content)
        self.assertIn("00:00:00,000 --> 00:00:02,500", content)
        self.assertIn("Hello world", content)

    def test_write_multiple_segments(self):
        """Test writing multiple segments."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First line"},
            {"start": 2.5, "end": 5.0, "text": "Second line"},
        ]

        write_srt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            content = f.read()

        self.assertIn("1\n", content)
        self.assertIn("2\n", content)
        self.assertIn("First line", content)
        self.assertIn("Second line", content)

    def test_srt_format_structure(self):
        """Test that SRT format structure is correct."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Test"}
        ]

        write_srt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            lines = f.readlines()

        self.assertEqual(lines[0].strip(), "1")
        self.assertIn("-->", lines[1])
        self.assertEqual(lines[2].strip(), "Test")


class TestWriteVTT(unittest.TestCase):
    """Test the write_vtt subtitle writing function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "output.vtt")

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_vtt_header(self):
        """Test that VTT file has proper header."""
        segments = []

        write_vtt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            content = f.read()

        self.assertTrue(content.startswith("WEBVTT\n\n"))

    def test_write_single_segment_vtt(self):
        """Test writing a single segment in VTT format."""
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"}
        ]

        write_vtt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            content = f.read()

        self.assertIn("WEBVTT", content)
        self.assertIn("00:00:00.000 --> 00:00:02.500", content)
        self.assertIn("Hello world", content)

    def test_vtt_uses_periods_not_commas(self):
        """Test that VTT format uses periods for milliseconds, not commas."""
        segments = [
            {"start": 1.5, "end": 3.0, "text": "Test"}
        ]

        write_vtt(segments, self.output_file)

        with open(self.output_file, 'r') as f:
            content = f.read()

        self.assertIn("00:00:01.500", content)
        self.assertNotIn(",", content.split("\n")[2])  # Skip header lines


class TestWhisperTranscriber(unittest.TestCase):
    """Test the WhisperTranscriber class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_default(self):
        """Test default initialization."""
        transcriber = WhisperTranscriber(model_dir=self.temp_dir)

        self.assertEqual(transcriber.model_name, "large-v3")
        self.assertIsNone(transcriber.language)
        self.assertEqual(transcriber.model_dir, self.temp_dir)
        self.assertIsNotNone(transcriber.device)
        self.assertIsNotNone(transcriber.compute_type)

    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        transcriber = WhisperTranscriber(
            model_name="tiny",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.model_name, "tiny")

    def test_initialization_custom_language(self):
        """Test initialization with custom language."""
        transcriber = WhisperTranscriber(
            language="en",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.language, "en")

    def test_initialization_creates_model_dir(self):
        """Test that initialization creates model directory."""
        new_dir = os.path.join(self.temp_dir, "new_models")
        self.assertFalse(os.path.exists(new_dir))

        transcriber = WhisperTranscriber(model_dir=new_dir)

        self.assertTrue(os.path.exists(new_dir))

    @patch('transcriber.whisper_transcriber.platform.system')
    def test_device_defaults_macos(self, mock_system):
        """Test device defaults on macOS."""
        mock_system.return_value = "Darwin"

        transcriber = WhisperTranscriber(
            device="auto",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.device, "auto")
        self.assertEqual(transcriber.compute_type, "float16")

    @patch('transcriber.whisper_transcriber.platform.system')
    def test_device_defaults_linux(self, mock_system):
        """Test device defaults on Linux."""
        mock_system.return_value = "Linux"

        transcriber = WhisperTranscriber(
            device="auto",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.device, "auto")
        self.assertEqual(transcriber.compute_type, "int8")

    def test_device_override(self):
        """Test that user can override device."""
        transcriber = WhisperTranscriber(
            device="cpu",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.device, "cpu")

    def test_compute_type_override(self):
        """Test that user can override compute type."""
        transcriber = WhisperTranscriber(
            compute_type="int8",
            model_dir=self.temp_dir
        )

        self.assertEqual(transcriber.compute_type, "int8")

    def test_default_device_and_compute_method(self):
        """Test the _default_device_and_compute method."""
        transcriber = WhisperTranscriber(model_dir=self.temp_dir)

        # Test with explicit values
        device, compute = transcriber._default_device_and_compute("cuda", "float16")
        self.assertEqual(device, "cuda")
        self.assertEqual(compute, "float16")

        # Test with auto values
        device, compute = transcriber._default_device_and_compute("auto", None)
        self.assertEqual(device, "auto")
        self.assertIn(compute, ["float16", "int8"])

    @patch('transcriber.whisper_transcriber.subprocess.run')
    @patch('transcriber.whisper_transcriber.faster_whisper')
    @patch('transcriber.whisper_transcriber.whisperx')
    def test_transcribe_creates_output_files(self, mock_whisperx, mock_fw, mock_subprocess):
        """Test that transcribe creates expected output files."""
        # This is a complex integration test that would require extensive mocking
        # Keeping it as a placeholder for when full integration tests are needed
        pass


class TestFFMpegNorm(unittest.TestCase):
    """Test the ffmpeg_norm audio normalization function."""

    @patch('transcriber.whisper_transcriber.subprocess.run')
    def test_ffmpeg_command_structure(self, mock_subprocess):
        """Test that ffmpeg is called with correct arguments."""
        from transcriber.whisper_transcriber import ffmpeg_norm

        input_file = "input.mp4"
        output_file = "output.wav"

        ffmpeg_norm(input_file, output_file)

        # Check subprocess.run was called
        mock_subprocess.assert_called_once()

        # Check arguments
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn("ffmpeg", call_args)
        self.assertIn("-y", call_args)
        self.assertIn("-i", call_args)
        self.assertIn(input_file, call_args)
        self.assertIn("-ac", call_args)
        self.assertIn("1", call_args)
        self.assertIn("-ar", call_args)
        self.assertIn("16000", call_args)
        self.assertIn(output_file, call_args)

        # Check check=True was passed
        self.assertTrue(mock_subprocess.call_args[1]['check'])


if __name__ == '__main__':
    unittest.main()
