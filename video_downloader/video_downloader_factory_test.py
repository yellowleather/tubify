"""
test_factory.py

Unit tests for video downloader factory function.
"""

import unittest
from unittest.mock import patch

from video_downloader import get_downloader
from video_downloader.video_downloader_factory import VideoDownloader
from video_downloader.youtube_downloader import YouTubeDownloader


class TestVideoDownloaderFactory(unittest.TestCase):
    """Test the video downloader factory function."""

    def test_get_youtube_downloader(self):
        """Test getting YouTube downloader."""
        downloader = get_downloader("youtube")

        self.assertIsInstance(downloader, YouTubeDownloader)
        self.assertTrue(hasattr(downloader, 'download'))

    def test_default_platform_is_youtube(self):
        """Test that default platform is YouTube."""
        downloader = get_downloader()

        self.assertIsInstance(downloader, YouTubeDownloader)

    def test_unsupported_platform_raises_error(self):
        """Test that unsupported platform raises ValueError."""
        with self.assertRaises(ValueError) as context:
            get_downloader("unsupported_platform")  # type: ignore

        self.assertIn("Unsupported platform", str(context.exception))
        self.assertIn("unsupported_platform", str(context.exception))

    def test_youtube_downloader_implements_protocol(self):
        """Test that YouTubeDownloader implements VideoDownloader protocol."""
        downloader = get_downloader("youtube")

        # Check that it has the required download method
        self.assertTrue(hasattr(downloader, 'download'))
        self.assertTrue(callable(downloader.download))

    def test_returned_downloader_has_correct_signature(self):
        """Test that returned downloader has correct method signature."""
        downloader = get_downloader("youtube")

        # Get the download method
        download_method = getattr(downloader, 'download')

        # Check it's callable
        self.assertTrue(callable(download_method))

        # Check signature has url and output_dir parameters
        import inspect
        sig = inspect.signature(download_method)
        params = list(sig.parameters.keys())

        self.assertIn('url', params)
        self.assertIn('output_dir', params)

    def test_factory_returns_new_instance_each_time(self):
        """Test that factory returns new instances."""
        downloader1 = get_downloader("youtube")
        downloader2 = get_downloader("youtube")

        # Should be different instances
        self.assertIsNot(downloader1, downloader2)

        # But same type
        self.assertEqual(type(downloader1), type(downloader2))


class TestVideoDownloaderProtocol(unittest.TestCase):
    """Test the VideoDownloader Protocol."""

    def test_protocol_defines_download_method(self):
        """Test that Protocol defines download method."""
        # VideoDownloader is a Protocol, check it has the method defined
        self.assertTrue(hasattr(VideoDownloader, 'download'))

    def test_youtube_downloader_conforms_to_protocol(self):
        """Test that YouTubeDownloader conforms to VideoDownloader protocol."""
        from typing import Protocol, runtime_checkable

        # Create a runtime checkable version for testing
        @runtime_checkable
        class TestProtocol(Protocol):
            def download(self, url: str, output_dir: str) -> int:
                ...

        downloader = YouTubeDownloader()

        # Check if it conforms to the protocol
        self.assertIsInstance(downloader, TestProtocol)


if __name__ == '__main__':
    unittest.main()
