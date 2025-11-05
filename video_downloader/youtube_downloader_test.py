"""
test_youtube_downloader.py

Unit tests for YouTube downloader functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import os
import tempfile
import shutil

from video_downloader.youtube_downloader import (
    extract_video_id_from_shorts,
    normalize_youtube_url,
    ensure_dir,
    YouTubeDownloader,
)


class TestExtractVideoIdFromShorts(unittest.TestCase):
    """Test the extract_video_id_from_shorts helper function."""

    def test_standard_shorts_url(self):
        """Test extraction from standard shorts URL."""
        url = "https://www.youtube.com/shorts/QMJAUg2snas"
        result = extract_video_id_from_shorts(url)
        self.assertEqual(result, "QMJAUg2snas")

    def test_shorts_url_with_query_params(self):
        """Test extraction from shorts URL with query parameters."""
        url = "https://www.youtube.com/shorts/QMJAUg2snas?feature=share"
        result = extract_video_id_from_shorts(url)
        self.assertEqual(result, "QMJAUg2snas")

    def test_shorts_url_without_www(self):
        """Test extraction from shorts URL without www."""
        url = "https://youtube.com/shorts/abc123XYZ"
        result = extract_video_id_from_shorts(url)
        self.assertEqual(result, "abc123XYZ")

    def test_short_video_id(self):
        """Test extraction with minimum length video ID."""
        url = "https://www.youtube.com/shorts/abcde"
        result = extract_video_id_from_shorts(url)
        self.assertEqual(result, "abcde")

    def test_video_id_with_underscores_and_hyphens(self):
        """Test extraction with video ID containing underscores and hyphens."""
        url = "https://www.youtube.com/shorts/abc_123-XYZ"
        result = extract_video_id_from_shorts(url)
        self.assertEqual(result, "abc_123-XYZ")

    def test_non_shorts_url_returns_none(self):
        """Test that non-shorts URLs return None."""
        url = "https://www.youtube.com/watch?v=QMJAUg2snas"
        result = extract_video_id_from_shorts(url)
        self.assertIsNone(result)

    def test_invalid_url_returns_none(self):
        """Test that invalid URLs return None."""
        url = "https://example.com/video"
        result = extract_video_id_from_shorts(url)
        self.assertIsNone(result)

    def test_empty_url_returns_none(self):
        """Test that empty URL returns None."""
        url = ""
        result = extract_video_id_from_shorts(url)
        self.assertIsNone(result)


class TestNormalizeYoutubeUrl(unittest.TestCase):
    """Test the normalize_youtube_url helper function."""

    def test_shorts_url_conversion(self):
        """Test conversion of shorts URL to watch URL."""
        url = "https://www.youtube.com/shorts/QMJAUg2snas"
        result = normalize_youtube_url(url)
        self.assertEqual(result, "https://www.youtube.com/watch?v=QMJAUg2snas")

    def test_shorts_url_with_params_conversion(self):
        """Test conversion of shorts URL with query params."""
        url = "https://www.youtube.com/shorts/QMJAUg2snas?feature=share"
        result = normalize_youtube_url(url)
        self.assertEqual(result, "https://www.youtube.com/watch?v=QMJAUg2snas")

    def test_youtu_be_url_conversion(self):
        """Test conversion of youtu.be short URL."""
        url = "https://youtu.be/QMJAUg2snas"
        result = normalize_youtube_url(url)
        self.assertEqual(result, "https://www.youtube.com/watch?v=QMJAUg2snas")

    def test_youtu_be_url_with_params(self):
        """Test conversion of youtu.be URL with params."""
        url = "https://youtu.be/QMJAUg2snas?t=10"
        result = normalize_youtube_url(url)
        self.assertEqual(result, "https://www.youtube.com/watch?v=QMJAUg2snas")

    def test_watch_url_unchanged(self):
        """Test that watch URLs remain unchanged."""
        url = "https://www.youtube.com/watch?v=QMJAUg2snas"
        result = normalize_youtube_url(url)
        self.assertEqual(result, url)

    def test_watch_url_with_params_unchanged(self):
        """Test that watch URLs with params remain unchanged."""
        url = "https://www.youtube.com/watch?v=QMJAUg2snas&t=10"
        result = normalize_youtube_url(url)
        self.assertEqual(result, url)

    def test_url_with_v_param(self):
        """Test extraction of v parameter from non-standard URL."""
        url = "https://www.youtube.com/embed?v=QMJAUg2snas"
        result = normalize_youtube_url(url)
        self.assertEqual(result, "https://www.youtube.com/watch?v=QMJAUg2snas")

    def test_unknown_url_returned_unchanged(self):
        """Test that unknown URLs are returned unchanged."""
        url = "https://example.com/video"
        result = normalize_youtube_url(url)
        self.assertEqual(result, url)


class TestEnsureDir(unittest.TestCase):
    """Test the ensure_dir helper function."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_new_directory(self):
        """Test creating a new directory."""
        new_dir = os.path.join(self.temp_dir, "test_dir")
        self.assertFalse(os.path.exists(new_dir))

        ensure_dir(new_dir)

        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))

    def test_create_nested_directories(self):
        """Test creating nested directories."""
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        self.assertFalse(os.path.exists(nested_dir))

        ensure_dir(nested_dir)

        self.assertTrue(os.path.exists(nested_dir))
        self.assertTrue(os.path.isdir(nested_dir))

    def test_existing_directory_no_error(self):
        """Test that existing directory doesn't raise error."""
        existing_dir = os.path.join(self.temp_dir, "existing")
        os.makedirs(existing_dir)

        # Should not raise error
        ensure_dir(existing_dir)

        self.assertTrue(os.path.exists(existing_dir))


class TestYouTubeDownloader(unittest.TestCase):
    """Test the YouTubeDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.downloader = YouTubeDownloader()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test that downloader initializes correctly."""
        downloader = YouTubeDownloader()
        self.assertIsInstance(downloader, YouTubeDownloader)

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_success(self, mock_yt_dlp_class):
        """Test successful video download."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'test_video', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = os.path.join(self.temp_dir, 'test_video.mp4')
        mock_ydl.download.return_value = 0

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download(url, self.temp_dir)

        # Assert
        self.assertEqual(result, 0)
        mock_ydl.extract_info.assert_called()
        mock_ydl.download.assert_called_once()

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_shorts_url_normalized(self, mock_yt_dlp_class):
        """Test that shorts URLs are normalized before download."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'test_short', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = os.path.join(self.temp_dir, 'test_short.mp4')
        mock_ydl.download.return_value = 0

        # Execute with shorts URL
        url = "https://www.youtube.com/shorts/test123"
        result = self.downloader.download(url, self.temp_dir)

        # Assert - should call with normalized URL
        self.assertEqual(result, 0)
        # Check that extract_info was called with normalized URL
        call_args = mock_ydl.extract_info.call_args
        called_url = call_args[0][0]
        self.assertIn("watch?v=", called_url)
        self.assertNotIn("shorts", called_url)

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    @patch('os.path.exists')
    def test_download_skip_existing_file(self, mock_exists, mock_yt_dlp_class):
        """Test that existing files are skipped."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'existing_video', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info
        expected_file = os.path.join(self.temp_dir, 'existing_video.mp4')
        mock_ydl.prepare_filename.return_value = expected_file

        # Mock file exists
        mock_exists.return_value = True

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download(url, self.temp_dir)

        # Assert
        self.assertEqual(result, 0)
        mock_ydl.download.assert_not_called()  # Should not download

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_error_handling(self, mock_yt_dlp_class):
        """Test error handling during download."""
        # Setup mock to raise DownloadError
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        from yt_dlp.utils import DownloadError
        mock_ydl.extract_info.side_effect = DownloadError("Test error")

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download(url, self.temp_dir)

        # Assert - should return error code
        self.assertEqual(result, 3)

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_unexpected_error(self, mock_yt_dlp_class):
        """Test handling of unexpected errors."""
        # Setup mock to raise generic exception
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_ydl.extract_info.side_effect = Exception("Unexpected error")

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download(url, self.temp_dir)

        # Assert - should return error code
        self.assertEqual(result, 4)

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_creates_output_directory(self, mock_yt_dlp_class):
        """Test that output directory is created if it doesn't exist."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'test_video', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info

        # Use non-existent directory
        new_dir = os.path.join(self.temp_dir, 'new_output')
        self.assertFalse(os.path.exists(new_dir))

        mock_ydl.prepare_filename.return_value = os.path.join(new_dir, 'test_video.mp4')
        mock_ydl.download.return_value = 0

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        result = self.downloader.download(url, new_dir)

        # Assert
        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(new_dir))

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_with_android_headers(self, mock_yt_dlp_class):
        """Test that Android headers are set correctly."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'test_video', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = os.path.join(self.temp_dir, 'test_video.mp4')
        mock_ydl.download.return_value = 0

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        self.downloader.download(url, self.temp_dir)

        # Assert - check that YoutubeDL was called with correct options
        call_args = mock_yt_dlp_class.call_args
        options = call_args[0][0]

        self.assertIn('http_headers', options)
        self.assertIn('User-Agent', options['http_headers'])
        self.assertIn('android', options['http_headers']['User-Agent'].lower())

        self.assertIn('extractor_args', options)
        self.assertIn('youtube', options['extractor_args'])

    @patch('video_downloader.youtube_downloader.yt_dlp.YoutubeDL')
    def test_download_format_selector(self, mock_yt_dlp_class):
        """Test that format selector is set correctly."""
        # Setup mock
        mock_ydl = MagicMock()
        mock_yt_dlp_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {'title': 'test_video', 'ext': 'mp4'}
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = os.path.join(self.temp_dir, 'test_video.mp4')
        mock_ydl.download.return_value = 0

        # Execute
        url = "https://www.youtube.com/watch?v=test123"
        self.downloader.download(url, self.temp_dir)

        # Assert - check format selector
        call_args = mock_yt_dlp_class.call_args
        options = call_args[0][0]

        self.assertIn('format', options)
        self.assertIn('18', options['format'])  # Should include format 18


if __name__ == '__main__':
    unittest.main()
