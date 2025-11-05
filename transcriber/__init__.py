"""
transcriber package

Factory-based transcriber supporting multiple backends (currently Whisper).
"""

from .transcriber_factory import (
    get_transcriber,
    Transcriber,
    BackendType,
)
from .whisper_transcriber import WhisperTranscriber

__all__ = [
    "get_transcriber",
    "Transcriber",
    "BackendType",
    "WhisperTranscriber",
]
