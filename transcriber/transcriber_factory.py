"""
transcriber_factory.py

Factory for creating transcribers for different backends.
Currently supports Whisper (faster-whisper + WhisperX).
"""

from typing import Dict, Literal, Optional, Protocol

from .whisper_transcriber import WhisperTranscriber


class Transcriber(Protocol):
    """Protocol for transcriber implementations."""

    def transcribe(self, input_file: str, output_dir: str = "outputs") -> Dict[str, str]:
        """
        Transcribe audio/video file.

        Args:
            input_file: Path to audio/video file
            output_dir: Output directory for results

        Returns:
            Dictionary with paths to generated files
        """
        ...


BackendType = Literal["whisper"]


def get_transcriber(
    backend: BackendType = "whisper",
    model_name: str = "large-v3",
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    model_dir: str = "models",
    device: str = "auto"
) -> Transcriber:
    """
    Factory function to get the appropriate transcriber.

    Args:
        backend: Transcription backend ("whisper" currently supported)
        model_name: Model name (for Whisper: "tiny", "base", "small", "medium", "large-v3")
        compute_type: Compute type ("float16", "int8", etc.)
        language: Force language code or None for auto-detect
        model_dir: Directory for model cache
        device: Device backend ("metal", "cpu", "cuda", "auto")

    Returns:
        A Transcriber instance for the specified backend

    Raises:
        ValueError: If the backend is not supported

    Examples:
        >>> transcriber = get_transcriber("whisper", model_name="large-v3")
        >>> result = transcriber.transcribe("input.mp4", "outputs/")
    """
    if backend == "whisper":
        return WhisperTranscriber(
            model_name=model_name,
            compute_type=compute_type,
            language=language,
            model_dir=model_dir,
            device=device
        )
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: whisper"
        )
