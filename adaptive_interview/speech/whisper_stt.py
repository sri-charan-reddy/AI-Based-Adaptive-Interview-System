"""Local Speech-to-Text provider powered by OpenAI Whisper.

This module provides the WhisperSpeechToText class, implementing SpeechToTextProvider
using the local OpenAI Whisper library without requiring any API keys or network calls.
The Whisper model is loaded lazily upon first transcription or explicit load call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_interview.speech.base import SpeechToTextProvider
from adaptive_interview.speech.models import TranscriptionResult


class WhisperSpeechToText(SpeechToTextProvider):
    """Speech-to-text provider utilizing local OpenAI Whisper models.

    Models are loaded lazily so importing or initializing this class does not trigger
    immediate model downloads or heavyweight memory allocations.
    """

    DEFAULT_MODEL: str = "base"
    SUPPORTED_AUDIO_EXTENSIONS: set[str] = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
        ".aac",
        ".wma",
        ".webm",
        ".opus",
        ".mp4",
        ".mpga",
        ".oga",
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize WhisperSpeechToText configuration.

        Args:
            model_name: Whisper model identifier ('tiny', 'base', 'small', 'medium', 'large').
                        Defaults to 'base'.
            device: Target device for inference (e.g., 'cpu', 'cuda'). Defaults to auto.
            model: Optional pre-loaded or mock Whisper model instance for testing/dependency injection.

        Raises:
            ValueError: If model_name is not a valid non-empty string.
        """
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        self.model_name: str = model_name.strip()
        self.device: str | None = device
        self._model: Any | None = model

    @property
    def is_model_loaded(self) -> bool:
        """Check whether the Whisper model has been loaded into memory."""
        return self._model is not None

    def load_model(self) -> Any:
        """Lazily load and cache the Whisper model instance.

        Returns:
            The loaded Whisper model.

        Raises:
            RuntimeError: If the 'openai-whisper' package is not installed or loading fails.
        """
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Whisper is not installed. Please install 'openai-whisper' "
                "(e.g., pip install openai-whisper) to use WhisperSpeechToText."
            ) from exc

        try:
            load_kwargs: dict[str, Any] = {}
            if self.device is not None:
                load_kwargs["device"] = self.device
            self._model = whisper.load_model(self.model_name, **load_kwargs)
            return self._model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_name}': {exc}"
            ) from exc

    def validate_audio_path(self, audio_path: str | Path) -> Path:
        """Validate that the audio path exists and points to a supported audio file.

        Args:
            audio_path: File path to audio file (string or pathlib.Path).

        Returns:
            Validated pathlib.Path object.

        Raises:
            ValueError: If audio_path is empty, wrong type, a directory, or an unsupported format.
            FileNotFoundError: If the audio file does not exist on disk.
        """
        if audio_path is None or not isinstance(audio_path, (str, Path)):
            raise ValueError(
                f"Audio path must be a string or Path, got {type(audio_path).__name__ if audio_path is not None else 'None'}"
            )

        path_str = str(audio_path).strip()
        if not path_str:
            raise ValueError("Audio path cannot be empty.")

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path_str}")

        if path.is_dir():
            raise ValueError(f"Audio path points to a directory, not a file: {path_str}")

        if path.suffix.lower() not in self.SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio file format '{path.suffix}'. "
                f"Supported formats: {sorted(self.SUPPORTED_AUDIO_EXTENSIONS)}"
            )

        return path

    def transcribe(
        self,
        audio_path: str | Path,
        **transcribe_kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe an audio file into text using OpenAI Whisper.

        Args:
            audio_path: Path to the audio file to transcribe.
            **transcribe_kwargs: Optional kwargs forwarded to whisper.model.transcribe().

        Returns:
            A TranscriptionResult containing text, language, duration, and provider="whisper".

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the audio path is invalid or format unsupported.
            RuntimeError: If Whisper is not installed, model loading fails, or transcription fails.
        """
        validated_path = self.validate_audio_path(audio_path)
        model = self.load_model()

        try:
            raw_result = model.transcribe(str(validated_path), **transcribe_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

        # Extract text
        text = raw_result.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        # Extract detected language
        language = raw_result.get("language")
        if language is not None and isinstance(language, str):
            language = language.strip()
            if not language:
                language = None
        else:
            language = None

        # Extract duration if present or calculate from segments
        duration: float | None = None
        if isinstance(raw_result.get("duration"), (int, float)):
            duration = float(raw_result["duration"])
        elif "segments" in raw_result and raw_result["segments"]:
            segments = raw_result["segments"]
            if isinstance(segments, list) and len(segments) > 0:
                last_seg = segments[-1]
                if isinstance(last_seg, dict) and "end" in last_seg:
                    try:
                        duration = float(last_seg["end"])
                    except (ValueError, TypeError):
                        duration = None

        return TranscriptionResult(
            text=text,
            language=language,
            duration_seconds=duration,
            provider="whisper",
        )
