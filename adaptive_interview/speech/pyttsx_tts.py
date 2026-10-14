"""Local Text-to-Speech provider powered by pyttsx3.

This module provides the Pyttsx3TextToSpeech class, implementing TextToSpeechProvider
using the local pyttsx3 library (using SAPI5 natively on Windows) without requiring
any external API keys, network calls, or cloud services.
The pyttsx3 engine is loaded lazily upon first synthesis or explicit load call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptive_interview.speech.base import TextToSpeechProvider
from adaptive_interview.speech.models import SpeechSynthesisResult


class Pyttsx3TextToSpeech(TextToSpeechProvider):
    """Text-to-speech provider utilizing the local pyttsx3 engine.

    The speech engine is initialized lazily so importing or instantiating this class
    does not trigger immediate COM/audio subsystem allocations or audio driver binding.
    """

    PROVIDER_NAME: str = "pyttsx3"
    SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".wav"}

    def __init__(
        self,
        voice: str | None = None,
        rate: int | None = None,
        volume: float | None = None,
        driver_name: str | None = None,
        engine: Any | None = None,
    ) -> None:
        """Initialize Pyttsx3TextToSpeech configuration.

        Args:
            voice: Optional voice identifier or name to select.
            rate: Optional speaking speed in words per minute (e.g. 150-200). Must be positive.
            volume: Optional speech volume between 0.0 and 1.0.
            driver_name: Optional driver name to pass to pyttsx3.init (e.g. 'sapi5').
            engine: Optional pre-initialized or mock pyttsx3 engine for testing/dependency injection.

        Raises:
            ValueError: If voice, rate, or volume fail validation.
        """
        if voice is not None:
            if not isinstance(voice, str) or not voice.strip():
                raise ValueError("voice must be a non-empty string when provided.")
            voice = voice.strip()

        if rate is not None:
            if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
                raise ValueError("rate must be a positive integer.")

        if volume is not None:
            if (
                not isinstance(volume, (int, float))
                or isinstance(volume, bool)
                or volume < 0.0
                or volume > 1.0
            ):
                raise ValueError("volume must be a float between 0.0 and 1.0.")
            volume = float(volume)

        self.voice: str | None = voice
        self.rate: int | None = rate
        self.volume: float | None = volume
        self.driver_name: str | None = driver_name
        self._engine: Any | None = engine

    @property
    def is_engine_initialized(self) -> bool:
        """Check whether the underlying pyttsx3 engine has been initialized."""
        return self._engine is not None

    def load_engine(self) -> Any:
        """Lazily initialize and configure the pyttsx3 engine instance.

        Returns:
            The initialized pyttsx3 engine.

        Raises:
            RuntimeError: If pyttsx3 is not installed or engine initialization fails.
        """
        if self._engine is not None:
            return self._engine

        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "The 'pyttsx3' package is not installed. Please install 'pyttsx3' "
                "(e.g., pip install pyttsx3) to use Pyttsx3TextToSpeech."
            ) from exc

        try:
            engine = (
                pyttsx3.init(driverName=self.driver_name)
                if self.driver_name
                else pyttsx3.init()
            )
            if self.rate is not None:
                engine.setProperty("rate", self.rate)
            if self.volume is not None:
                engine.setProperty("volume", self.volume)
            if self.voice is not None:
                engine.setProperty("voice", self.voice)

            self._engine = engine
            return self._engine
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize pyttsx3 engine: {exc}") from exc

    def validate_text(self, text: str) -> str:
        """Validate that the text to synthesize is a non-empty string.

        Args:
            text: Text prompt to synthesize.

        Returns:
            Stripped non-empty text string.

        Raises:
            ValueError: If text is None, not a string, or whitespace-only.
        """
        if text is None or not isinstance(text, str):
            raise ValueError(
                f"Text must be a string, got {type(text).__name__ if text is not None else 'None'}"
            )

        stripped = text.strip()
        if not stripped:
            raise ValueError("Text cannot be empty or whitespace only.")

        return stripped

    def validate_output_path(self, output_path: str | Path) -> Path:
        """Validate destination audio path for speech synthesis.

        Args:
            output_path: Target audio file path (str or pathlib.Path).

        Returns:
            Validated resolved pathlib.Path.

        Raises:
            ValueError: If output_path is empty, wrong type, a directory, or not a .wav file.
        """
        if output_path is None or not isinstance(output_path, (str, Path)):
            raise ValueError(
                f"Output path must be a string or Path, got {type(output_path).__name__ if output_path is not None else 'None'}"
            )

        path_str = str(output_path).strip()
        if not path_str:
            raise ValueError("Output path cannot be empty.")

        path = Path(path_str)
        if path.is_dir():
            raise ValueError(
                f"Output path points to a directory, not a file: {path_str}"
            )

        if path.suffix.lower() not in self.SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported output audio format '{path.suffix}'. "
                f"Supported formats: {sorted(self.SUPPORTED_AUDIO_EXTENSIONS)}"
            )

        return path

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        rate: int | None = None,
        volume: float | None = None,
        voice: str | None = None,
        **kwargs: Any,
    ) -> SpeechSynthesisResult:
        """Synthesize text into speech and save the audio to a .wav file.

        Args:
            text: Non-empty text to synthesize.
            output_path: Destination path for the generated .wav file.
            rate: Optional rate override for this synthesis operation.
            volume: Optional volume override for this synthesis operation.
            voice: Optional voice override for this synthesis operation.
            **kwargs: Extra arguments forwarded to engine properties if needed.

        Returns:
            SpeechSynthesisResult with text, audio_path, provider, and voice.

        Raises:
            ValueError: If text is empty or output_path is invalid.
            RuntimeError: If pyttsx3 is not installed or synthesis fails.
        """
        validated_text = self.validate_text(text)
        validated_path = self.validate_output_path(output_path)
        engine = self.load_engine()

        # Validate and apply overrides if specified
        active_rate = rate if rate is not None else self.rate
        active_volume = volume if volume is not None else self.volume
        active_voice = voice if voice is not None else self.voice

        try:
            if active_rate is not None:
                engine.setProperty("rate", active_rate)
            if active_volume is not None:
                engine.setProperty("volume", active_volume)
            if active_voice is not None:
                engine.setProperty("voice", active_voice)

            # Ensure parent directory exists
            if not validated_path.parent.exists():
                validated_path.parent.mkdir(parents=True, exist_ok=True)

            # Synthesize and write audio
            engine.save_to_file(validated_text, str(validated_path))
            engine.runAndWait()
        except Exception as exc:
            raise RuntimeError(f"Speech synthesis failed: {exc}") from exc

        return SpeechSynthesisResult(
            text=validated_text,
            audio_path=str(validated_path),
            provider=self.PROVIDER_NAME,
            voice=active_voice,
        )
