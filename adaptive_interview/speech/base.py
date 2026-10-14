"""Base interfaces for Speech-to-Text and Text-to-Speech providers.

Defines the abstract interfaces that all STT and TTS service implementations
must adhere to in the Adaptive Interview Engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adaptive_interview.speech.models import (
    SpeechSynthesisResult,
    TranscriptionResult,
)


class SpeechToTextProvider(ABC):
    """Abstract base class defining the speech-to-text provider interface."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe an audio file into text.

        Args:
            audio_path: Path to the target audio file (str or pathlib.Path).

        Returns:
            A TranscriptionResult containing the transcribed text and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the audio path is invalid or corrupted.
            RuntimeError: If transcription fails or backend is unavailable.
        """
        pass


class TextToSpeechProvider(ABC):
    """Abstract base class defining the text-to-speech provider interface."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        **kwargs: Any,
    ) -> SpeechSynthesisResult:
        """Synthesize text into speech and save the audio to an output file.

        Args:
            text: Non-empty text string to synthesize.
            output_path: Target path for the generated audio file (.wav).
            **kwargs: Provider-specific configuration options (e.g. rate, volume, voice).

        Returns:
            A SpeechSynthesisResult containing synthesized text, output path, and metadata.

        Raises:
            ValueError: If text is empty or output_path is invalid.
            RuntimeError: If speech synthesis fails or provider is unavailable.
        """
        pass
