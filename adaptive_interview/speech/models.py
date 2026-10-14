"""Data models for Speech-to-Text processing.

This module defines data structures representing speech recognition outcomes,
including transcribed text, audio metadata, language detection, and provider attribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_interview.models import AdaptiveResult


@dataclass
class TranscriptionResult:
    """Represents the structured outcome of a speech-to-text transcription.

    Attributes:
        text: The transcribed speech text.
        language: Detected or specified language code (e.g., 'en', 'es') or None.
        duration_seconds: Audio length in seconds, if available or computable.
        provider: Identifier of the STT provider used (defaults to 'whisper').
    """

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    provider: str = "whisper"

    def __post_init__(self) -> None:
        """Validate and normalize transcription result attributes.

        Raises:
            ValueError: If text, provider, language, or duration_seconds are invalid.
        """
        if self.text is None or not isinstance(self.text, str):
            raise ValueError(
                f"Transcription text must be a string, got {type(self.text).__name__ if self.text is not None else 'None'}"
            )

        if not self.provider or not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")

        self.text = self.text.strip()
        self.provider = self.provider.strip()

        if self.language is not None:
            if not isinstance(self.language, str) or not self.language.strip():
                raise ValueError("language must be a non-empty string when provided.")
            self.language = self.language.strip()

        if self.duration_seconds is not None:
            if (
                not isinstance(self.duration_seconds, (int, float))
                or isinstance(self.duration_seconds, bool)
                or self.duration_seconds < 0
            ):
                raise ValueError("duration_seconds must be a non-negative number.")
            self.duration_seconds = float(self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert the transcription result to a serializable dictionary.

        Returns:
            Dictionary containing transcription attributes.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionResult:
        """Create a TranscriptionResult instance from a dictionary.

        Args:
            data: Dictionary containing transcription fields.

        Returns:
            A new TranscriptionResult instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data).__name__}")
        if "text" not in data:
            raise KeyError("Missing required key 'text' in data dictionary.")

        return cls(
            text=data["text"],
            language=data.get("language"),
            duration_seconds=data.get("duration_seconds"),
            provider=data.get("provider", "whisper"),
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. result['text']).

        Args:
            item: Attribute name to access.

        Returns:
            The value of the requested attribute.

        Raises:
            KeyError: If the attribute does not exist.
        """
        if hasattr(self, item) and not item.startswith("_"):
            return getattr(self, item)
        raise KeyError(item)


@dataclass
class SpeechSynthesisResult:
    """Represents the structured outcome of a text-to-speech synthesis operation.

    Attributes:
        text: The synthesized speech text.
        audio_path: File path where the synthesized audio is saved.
        provider: Identifier of the TTS provider used (defaults to 'pyttsx3').
        duration_seconds: Optional audio length in seconds, if available or computable.
        voice: Optional voice ID or name used during synthesis.
    """

    text: str
    audio_path: str
    provider: str = "pyttsx3"
    duration_seconds: float | None = None
    voice: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize synthesis result attributes.

        Raises:
            ValueError: If text, audio_path, provider, duration_seconds, or voice are invalid.
        """
        if self.text is None or not isinstance(self.text, str):
            raise ValueError(
                f"Synthesized text must be a string, got {type(self.text).__name__ if self.text is not None else 'None'}"
            )

        if not self.text.strip():
            raise ValueError("Synthesized text must be a non-empty string.")

        if self.audio_path is None or not isinstance(self.audio_path, str) or not str(self.audio_path).strip():
            raise ValueError("audio_path must be a non-empty string or path.")

        if not self.provider or not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty string.")

        self.text = self.text.strip()
        self.audio_path = str(self.audio_path).strip()
        self.provider = self.provider.strip()

        if self.voice is not None:
            if not isinstance(self.voice, str) or not self.voice.strip():
                raise ValueError("voice must be a non-empty string when provided.")
            self.voice = self.voice.strip()

        if self.duration_seconds is not None:
            if (
                not isinstance(self.duration_seconds, (int, float))
                or isinstance(self.duration_seconds, bool)
                or self.duration_seconds < 0
            ):
                raise ValueError("duration_seconds must be a non-negative number.")
            self.duration_seconds = float(self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Convert the synthesis result to a serializable dictionary.

        Returns:
            Dictionary containing speech synthesis attributes.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechSynthesisResult:
        """Create a SpeechSynthesisResult instance from a dictionary.

        Args:
            data: Dictionary containing synthesis fields.

        Returns:
            A new SpeechSynthesisResult instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data).__name__}")
        if "text" not in data:
            raise KeyError("Missing required key 'text' in data dictionary.")
        if "audio_path" not in data:
            raise KeyError("Missing required key 'audio_path' in data dictionary.")

        return cls(
            text=data["text"],
            audio_path=str(data["audio_path"]),
            provider=data.get("provider", "pyttsx3"),
            duration_seconds=data.get("duration_seconds"),
            voice=data.get("voice"),
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. result['audio_path']).

        Args:
            item: Attribute name to access.

        Returns:
            The value of the requested attribute.

        Raises:
            KeyError: If the attribute does not exist.
        """
        if hasattr(self, item) and not item.startswith("_"):
            return getattr(self, item)
        raise KeyError(item)


@dataclass
class SpeechAnalyticsResult:
    """Represents the structured outcome of speech transcript analysis.

    Attributes:
        word_count: Total number of recognized words in the transcript.
        filler_words_count: Total count of detected filler words and phrases.
        filler_words: Mapping of detected filler words/phrases to occurrence counts.
        duration_seconds: Optional audio duration in seconds.
        words_per_minute: Speaking rate (WPM) when duration is provided.
        filler_ratio: Ratio of filler words to total words (0.0 to 1.0).
        pace_status: Speaking pace category ('slow', 'moderate', 'fast', or 'unknown').
        fluency_score: Composite score estimating text-level fluency (0.0 to 100.0).
        fluency_indicators: Detailed dictionary of computed fluency metrics.
    """

    word_count: int
    filler_words_count: int
    filler_words: dict[str, int]
    duration_seconds: float | None = None
    words_per_minute: float | None = None
    filler_ratio: float = 0.0
    pace_status: str | None = None
    fluency_score: float | None = None
    fluency_indicators: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize analytics result attributes.

        Raises:
            ValueError: If word_count, filler_words_count, filler_ratio, duration, WPM, or score are invalid.
        """
        if not isinstance(self.word_count, int) or isinstance(self.word_count, bool) or self.word_count < 0:
            raise ValueError(f"word_count must be a non-negative integer, got {self.word_count}")

        if not isinstance(self.filler_words_count, int) or isinstance(self.filler_words_count, bool) or self.filler_words_count < 0:
            raise ValueError(f"filler_words_count must be a non-negative integer, got {self.filler_words_count}")

        if not isinstance(self.filler_words, dict):
            raise ValueError(f"filler_words must be a dictionary, got {type(self.filler_words).__name__}")

        if not isinstance(self.filler_ratio, (int, float)) or isinstance(self.filler_ratio, bool) or self.filler_ratio < 0.0:
            raise ValueError(f"filler_ratio must be a non-negative float, got {self.filler_ratio}")
        self.filler_ratio = float(self.filler_ratio)

        if self.duration_seconds is not None:
            if not isinstance(self.duration_seconds, (int, float)) or isinstance(self.duration_seconds, bool) or self.duration_seconds <= 0:
                raise ValueError("duration_seconds must be a positive number (> 0).")
            self.duration_seconds = float(self.duration_seconds)

        if self.words_per_minute is not None:
            if not isinstance(self.words_per_minute, (int, float)) or isinstance(self.words_per_minute, bool) or self.words_per_minute < 0:
                raise ValueError("words_per_minute must be a non-negative number.")
            self.words_per_minute = float(self.words_per_minute)

        if self.fluency_score is not None:
            if not isinstance(self.fluency_score, (int, float)) or isinstance(self.fluency_score, bool) or self.fluency_score < 0.0 or self.fluency_score > 100.0:
                raise ValueError("fluency_score must be a float between 0.0 and 100.0.")
            self.fluency_score = float(self.fluency_score)

    @property
    def filler_count(self) -> int:
        """Convenience alias for filler_words_count."""
        return self.filler_words_count

    @property
    def wpm(self) -> float | None:
        """Convenience alias for words_per_minute."""
        return self.words_per_minute

    def to_dict(self) -> dict[str, Any]:
        """Convert the speech analytics result to a serializable dictionary.

        Returns:
            Dictionary containing analytics attributes.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechAnalyticsResult:
        """Create a SpeechAnalyticsResult instance from a dictionary.

        Args:
            data: Dictionary containing speech analytics fields.

        Returns:
            A new SpeechAnalyticsResult instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data is not a dictionary.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data).__name__}")

        required_keys = ["word_count", "filler_words_count", "filler_words"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in data dictionary.")

        return cls(
            word_count=int(data["word_count"]),
            filler_words_count=int(data["filler_words_count"]),
            filler_words=dict(data["filler_words"]),
            duration_seconds=data.get("duration_seconds"),
            words_per_minute=data.get("words_per_minute"),
            filler_ratio=float(data.get("filler_ratio", 0.0)),
            pace_status=data.get("pace_status"),
            fluency_score=data.get("fluency_score"),
            fluency_indicators=dict(data.get("fluency_indicators", {})),
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. result['word_count']).

        Args:
            item: Attribute name to access.

        Returns:
            The value of the requested attribute.

        Raises:
            KeyError: If the attribute does not exist.
        """
        if hasattr(self, item) and not item.startswith("_"):
            return getattr(self, item)
        raise KeyError(item)


@dataclass
class VoiceAnswerResult:
    """Represents the structured outcome of processing a candidate's voice answer.

    Attributes:
        audio_path: File path of the processed voice recording.
        transcription: Speech-to-text transcription result.
        analytics: Speech analytics evaluation metrics (pace, fillers, fluency).
        adaptive_result: Optional adaptive performance result if an evaluation score was provided.
    """

    audio_path: str
    transcription: TranscriptionResult
    analytics: SpeechAnalyticsResult
    adaptive_result: AdaptiveResult | None = None

    def __post_init__(self) -> None:
        """Validate voice answer result attributes.

        Raises:
            ValueError: If audio_path is empty/invalid, or nested objects are of wrong type.
        """
        if self.audio_path is None or not isinstance(self.audio_path, (str, Path)):
            raise ValueError(
                f"audio_path must be a non-empty string or Path, got {type(self.audio_path).__name__ if self.audio_path is not None else 'None'}"
            )

        audio_str = str(self.audio_path).strip()
        if not audio_str:
            raise ValueError("audio_path cannot be empty.")
        self.audio_path = audio_str

        if not isinstance(self.transcription, TranscriptionResult):
            raise ValueError(
                f"transcription must be a TranscriptionResult instance, got {type(self.transcription).__name__}"
            )

        if not isinstance(self.analytics, SpeechAnalyticsResult):
            raise ValueError(
                f"analytics must be a SpeechAnalyticsResult instance, got {type(self.analytics).__name__}"
            )

        if self.adaptive_result is not None:
            from adaptive_interview.models import AdaptiveResult

            if not isinstance(self.adaptive_result, AdaptiveResult):
                raise ValueError(
                    f"adaptive_result must be an AdaptiveResult instance or None, got {type(self.adaptive_result).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert the voice answer result to a serializable dictionary.

        Returns:
            Dictionary containing audio path, transcription, analytics, and adaptive result.
        """
        return {
            "audio_path": self.audio_path,
            "transcription": self.transcription.to_dict(),
            "analytics": self.analytics.to_dict(),
            "adaptive_result": (
                self.adaptive_result.to_dict() if self.adaptive_result is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceAnswerResult:
        """Create a VoiceAnswerResult instance from a dictionary.

        Args:
            data: Dictionary containing voice answer fields.

        Returns:
            A new VoiceAnswerResult instance.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If data is not a dictionary or contains invalid types.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data).__name__}")
        if "audio_path" not in data:
            raise KeyError("Missing required key 'audio_path' in data dictionary.")
        if "transcription" not in data:
            raise KeyError("Missing required key 'transcription' in data dictionary.")
        if "analytics" not in data:
            raise KeyError("Missing required key 'analytics' in data dictionary.")

        transcription_raw = data["transcription"]
        if isinstance(transcription_raw, dict):
            transcription = TranscriptionResult.from_dict(transcription_raw)
        elif isinstance(transcription_raw, TranscriptionResult):
            transcription = transcription_raw
        else:
            raise ValueError(f"Invalid transcription data type: {type(transcription_raw).__name__}")

        analytics_raw = data["analytics"]
        if isinstance(analytics_raw, dict):
            analytics = SpeechAnalyticsResult.from_dict(analytics_raw)
        elif isinstance(analytics_raw, SpeechAnalyticsResult):
            analytics = analytics_raw
        else:
            raise ValueError(f"Invalid analytics data type: {type(analytics_raw).__name__}")

        adaptive_raw = data.get("adaptive_result")
        adaptive_result = None
        if adaptive_raw is not None:
            from adaptive_interview.models import AdaptiveResult

            if isinstance(adaptive_raw, dict):
                adaptive_result = AdaptiveResult.from_dict(adaptive_raw)
            elif isinstance(adaptive_raw, AdaptiveResult):
                adaptive_result = adaptive_raw
            else:
                raise ValueError(f"Invalid adaptive_result data type: {type(adaptive_raw).__name__}")

        return cls(
            audio_path=data["audio_path"],
            transcription=transcription,
            analytics=analytics,
            adaptive_result=adaptive_result,
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. result['transcription']).

        Args:
            item: Attribute name to access.

        Returns:
            The value of the requested attribute.

        Raises:
            KeyError: If the attribute does not exist.
        """
        if hasattr(self, item) and not item.startswith("_"):
            return getattr(self, item)
        raise KeyError(item)
