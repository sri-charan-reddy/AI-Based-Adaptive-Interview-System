"""Speech analytics module for the Adaptive Interview Engine.

Provides tools for evaluating speech transcripts, including total word count,
filler word frequency detection, speaking rate (WPM), and text-derived fluency indicators.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from adaptive_interview.speech.models import SpeechAnalyticsResult


class SpeechAnalytics:
    """Analyzer for speech transcripts and verbal candidate responses.

    Evaluates word counts, filler words, speaking rate, and text-based fluency indicators.
    Note: Text-only analysis evaluates textual characteristics and cannot measure
    acoustic qualities such as actual pause durations, pitch, volume, or pronunciation.
    """

    DEFAULT_FILLER_WORDS: list[str] = [
        "um",
        "uh",
        "erm",
        "like",
        "you know",
    ]

    # Conversational speaking pace thresholds (Words Per Minute)
    SLOW_PACE_THRESHOLD: float = 110.0
    FAST_PACE_THRESHOLD: float = 165.0

    def __init__(
        self,
        filler_words: Sequence[str] | None = None,
    ) -> None:
        """Initialize SpeechAnalytics with optional custom filler words.

        Args:
            filler_words: Optional sequence of filler words or phrases to detect.
                          Defaults to DEFAULT_FILLER_WORDS.

        Raises:
            ValueError: If custom filler_words contains invalid non-string or empty items.
        """
        if filler_words is not None:
            if not isinstance(filler_words, (list, tuple, set)):
                raise ValueError("filler_words must be a sequence of strings.")
            validated = []
            for item in filler_words:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"Filler word items must be non-empty strings, got {item!r}")
                validated.append(item.strip().lower())
            self.filler_words: list[str] = validated
        else:
            self.filler_words = list(self.DEFAULT_FILLER_WORDS)

    @staticmethod
    def _extract_words(text: str) -> list[str]:
        """Extract word tokens from text, supporting contractions and hyphens.

        Args:
            text: Input text string.

        Returns:
            List of extracted word tokens containing alphanumeric characters.
        """
        raw_tokens = re.findall(r"\b[\w'-]+\b", text)
        return [t for t in raw_tokens if any(c.isalnum() for c in t)]

    def analyze(
        self,
        text: str,
        duration_seconds: float | None = None,
    ) -> SpeechAnalyticsResult:
        """Analyze transcription text for word count, fillers, pace, and fluency.

        Args:
            text: Non-empty transcription text string.
            duration_seconds: Optional audio duration in seconds. Must be > 0 if provided.

        Returns:
            A SpeechAnalyticsResult with calculated metrics.

        Raises:
            ValueError: If text is empty/invalid, or duration_seconds is non-positive.
        """
        # Validate text input
        if text is None or not isinstance(text, str):
            raise ValueError(
                f"Transcription text must be a string, got {type(text).__name__ if text is not None else 'None'}"
            )

        stripped_text = text.strip()
        if not stripped_text:
            raise ValueError("Transcription text cannot be empty or whitespace only.")

        words = self._extract_words(stripped_text)
        word_count = len(words)
        if word_count == 0:
            raise ValueError("Transcription text contains no recognizable words.")

        # Validate duration input
        validated_duration: float | None = None
        if duration_seconds is not None:
            if not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool):
                raise ValueError(
                    f"duration_seconds must be a number, got {type(duration_seconds).__name__}"
                )
            if duration_seconds <= 0:
                raise ValueError(
                    f"duration_seconds must be a positive number (> 0), got {duration_seconds}"
                )
            validated_duration = float(duration_seconds)

        # Detect filler words and phrases (whole word/phrase matching without substrings)
        detected_fillers: dict[str, int] = {}
        total_fillers = 0

        for filler in self.filler_words:
            tokens = filler.split()
            escaped_tokens = [re.escape(t) for t in tokens]
            # Use negative lookaround to prevent matching substrings within longer words or hyphenated words
            pattern = rf"(?<![\w-]){'[\s,]+'.join(escaped_tokens)}(?![\w-])"
            matches = re.findall(pattern, stripped_text, flags=re.IGNORECASE)
            count = len(matches)
            if count > 0:
                detected_fillers[filler] = count
                total_fillers += count

        # Calculate filler ratio
        filler_ratio = round(total_fillers / word_count, 4)

        # Determine filler density status
        if total_fillers == 0:
            filler_density_status = "none"
        elif filler_ratio <= 0.03:
            filler_density_status = "low"
        elif filler_ratio <= 0.08:
            filler_density_status = "moderate"
        else:
            filler_density_status = "high"

        # Calculate speaking rate and pace status if duration is provided
        words_per_minute: float | None = None
        pace_status: str = "unknown"
        pace_deduction = 0.0

        if validated_duration is not None:
            words_per_minute = round((word_count / validated_duration) * 60.0, 2)
            if words_per_minute < self.SLOW_PACE_THRESHOLD:
                pace_status = "slow"
                pace_deduction = min((self.SLOW_PACE_THRESHOLD - words_per_minute) * 0.4, 25.0)
            elif words_per_minute > self.FAST_PACE_THRESHOLD:
                pace_status = "fast"
                pace_deduction = min((words_per_minute - self.FAST_PACE_THRESHOLD) * 0.4, 25.0)
            else:
                pace_status = "moderate"
                pace_deduction = 0.0

        # Calculate text-derived fluency score (0.0 to 100.0)
        filler_deduction = min(filler_ratio * 200.0, 45.0)
        fluency_score = round(max(0.0, min(100.0, 100.0 - filler_deduction - pace_deduction)), 1)

        fluency_indicators: dict[str, Any] = {
            "filler_ratio": filler_ratio,
            "filler_density_status": filler_density_status,
            "pace_status": pace_status,
            "fluency_score": fluency_score,
        }
        if words_per_minute is not None:
            fluency_indicators["words_per_minute"] = words_per_minute
        if validated_duration is not None:
            fluency_indicators["duration_seconds"] = validated_duration

        return SpeechAnalyticsResult(
            word_count=word_count,
            filler_words_count=total_fillers,
            filler_words=detected_fillers,
            duration_seconds=validated_duration,
            words_per_minute=words_per_minute,
            filler_ratio=filler_ratio,
            pace_status=pace_status,
            fluency_score=fluency_score,
            fluency_indicators=fluency_indicators,
        )


# Convenience alias
SpeechAnalyzer = SpeechAnalytics


def analyze_speech(
    text: str,
    duration_seconds: float | None = None,
    filler_words: Sequence[str] | None = None,
) -> SpeechAnalyticsResult:
    """Convenience helper function to analyze speech transcript text.

    Args:
        text: Non-empty transcription text string.
        duration_seconds: Optional audio duration in seconds.
        filler_words: Optional custom sequence of filler words to detect.

    Returns:
        A SpeechAnalyticsResult with calculated metrics.
    """
    analyzer = SpeechAnalytics(filler_words=filler_words)
    return analyzer.analyze(text=text, duration_seconds=duration_seconds)
