"""Speech processing package for the Adaptive Interview Engine.

Provides models and provider implementations for:
- Speech-to-Text (STT) transcription via WhisperSpeechToText.
- Text-to-Speech (TTS) speech synthesis via Pyttsx3TextToSpeech.
- Speech Analytics for transcript evaluation via SpeechAnalytics and analyze_speech.
"""

from __future__ import annotations

from adaptive_interview.speech.analytics import (
    SpeechAnalytics,
    SpeechAnalyzer,
    analyze_speech,
)
from adaptive_interview.speech.base import (
    SpeechToTextProvider,
    TextToSpeechProvider,
)
from adaptive_interview.speech.models import (
    SpeechAnalyticsResult,
    SpeechSynthesisResult,
    TranscriptionResult,
    VoiceAnswerResult,
)
from adaptive_interview.speech.pyttsx_tts import Pyttsx3TextToSpeech
from adaptive_interview.speech.whisper_stt import WhisperSpeechToText

# Helpful aliases
PyttsxTextToSpeech = Pyttsx3TextToSpeech

__all__ = [
    "SpeechAnalytics",
    "SpeechAnalyticsResult",
    "SpeechAnalyzer",
    "SpeechToTextProvider",
    "SpeechSynthesisResult",
    "TextToSpeechProvider",
    "TranscriptionResult",
    "VoiceAnswerResult",
    "Pyttsx3TextToSpeech",
    "PyttsxTextToSpeech",
    "WhisperSpeechToText",
    "analyze_speech",
]
