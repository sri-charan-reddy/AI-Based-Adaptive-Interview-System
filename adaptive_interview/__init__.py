"""Adaptive Interview Module (Member 2).

Provides the foundation for tracking interview state, recording interactions,
managing session lifecycle, dynamically adapting question difficulty, processing answer
evaluations, and selecting targeted questions and follow-ups.
"""

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import (
    AdaptiveResult,
    Difficulty,
    InterviewState,
    Question,
    SessionStatus,
    VoiceAnswerResult,
)
from adaptive_interview.question_bank import DEFAULT_QUESTION_BANK, get_default_question_bank
from adaptive_interview.selector import QuestionSelector
from adaptive_interview.speech import (
    Pyttsx3TextToSpeech,
    PyttsxTextToSpeech,
    SpeechAnalytics,
    SpeechAnalyticsResult,
    SpeechAnalyzer,
    SpeechSynthesisResult,
    SpeechToTextProvider,
    TextToSpeechProvider,
    TranscriptionResult,
    WhisperSpeechToText,
    analyze_speech,
)

__all__ = [
    "AdaptiveInterviewEngine",
    "AdaptiveResult",
    "Difficulty",
    "InterviewState",
    "Question",
    "QuestionSelector",
    "SessionStatus",
    "VoiceAnswerResult",
    "SpeechAnalytics",
    "SpeechAnalyticsResult",
    "SpeechAnalyzer",
    "SpeechSynthesisResult",
    "SpeechToTextProvider",
    "TextToSpeechProvider",
    "TranscriptionResult",
    "Pyttsx3TextToSpeech",
    "PyttsxTextToSpeech",
    "WhisperSpeechToText",
    "analyze_speech",
    "DEFAULT_QUESTION_BANK",
    "get_default_question_bank",
]
