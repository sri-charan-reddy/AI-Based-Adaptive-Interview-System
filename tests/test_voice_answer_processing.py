"""Unit tests for Voice Answer Processing pipeline (Task 8 / Task Integration).

Tests:
- VoiceAnswerResult data model validation, dictionary access, and serialization.
- AdaptiveInterviewEngine dependency injection of SpeechToTextProvider and SpeechAnalytics.
- process_voice_answer() end-to-end workflow (transcription -> analytics -> answer recording -> adaptation).
- Session lifecycle validation, error handling, audio path validation, and score validation.
- All tests use mocked STT providers and analytics to avoid downloading Whisper models or requiring live audio files.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import (
    AdaptiveResult,
    Difficulty,
    InterviewState,
    Question,
    SessionStatus,
    VoiceAnswerResult,
)
from adaptive_interview.speech.analytics import SpeechAnalytics
from adaptive_interview.speech.base import SpeechToTextProvider
from adaptive_interview.speech.models import (
    SpeechAnalyticsResult,
    TranscriptionResult,
)


class TestVoiceAnswerResultModel(unittest.TestCase):
    """Test suite for the VoiceAnswerResult dataclass."""

    def setUp(self) -> None:
        self.transcription = TranscriptionResult(
            text="I use dictionaries for fast lookups in Python.",
            language="en",
            duration_seconds=5.0,
            provider="whisper",
        )
        self.analytics = SpeechAnalyticsResult(
            word_count=8,
            filler_words_count=0,
            filler_words={},
            duration_seconds=5.0,
            words_per_minute=96.0,
            filler_ratio=0.0,
            pace_status="slow",
            fluency_score=94.4,
            fluency_indicators={"filler_ratio": 0.0, "pace_status": "slow"},
        )
        self.adaptive = AdaptiveResult(
            score=85.0,
            topic="Data Structures",
            difficulty="hard",
            is_weak_topic=False,
            should_follow_up=False,
        )

    def test_voice_answer_result_creation_without_adaptive_result(self) -> None:
        """Test creating VoiceAnswerResult without optional adaptive_result."""
        result = VoiceAnswerResult(
            audio_path="test_audio.wav",
            transcription=self.transcription,
            analytics=self.analytics,
        )
        self.assertEqual(result.audio_path, "test_audio.wav")
        self.assertEqual(result.transcription.text, self.transcription.text)
        self.assertEqual(result.analytics.word_count, 8)
        self.assertIsNone(result.adaptive_result)

    def test_voice_answer_result_creation_with_adaptive_result(self) -> None:
        """Test creating VoiceAnswerResult with a valid adaptive_result."""
        result = VoiceAnswerResult(
            audio_path="test_audio.wav",
            transcription=self.transcription,
            analytics=self.analytics,
            adaptive_result=self.adaptive,
        )
        self.assertIsNotNone(result.adaptive_result)
        self.assertEqual(result.adaptive_result.difficulty, "hard")
        self.assertEqual(result.adaptive_result.score, 85.0)

    def test_voice_answer_result_pathlib_audio_path_accepted(self) -> None:
        """Test that pathlib.Path is accepted and normalized to a string."""
        path_obj = Path("recordings/candidate_reply.wav")
        result = VoiceAnswerResult(
            audio_path=path_obj,
            transcription=self.transcription,
            analytics=self.analytics,
        )
        self.assertEqual(result.audio_path, str(path_obj))

    def test_voice_answer_result_subscript_access(self) -> None:
        """Test dictionary-style subscripting on VoiceAnswerResult."""
        result = VoiceAnswerResult(
            audio_path="audio.wav",
            transcription=self.transcription,
            analytics=self.analytics,
            adaptive_result=self.adaptive,
        )
        self.assertEqual(result["audio_path"], "audio.wav")
        self.assertIs(result["transcription"], self.transcription)
        self.assertIs(result["analytics"], self.analytics)
        self.assertIs(result["adaptive_result"], self.adaptive)

    def test_voice_answer_result_subscript_key_error(self) -> None:
        """Test KeyError on invalid subscript key."""
        result = VoiceAnswerResult(
            audio_path="audio.wav",
            transcription=self.transcription,
            analytics=self.analytics,
        )
        with self.assertRaises(KeyError):
            _ = result["non_existent_key"]

    def test_voice_answer_result_to_dict_and_from_dict_roundtrip(self) -> None:
        """Test complete serialization and deserialization roundtrip."""
        result = VoiceAnswerResult(
            audio_path="path/to/answer.wav",
            transcription=self.transcription,
            analytics=self.analytics,
            adaptive_result=self.adaptive,
        )
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["audio_path"], "path/to/answer.wav")
        self.assertIsInstance(data["transcription"], dict)
        self.assertIsInstance(data["analytics"], dict)
        self.assertIsInstance(data["adaptive_result"], dict)

        reconstructed = VoiceAnswerResult.from_dict(data)
        self.assertEqual(reconstructed.audio_path, result.audio_path)
        self.assertEqual(reconstructed.transcription.text, result.transcription.text)
        self.assertEqual(reconstructed.analytics.word_count, result.analytics.word_count)
        self.assertIsNotNone(reconstructed.adaptive_result)
        self.assertEqual(reconstructed.adaptive_result.score, result.adaptive_result.score)

    def test_voice_answer_result_from_dict_without_adaptive_result(self) -> None:
        """Test from_dict when adaptive_result is None or omitted."""
        data = {
            "audio_path": "sample.wav",
            "transcription": self.transcription.to_dict(),
            "analytics": self.analytics.to_dict(),
            "adaptive_result": None,
        }
        result = VoiceAnswerResult.from_dict(data)
        self.assertIsNone(result.adaptive_result)

    def test_voice_answer_result_from_dict_missing_keys(self) -> None:
        """Test from_dict raises KeyError on missing required keys."""
        with self.assertRaises(KeyError):
            VoiceAnswerResult.from_dict({"audio_path": "x.wav"})

        with self.assertRaises(ValueError):
            VoiceAnswerResult.from_dict("not-a-dict")  # type: ignore

    def test_voice_answer_result_validation_errors(self) -> None:
        """Test validation rules in VoiceAnswerResult.__post_init__."""
        # Invalid audio_path
        with self.assertRaises(ValueError):
            VoiceAnswerResult(
                audio_path="",
                transcription=self.transcription,
                analytics=self.analytics,
            )
        with self.assertRaises(ValueError):
            VoiceAnswerResult(
                audio_path=None,  # type: ignore
                transcription=self.transcription,
                analytics=self.analytics,
            )

        # Invalid transcription
        with self.assertRaises(ValueError):
            VoiceAnswerResult(
                audio_path="test.wav",
                transcription="not-a-transcription",  # type: ignore
                analytics=self.analytics,
            )

        # Invalid analytics
        with self.assertRaises(ValueError):
            VoiceAnswerResult(
                audio_path="test.wav",
                transcription=self.transcription,
                analytics="not-analytics",  # type: ignore
            )

        # Invalid adaptive_result
        with self.assertRaises(ValueError):
            VoiceAnswerResult(
                audio_path="test.wav",
                transcription=self.transcription,
                analytics=self.analytics,
                adaptive_result="not-an-adaptive-result",  # type: ignore
            )


class TestEngineDependencyInjection(unittest.TestCase):
    """Test dependency injection of STT provider and analytics in AdaptiveInterviewEngine."""

    def test_engine_init_without_providers_does_not_load_whisper_models(self) -> None:
        """Verify engine instantiation does not eagerly instantiate or load Whisper models."""
        engine = AdaptiveInterviewEngine()
        self.assertIsNone(engine._stt_provider)
        self.assertIsNone(engine._speech_analytics)

    def test_engine_init_with_injected_providers(self) -> None:
        """Verify custom STT provider and analytics are stored during initialization."""
        mock_stt = Mock(spec=SpeechToTextProvider)
        mock_analytics = Mock(spec=SpeechAnalytics)

        engine = AdaptiveInterviewEngine(
            stt_provider=mock_stt,
            speech_analytics=mock_analytics,
        )
        self.assertIs(engine.stt_provider, mock_stt)
        self.assertIs(engine.speech_analytics, mock_analytics)

    def test_engine_property_setters(self) -> None:
        """Verify custom STT provider and analytics can be updated via property setters."""
        engine = AdaptiveInterviewEngine()
        mock_stt = Mock(spec=SpeechToTextProvider)
        mock_analytics = Mock(spec=SpeechAnalytics)

        engine.stt_provider = mock_stt
        engine.speech_analytics = mock_analytics

        self.assertIs(engine.stt_provider, mock_stt)
        self.assertIs(engine.speech_analytics, mock_analytics)


class TestProcessVoiceAnswerWorkflow(unittest.TestCase):
    """Test suite for the process_voice_answer() end-to-end pipeline."""

    def setUp(self) -> None:
        self.mock_stt = Mock(spec=SpeechToTextProvider)
        self.mock_analytics = SpeechAnalytics()  # Real analytics with standard filler logic
        self.engine = AdaptiveInterviewEngine(
            stt_provider=self.mock_stt,
            speech_analytics=self.mock_analytics,
        )
        self.state = self.engine.start_interview(
            role="Backend Engineer",
            initial_difficulty="medium",
            topics=["Python", "Data Structures"],
        )

    def test_process_voice_answer_without_score(self) -> None:
        """Test voice answer processing when no evaluation score is provided."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Lists in Python are dynamic arrays, while tuples are immutable sequences.",
            language="en",
            duration_seconds=6.0,
            provider="whisper",
        )

        question_text = "What is the difference between a list and a tuple?"
        result = self.engine.process_voice_answer(
            question=question_text,
            audio_path="answer_01.wav",
        )

        # Verify transcription called with audio path
        self.mock_stt.transcribe.assert_called_once_with("answer_01.wav")

        # Verify VoiceAnswerResult attributes
        self.assertIsInstance(result, VoiceAnswerResult)
        self.assertEqual(result.audio_path, "answer_01.wav")
        self.assertEqual(result.transcription.text, "Lists in Python are dynamic arrays, while tuples are immutable sequences.")
        self.assertEqual(result.analytics.word_count, 11)
        self.assertEqual(result.analytics.filler_words_count, 0)
        self.assertIsNone(result.adaptive_result)

        # Verify state mutation (answer recorded)
        self.assertEqual(len(self.state.answers), 1)
        self.assertIn("Lists in Python are dynamic arrays", self.state.answers[0])
        self.assertIn(question_text, self.state.questions_asked)
        self.assertEqual(self.state.question_number, 1)
        self.assertEqual(len(self.state.scores), 0)

    def test_process_voice_answer_with_poor_score_triggers_adaptation(self) -> None:
        """Test voice answer processing with a poor score (< 50) adapting difficulty down."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Um, I think tuples are, uh, just like lists, you know.",
            language="en",
            duration_seconds=10.0,
            provider="whisper",
        )

        result = self.engine.process_voice_answer(
            question="Explain tuple immutability.",
            audio_path="audio_tuple.wav",
            score=40.0,
            topic="Python",
        )

        # Verify speech analytics computed filler words
        self.assertGreater(result.analytics.filler_words_count, 0)
        self.assertIn("um", result.analytics.filler_words)
        self.assertIn("you know", result.analytics.filler_words)

        # Verify adaptive evaluation outcome
        self.assertIsNotNone(result.adaptive_result)
        self.assertEqual(result.adaptive_result.score, 40.0)
        self.assertEqual(result.adaptive_result.difficulty, "easy")
        self.assertTrue(result.adaptive_result.is_weak_topic)
        self.assertTrue(result.adaptive_result.should_follow_up)

        # Verify state reflects adaptation
        self.assertEqual(self.state.difficulty, "easy")
        self.assertIn("Python", self.state.weak_topics)
        self.assertEqual(self.state.scores, [40.0])

    def test_process_voice_answer_with_high_score_scales_difficulty_up(self) -> None:
        """Test voice answer processing with a high score (>= 80) scaling difficulty up."""
        self.state.weak_topics = ["Python"]  # Existing weak topic

        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Python handles memory management using reference counting and a cyclic garbage collector.",
            language="en",
            duration_seconds=5.0,
            provider="whisper",
        )

        result = self.engine.process_voice_answer(
            question="How does Python manage memory?",
            audio_path="memory_answer.wav",
            score=95.0,
            topic="Python",
        )

        # Adaptive evaluation
        self.assertIsNotNone(result.adaptive_result)
        self.assertEqual(result.adaptive_result.score, 95.0)
        self.assertEqual(result.adaptive_result.difficulty, "hard")
        self.assertFalse(result.adaptive_result.is_weak_topic)
        self.assertFalse(result.adaptive_result.should_follow_up)

        # State updates: weak topic cleared, difficulty increased to hard
        self.assertEqual(self.state.difficulty, "hard")
        self.assertNotIn("Python", self.state.weak_topics)
        self.assertEqual(self.state.scores, [95.0])

    def test_process_voice_answer_with_explicit_state_argument(self) -> None:
        """Test passing an explicit InterviewState to process_voice_answer."""
        other_state = InterviewState(
            role="Data Scientist",
            difficulty=Difficulty.MEDIUM.value,
            status=SessionStatus.IN_PROGRESS.value,
        )

        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Pandas DataFrames provide tabular data structures with labeled axes.",
            language="en",
            duration_seconds=4.0,
        )

        result = self.engine.process_voice_answer(
            question="What is a DataFrame?",
            audio_path="pandas.wav",
            score=82.0,
            topic="Pandas",
            state=other_state,
        )

        self.assertEqual(len(other_state.answers), 1)
        self.assertEqual(other_state.difficulty, "hard")
        self.assertEqual(result.adaptive_result.difficulty, "hard")

        # Verify engine internal state was not touched
        self.assertEqual(len(self.state.answers), 0)

    def test_process_voice_answer_with_runtime_provider_overrides(self) -> None:
        """Test overriding STT and analytics directly in the method call."""
        override_stt = Mock(spec=SpeechToTextProvider)
        override_stt.transcribe.return_value = TranscriptionResult(
            text="Overridden provider response.",
            duration_seconds=2.0,
        )

        override_analytics = Mock(spec=SpeechAnalytics)
        mock_analytics_result = SpeechAnalyticsResult(
            word_count=3,
            filler_words_count=0,
            filler_words={},
            fluency_score=100.0,
        )
        override_analytics.analyze.return_value = mock_analytics_result

        result = self.engine.process_voice_answer(
            question="Override test question?",
            audio_path="override.wav",
            stt_provider=override_stt,
            speech_analytics=override_analytics,
        )

        override_stt.transcribe.assert_called_once_with("override.wav")
        override_analytics.analyze.assert_called_once_with(
            text="Overridden provider response.",
            duration_seconds=2.0,
        )
        self.assertEqual(result.analytics, mock_analytics_result)
        # Verify engine's default providers were not called
        self.mock_stt.transcribe.assert_not_called()

    def test_process_voice_answer_with_pathlib_path(self) -> None:
        """Test passing a pathlib.Path instance as audio_path."""
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Pathlib test successful.",
            duration_seconds=1.5,
        )

        path_obj = Path("recordings") / "sample_answer.wav"
        result = self.engine.process_voice_answer(
            question="Pathlib question?",
            audio_path=path_obj,
        )
        self.assertEqual(result.audio_path, str(path_obj))
        self.mock_stt.transcribe.assert_called_once_with(str(path_obj))


class TestProcessVoiceAnswerValidationAndErrors(unittest.TestCase):
    """Test validation rules and error propagation in process_voice_answer()."""

    def setUp(self) -> None:
        self.mock_stt = Mock(spec=SpeechToTextProvider)
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Valid test transcription.",
            duration_seconds=2.0,
        )
        self.engine = AdaptiveInterviewEngine(stt_provider=self.mock_stt)

    def test_session_not_started_raises_value_error(self) -> None:
        """Test ValueError when interview session is in NOT_STARTED state."""
        state = InterviewState(role="Developer", status=SessionStatus.NOT_STARTED.value)
        with self.assertRaises(ValueError) as ctx:
            self.engine.process_voice_answer(
                question="Question?",
                audio_path="audio.wav",
                state=state,
            )
        self.assertIn("session is not in progress", str(ctx.exception))

    def test_session_completed_raises_value_error(self) -> None:
        """Test ValueError when interview session is in COMPLETED state."""
        state = InterviewState(role="Developer", status=SessionStatus.COMPLETED.value)
        with self.assertRaises(ValueError) as ctx:
            self.engine.process_voice_answer(
                question="Question?",
                audio_path="audio.wav",
                state=state,
            )
        self.assertIn("session is not in progress", str(ctx.exception))

    def test_no_active_state_raises_value_error(self) -> None:
        """Test ValueError when no engine state exists and no state is passed."""
        engine = AdaptiveInterviewEngine(stt_provider=self.mock_stt)
        with self.assertRaises(ValueError) as ctx:
            engine.process_voice_answer(
                question="Question?",
                audio_path="audio.wav",
            )
        self.assertIn("No active interview state found", str(ctx.exception))

    def test_empty_question_raises_value_error(self) -> None:
        """Test ValueError on empty or invalid question argument."""
        self.engine.start_interview("Developer")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="", audio_path="audio.wav")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="   ", audio_path="audio.wav")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question=None, audio_path="audio.wav")  # type: ignore

    def test_invalid_audio_path_raises_value_error(self) -> None:
        """Test ValueError on empty or invalid audio_path argument."""
        self.engine.start_interview("Developer")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="   ")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path=None)  # type: ignore
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path=12345)  # type: ignore

    def test_invalid_score_raises_value_error(self) -> None:
        """Test ValueError when score is outside [0, 100] or non-numeric."""
        self.engine.start_interview("Developer")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="audio.wav", score=-5.0)
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="audio.wav", score=105.0)
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="audio.wav", score="90")  # type: ignore
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Q?", audio_path="audio.wav", score=True)  # type: ignore

    def test_transcription_runtime_error_propagates(self) -> None:
        """Test that STT RuntimeError propagates with clear messaging."""
        self.engine.start_interview("Developer")
        self.mock_stt.transcribe.side_effect = RuntimeError("Whisper transcription failed: CUDA error")

        with self.assertRaises(RuntimeError) as ctx:
            self.engine.process_voice_answer(
                question="Q?",
                audio_path="audio.wav",
            )
        self.assertIn("Whisper transcription failed", str(ctx.exception))

    def test_transcription_file_not_found_propagates(self) -> None:
        """Test that STT FileNotFoundError propagates."""
        self.engine.start_interview("Developer")
        self.mock_stt.transcribe.side_effect = FileNotFoundError("Audio file not found: missing.wav")

        with self.assertRaises(FileNotFoundError):
            self.engine.process_voice_answer(
                question="Q?",
                audio_path="missing.wav",
            )

    def test_stt_returning_invalid_type_raises_runtime_error(self) -> None:
        """Test that non-TranscriptionResult return from provider raises RuntimeError."""
        self.engine.start_interview("Developer")
        self.mock_stt.transcribe.return_value = "just a plain string"

        with self.assertRaises(RuntimeError) as ctx:
            self.engine.process_voice_answer(
                question="Q?",
                audio_path="audio.wav",
            )
        self.assertIn("Speech-to-text provider returned invalid result type", str(ctx.exception))

    def test_empty_transcription_text_raises_value_error_from_analytics(self) -> None:
        """Test that empty or whitespace-only transcription text fails validation in analytics."""
        self.engine.start_interview("Developer")
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="   ",
            duration_seconds=5.0,
        )

        with self.assertRaises(ValueError) as ctx:
            self.engine.process_voice_answer(
                question="Q?",
                audio_path="silent.wav",
            )
        self.assertIn("Transcription text cannot be empty", str(ctx.exception))
        self.assertEqual(self.engine.state.answers, [])


class TestEndToEndVoiceAnswerWorkflow(unittest.TestCase):
    """End-to-end integration tests for the complete voice-answer workflow.

    Workflow:
    1. The adaptive engine displays/selects an interview question as text.
    2. A candidate audio file is passed to process_voice_answer().
    3. The speech-to-text provider transcribes the audio.
    4. Speech analytics analyzes the transcript and duration.
    5. The transcribed answer is recorded in the interview state.
    6. If an evaluation score is supplied, adaptive difficulty and weak-topic tracking are updated.
    7. The result returns the transcription, analytics, and optional adaptive result.
    """

    def setUp(self) -> None:
        self.mock_stt = Mock(spec=SpeechToTextProvider)
        self.mock_analytics = Mock(spec=SpeechAnalytics)
        self.engine = AdaptiveInterviewEngine(
            stt_provider=self.mock_stt,
            speech_analytics=self.mock_analytics,
        )

    def test_end_to_end_workflow_with_evaluation_score(self) -> None:
        """Verify complete workflow when an evaluation score is provided:
        - Question selection as text
        - Audio transcription and duration forwarding to analytics
        - Answer recorded in state
        - Adaptive difficulty and weak-topic tracking updated
        - Full result structure returned
        - Follow-up selection respects adaptation
        """
        # Start session
        state = self.engine.start_interview("Backend Engineer", initial_difficulty="medium")

        # Step 1: Engine displays / selects interview question as text
        question = self.engine.get_next_question()
        self.assertIsNotNone(question)
        self.assertIsInstance(question.text, str)
        self.assertTrue(len(question.text) > 0)
        self.assertEqual(question.difficulty, "medium")
        self.assertIn(question.id, state.questions_asked)
        self.assertEqual(state.question_number, 1)

        # Step 2-4: Candidate audio submitted; STT transcribes; analytics evaluates
        expected_transcription = TranscriptionResult(
            text="Tuples are immutable sequence types in Python.",
            language="en",
            duration_seconds=4.5,
            provider="whisper",
        )
        self.mock_stt.transcribe.return_value = expected_transcription

        expected_analytics = SpeechAnalyticsResult(
            word_count=7,
            filler_words_count=0,
            filler_words={},
            duration_seconds=4.5,
            words_per_minute=93.33,
            filler_ratio=0.0,
            pace_status="slow",
            fluency_score=93.3,
            fluency_indicators={"filler_ratio": 0.0, "pace_status": "slow"},
        )
        self.mock_analytics.analyze.return_value = expected_analytics

        eval_score = 42.0
        voice_result = self.engine.process_voice_answer(
            question=question.text,
            audio_path="tuple_explanation.wav",
            score=eval_score,
            topic=question.topic,
        )

        # Verify STT received exact audio path
        self.mock_stt.transcribe.assert_called_once_with("tuple_explanation.wav")

        # Verify analytics received exact transcription text and duration
        self.mock_analytics.analyze.assert_called_once_with(
            text=expected_transcription.text,
            duration_seconds=4.5,
        )

        # Step 5: Transcribed answer recorded in state
        self.assertEqual(len(state.answers), 1)
        self.assertEqual(state.answers[0], expected_transcription.text)
        self.assertIn(question.text, state.questions_asked)

        # Step 6: Adaptive difficulty & weak-topic tracking updated
        self.assertEqual(state.scores, [eval_score])
        self.assertEqual(state.difficulty, "easy")  # Score < 50 decreases difficulty
        self.assertIn(question.topic, state.weak_topics)

        # Step 7: Result contains transcription, analytics, and adaptive result
        self.assertEqual(voice_result.audio_path, "tuple_explanation.wav")
        self.assertIs(voice_result.transcription, expected_transcription)
        self.assertIs(voice_result.analytics, expected_analytics)
        self.assertIsNotNone(voice_result.adaptive_result)
        self.assertEqual(voice_result.adaptive_result.score, eval_score)
        self.assertEqual(voice_result.adaptive_result.difficulty, "easy")
        self.assertTrue(voice_result.adaptive_result.is_weak_topic)
        self.assertTrue(voice_result.adaptive_result.should_follow_up)

        # Subsequent question selection incorporates adaptive outcome
        follow_up_q = self.engine.get_next_question(
            is_follow_up=voice_result.adaptive_result.should_follow_up,
            topic=voice_result.adaptive_result.topic,
        )
        self.assertIsNotNone(follow_up_q)
        self.assertIn(follow_up_q.id, state.questions_asked)

    def test_end_to_end_workflow_without_evaluation_score(self) -> None:
        """Verify complete workflow when no evaluation score is supplied:
        - Question selection as text
        - Audio transcription and duration forwarding to analytics
        - Answer recorded in state
        - Adaptive updates DO NOT occur (difficulty and weak topics unchanged)
        - Result returns transcription, analytics, and adaptive_result is None
        """
        state = self.engine.start_interview("Backend Engineer", initial_difficulty="medium")

        # Step 1: Engine displays question as text
        question = self.engine.get_next_question()
        self.assertIsNotNone(question)

        # Step 2-4: Candidate audio submitted without evaluation score
        expected_transcription = TranscriptionResult(
            text="Python functions are first-class objects.",
            language="en",
            duration_seconds=3.0,
            provider="whisper",
        )
        self.mock_stt.transcribe.return_value = expected_transcription

        expected_analytics = SpeechAnalyticsResult(
            word_count=5,
            filler_words_count=0,
            filler_words={},
            duration_seconds=3.0,
            words_per_minute=100.0,
            filler_ratio=0.0,
            pace_status="slow",
            fluency_score=96.0,
            fluency_indicators={"filler_ratio": 0.0, "pace_status": "slow"},
        )
        self.mock_analytics.analyze.return_value = expected_analytics

        voice_result = self.engine.process_voice_answer(
            question=question.text,
            audio_path="functions.wav",
            score=None,
            topic=question.topic,
        )

        # Verify STT and analytics calls
        self.mock_stt.transcribe.assert_called_once_with("functions.wav")
        self.mock_analytics.analyze.assert_called_once_with(
            text=expected_transcription.text,
            duration_seconds=3.0,
        )

        # Step 5: Answer recorded in state
        self.assertEqual(len(state.answers), 1)
        self.assertEqual(state.answers[0], expected_transcription.text)

        # Step 6: Verify adaptive updates DO NOT occur
        self.assertEqual(state.scores, [])
        self.assertEqual(state.difficulty, "medium")  # Remained unchanged
        self.assertEqual(state.weak_topics, [])       # Remained unchanged

        # Step 7: adaptive_result is None
        self.assertIsNone(voice_result.adaptive_result)
        self.assertIs(voice_result.transcription, expected_transcription)
        self.assertIs(voice_result.analytics, expected_analytics)

    def test_errors_do_not_leave_interview_state_partially_updated(self) -> None:
        """Verify that errors at any stage of process_voice_answer do not corrupt state."""
        state = self.engine.start_interview("Backend Engineer", initial_difficulty="medium")
        snapshot_answers = list(state.answers)
        snapshot_scores = list(state.scores)
        snapshot_weak = list(state.weak_topics)
        snapshot_difficulty = state.difficulty
        snapshot_count = state.question_number

        def assert_state_unmodified() -> None:
            self.assertEqual(state.answers, snapshot_answers)
            self.assertEqual(state.scores, snapshot_scores)
            self.assertEqual(state.weak_topics, snapshot_weak)
            self.assertEqual(state.difficulty, snapshot_difficulty)
            self.assertEqual(state.question_number, snapshot_count)

        # 1. Invalid question error -> state untouched
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="", audio_path="audio.wav")
        assert_state_unmodified()

        # 2. Invalid audio path error -> state untouched
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Question?", audio_path="")
        assert_state_unmodified()

        # 3. Invalid score error -> state untouched, STT not called
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Question?", audio_path="audio.wav", score=150.0)
        assert_state_unmodified()
        self.mock_stt.transcribe.assert_not_called()

        # 4. Transcription failure -> state untouched
        self.mock_stt.transcribe.side_effect = RuntimeError("Whisper transcription failed: OOM")
        with self.assertRaises(RuntimeError):
            self.engine.process_voice_answer(question="Question?", audio_path="audio.wav", score=80.0)
        assert_state_unmodified()

        # 5. Audio file not found -> state untouched
        self.mock_stt.transcribe.side_effect = FileNotFoundError("File not found: missing.wav")
        with self.assertRaises(FileNotFoundError):
            self.engine.process_voice_answer(question="Question?", audio_path="missing.wav")
        assert_state_unmodified()

        # 6. Analytics failure (e.g. empty transcript) -> state untouched
        self.mock_stt.transcribe.side_effect = None
        self.mock_stt.transcribe.return_value = TranscriptionResult(text="   ", duration_seconds=2.0)
        self.mock_analytics.analyze.side_effect = ValueError("Transcription text cannot be empty or whitespace only.")
        with self.assertRaises(ValueError):
            self.engine.process_voice_answer(question="Question?", audio_path="audio.wav", score=80.0)
        assert_state_unmodified()

    def test_multi_turn_voice_interview_lifecycle(self) -> None:
        """Verify multi-turn interview lifecycle with progressive difficulty scaling."""
        state = self.engine.start_interview("Python Engineer", initial_difficulty="medium")

        # Turn 1: Question 1 -> Poor score (35.0) -> Difficulty drops to 'easy'
        q1 = self.engine.get_next_question()
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="I am not sure about GIL in Python.",
            duration_seconds=5.0,
        )
        self.mock_analytics.analyze.return_value = SpeechAnalyticsResult(
            word_count=8,
            filler_words_count=0,
            filler_words={},
            fluency_score=85.0,
        )
        res1 = self.engine.process_voice_answer(question=q1.text, audio_path="turn1.wav", score=35.0, topic="Python")

        self.assertEqual(state.difficulty, "easy")
        self.assertIn("Python", state.weak_topics)
        self.assertEqual(len(state.answers), 1)
        self.assertTrue(res1.adaptive_result.should_follow_up)

        # Turn 2: Remediation Follow-up -> Strong score (88.0) -> Difficulty scales to 'medium', weak topic cleared
        q2 = self.engine.get_next_question(is_follow_up=True, topic="Python")
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="The Global Interpreter Lock ensures only one thread executes Python bytecode at a time.",
            duration_seconds=6.0,
        )
        self.mock_analytics.analyze.return_value = SpeechAnalyticsResult(
            word_count=13,
            filler_words_count=0,
            filler_words={},
            fluency_score=95.0,
        )
        res2 = self.engine.process_voice_answer(question=q2.text, audio_path="turn2.wav", score=88.0, topic="Python")

        self.assertEqual(state.difficulty, "medium")
        self.assertNotIn("Python", state.weak_topics)
        self.assertEqual(len(state.answers), 2)
        self.assertFalse(res2.adaptive_result.should_follow_up)

        # Turn 3: Next Question -> Voice answer without evaluation score
        q3 = self.engine.get_next_question()
        self.mock_stt.transcribe.return_value = TranscriptionResult(
            text="Generators yield values on the fly without loading all items in memory.",
            duration_seconds=5.0,
        )
        self.mock_analytics.analyze.return_value = SpeechAnalyticsResult(
            word_count=12,
            filler_words_count=0,
            filler_words={},
            fluency_score=98.0,
        )
        res3 = self.engine.process_voice_answer(question=q3.text, audio_path="turn3.wav", score=None)

        self.assertIsNone(res3.adaptive_result)
        self.assertEqual(state.difficulty, "medium")  # Unchanged
        self.assertEqual(len(state.answers), 3)

        # Complete interview session
        self.engine.complete_session()
        self.assertTrue(self.engine.is_session_complete())

        # Attempting voice answer on completed session raises ValueError
        with self.assertRaises(ValueError) as ctx:
            self.engine.process_voice_answer(question="Late question?", audio_path="late.wav")
        self.assertIn("session is not in progress", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
