"""Core engine for managing adaptive interview flow, session lifecycle, and difficulty adjustment.

This module houses the AdaptiveInterviewEngine, which evaluates candidate performance
scores, manages session state and lifecycle transitions, adaptively adjusts the interview difficulty,
tracks weak topics, and determines follow-up question strategies according to established heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from adaptive_interview.models import (
    AdaptiveResult,
    Difficulty,
    InterviewState,
    Question,
    SessionStatus,
    VoiceAnswerResult,
)
from adaptive_interview.selector import QuestionSelector
from adaptive_interview.speech.analytics import SpeechAnalytics
from adaptive_interview.speech.base import SpeechToTextProvider
from adaptive_interview.speech.models import TranscriptionResult


class AdaptiveInterviewEngine:
    """Adaptive Interview Engine that manages interview state, lifecycle, difficulty, and questions.

    This engine operates independently of external LLMs or databases and provides
    modular methods to start sessions, record interactions, update candidate performance,
    dynamically scale question difficulty, manage session lifecycle, and evaluate follow-ups.
    """

    # Difficulty progression order
    DIFFICULTY_LEVELS: list[str] = [
        Difficulty.EASY.value,
        Difficulty.MEDIUM.value,
        Difficulty.HARD.value,
    ]

    # Performance thresholds
    HIGH_PERFORMANCE_THRESHOLD: float = 80.0
    MEDIUM_PERFORMANCE_THRESHOLD: float = 50.0

    def __init__(
        self,
        state: InterviewState | None = None,
        selector: QuestionSelector | None = None,
        question_bank: Sequence[Question] | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        speech_analytics: SpeechAnalytics | None = None,
    ) -> None:
        """Initialize the engine with an optional pre-existing state and question selector.

        Args:
            state: Optional existing InterviewState instance.
            selector: Optional custom QuestionSelector instance.
            question_bank: Optional question bank to initialize the selector with.
            stt_provider: Optional SpeechToTextProvider for speech-to-text transcription.
            speech_analytics: Optional SpeechAnalytics for verbal fluency evaluation.
        """
        self.state: InterviewState | None = state
        self.selector: QuestionSelector = (
            selector if selector is not None else QuestionSelector(question_bank=question_bank)
        )
        self._stt_provider: SpeechToTextProvider | None = stt_provider
        self._speech_analytics: SpeechAnalytics | None = speech_analytics

    @property
    def stt_provider(self) -> SpeechToTextProvider:
        """Get or lazily initialize the SpeechToTextProvider."""
        if self._stt_provider is None:
            from adaptive_interview.speech.whisper_stt import WhisperSpeechToText

            self._stt_provider = WhisperSpeechToText()
        return self._stt_provider

    @stt_provider.setter
    def stt_provider(self, provider: SpeechToTextProvider) -> None:
        """Set a custom SpeechToTextProvider."""
        self._stt_provider = provider

    @property
    def speech_analytics(self) -> SpeechAnalytics:
        """Get or lazily initialize the SpeechAnalytics instance."""
        if self._speech_analytics is None:
            from adaptive_interview.speech.analytics import SpeechAnalytics

            self._speech_analytics = SpeechAnalytics()
        return self._speech_analytics

    @speech_analytics.setter
    def speech_analytics(self, analytics: SpeechAnalytics) -> None:
        """Set a custom SpeechAnalytics instance."""
        self._speech_analytics = analytics

    def start_interview(
        self,
        role: str,
        initial_difficulty: str | Difficulty = Difficulty.MEDIUM,
        topics: Sequence[str] | None = None,
        max_questions: int | None = None,
    ) -> InterviewState:
        """Start a new interview session.

        Initializes a fresh InterviewState with the given role, starting difficulty,
        initial topics (if any), and optional max questions limit.

        Args:
            role: The job role or domain being interviewed for (e.g., 'Backend Engineer').
            initial_difficulty: Initial difficulty level ('easy', 'medium', 'hard').
            topics: Optional initial list of topics to be covered.
            max_questions: Optional maximum number of questions for this session.

        Returns:
            The initialized InterviewState with status 'in_progress'.

        Raises:
            ValueError: If role is empty, difficulty is invalid, or max_questions is not a positive int.
        """
        if not role or not role.strip():
            raise ValueError("Role must be a non-empty string.")

        if max_questions is not None:
            if not isinstance(max_questions, int) or isinstance(max_questions, bool) or max_questions <= 0:
                raise ValueError("max_questions must be a positive integer.")

        validated_diff = Difficulty.from_string(initial_difficulty)
        topic_list = [t.strip() for t in topics if t and t.strip()] if topics is not None else []

        self.state = InterviewState(
            role=role.strip(),
            difficulty=validated_diff.value,
            question_number=0,
            questions_asked=[],
            answers=[],
            scores=[],
            topics=topic_list,
            weak_topics=[],
            status=SessionStatus.IN_PROGRESS.value,
            max_questions=max_questions,
        )
        return self.state

    def start_session(self, state: InterviewState | None = None) -> InterviewState:
        """Transition an interview session to 'in_progress'.

        Args:
            state: Optional InterviewState to start. If omitted, uses self.state.

        Returns:
            The updated InterviewState.

        Raises:
            ValueError: If the session is already completed or no state exists.
        """
        target_state = self._resolve_state(state)

        if target_state.status == SessionStatus.COMPLETED.value:
            raise ValueError("Cannot start an already completed interview session.")

        target_state.status = SessionStatus.IN_PROGRESS.value
        return target_state

    def complete_session(self, state: InterviewState | None = None) -> InterviewState:
        """Transition an active interview session to 'completed'.

        Args:
            state: Optional InterviewState to complete. If omitted, uses self.state.

        Returns:
            The updated InterviewState.

        Raises:
            ValueError: If the session is not currently in progress or no state exists.
        """
        target_state = self._resolve_state(state)

        if target_state.status != SessionStatus.IN_PROGRESS.value:
            raise ValueError(
                f"Cannot complete session: session is not in progress (current status: '{target_state.status}')."
            )

        target_state.status = SessionStatus.COMPLETED.value
        return target_state

    def is_session_complete(self, state: InterviewState | None = None) -> bool:
        """Check whether the interview session is marked completed.

        Args:
            state: Optional InterviewState to check. If omitted, uses self.state.

        Returns:
            True if session status is 'completed', False otherwise.
        """
        target_state = self._resolve_state(state)
        return target_state.status == SessionStatus.COMPLETED.value

    def record_answer(
        self,
        question: str,
        answer: str,
        state: InterviewState | None = None,
    ) -> InterviewState:
        """Record a question asked and the candidate's answer.

        Stores both the question and answer in state and updates question_number.

        Args:
            question: The interview question presented to the candidate.
            answer: The candidate's response.
            state: Optional InterviewState to update. If omitted, uses self.state.

        Returns:
            The updated InterviewState.

        Raises:
            ValueError: If session is not in progress or no active state exists.
        """
        target_state = self._resolve_state(state)

        if target_state.status != SessionStatus.IN_PROGRESS.value:
            raise ValueError(
                f"Cannot record answer: session is not in progress (current status: '{target_state.status}')."
            )

        if question not in target_state.questions_asked:
            target_state.questions_asked.append(question)
        target_state.answers.append(answer)

        if len(target_state.answers) > target_state.question_number:
            target_state.question_number = len(target_state.answers)

        return target_state

    def update_performance(
        self,
        score: float,
        topic: str | None = None,
        state: InterviewState | None = None,
    ) -> str:
        """Update candidate performance score, weak topics, and adjust difficulty.

        Evaluation rules:
            - Validates score in [0, 100].
            - Appends score to state.scores.
            - Normalizes topic name.
            - If score < 50: Adds topic to state.weak_topics (avoiding duplicates).
            - If score >= 80: Removes topic from state.weak_topics if previously present.
            - If 50 <= score < 80: Keeps weak_topics unchanged.
            - Adjusts state.difficulty based on score thresholds.

        Args:
            score: Performance score on a 0-100 scale.
            topic: Optional topic associated with the evaluated answer.
            state: Optional InterviewState to update. If omitted, uses self.state.

        Returns:
            The new difficulty level ('easy', 'medium', or 'hard').

        Raises:
            ValueError: If score is outside 0-100 range or no active state exists.
        """
        target_state = self._resolve_state(state)
        self._validate_score(score)

        target_state.scores.append(float(score))

        if topic is not None and str(topic).strip():
            clean_topic = str(topic).strip()
            topic_lower = clean_topic.lower()

            # Record in session topics if not already present
            existing_topics_lower = [t.lower() for t in target_state.topics]
            if topic_lower not in existing_topics_lower:
                target_state.topics.append(clean_topic)

            weak_topics_lower = [w.lower() for w in target_state.weak_topics]

            if score < self.MEDIUM_PERFORMANCE_THRESHOLD:
                # Poor performance (< 50): add to weak_topics
                if topic_lower not in weak_topics_lower:
                    target_state.weak_topics.append(clean_topic)
            elif score >= self.HIGH_PERFORMANCE_THRESHOLD:
                # Strong performance (>= 80): remove from weak_topics if present
                target_state.weak_topics = [
                    w for w in target_state.weak_topics if w.lower() != topic_lower
                ]

        new_difficulty = self.get_next_difficulty(
            current_difficulty=target_state.difficulty,
            score=score,
        )
        target_state.difficulty = new_difficulty

        return new_difficulty

    def should_ask_follow_up(
        self,
        score: float,
        topic: str | None = None,
        state: InterviewState | None = None,
    ) -> bool:
        """Determine whether an adaptive follow-up question should be triggered.

        Rules:
            - score < 50: True (low score triggers remediation follow-up).
            - score 50-79: True if topic is currently in weak_topics, False otherwise.
            - score >= 80: False (strong performance moves to next question).

        Args:
            score: Performance score on a 0-100 scale.
            topic: Optional topic associated with the answer.
            state: Optional InterviewState to inspect weak topics. If omitted, uses self.state.

        Returns:
            Boolean indicating if follow-up is appropriate.

        Raises:
            ValueError: If score is outside 0-100 range.
        """
        self._validate_score(score)
        target_state = state if state is not None else self.state

        if score < self.MEDIUM_PERFORMANCE_THRESHOLD:
            return True

        if score >= self.HIGH_PERFORMANCE_THRESHOLD:
            return False

        # score is between 50 and 79
        if topic is not None and str(topic).strip() and target_state is not None:
            clean_topic_lower = str(topic).strip().lower()
            weak_topics_lower = [w.lower() for w in target_state.weak_topics]
            if clean_topic_lower in weak_topics_lower:
                return True

        return False

    def process_answer(
        self,
        score: float,
        topic: str | None = None,
        state: InterviewState | None = None,
    ) -> AdaptiveResult:
        """Process a candidate's answer score in a single unified operation.

        Performs end-to-end evaluation:
            1. Validates score.
            2. Determines follow-up need.
            3. Updates performance, weak topics, and difficulty.
            4. Packages the evaluation and adaptation metrics into an AdaptiveResult.

        Args:
            score: Performance score on a 0-100 scale.
            topic: Optional topic associated with the answer.
            state: Optional InterviewState to update. If omitted, uses self.state.

        Returns:
            An AdaptiveResult dataclass containing score, topic, new difficulty,
            weak topic status, and should_follow_up flag.

        Raises:
            ValueError: If score is outside 0-100 range or no active state exists.
        """
        target_state = self._resolve_state(state)
        self._validate_score(score)

        clean_topic = str(topic).strip() if topic is not None and str(topic).strip() else None

        # Determine follow-up requirement (evaluates prior weak status for borderline scores)
        should_follow = self.should_ask_follow_up(
            score=score,
            topic=clean_topic,
            state=target_state,
        )

        # Update performance metrics, weak topics, and difficulty
        new_difficulty = self.update_performance(
            score=score,
            topic=clean_topic,
            state=target_state,
        )

        # Check if topic is currently categorized as weak
        is_weak = False
        if clean_topic is not None:
            clean_lower = clean_topic.lower()
            is_weak = any(w.lower() == clean_lower for w in target_state.weak_topics)

        return AdaptiveResult(
            score=float(score),
            topic=clean_topic,
            difficulty=new_difficulty,
            is_weak_topic=is_weak,
            should_follow_up=should_follow,
        )

    def process_voice_answer(
        self,
        question: str,
        audio_path: str | Path,
        score: float | None = None,
        topic: str | None = None,
        state: InterviewState | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        speech_analytics: SpeechAnalytics | None = None,
    ) -> VoiceAnswerResult:
        """Process a candidate's voice answer through STT, analytics, and adaptive evaluation.

        Coordinates the end-to-end voice answer pipeline:
            1. Validates session state, inputs, and score.
            2. Transcribes candidate audio into text using the STT provider.
            3. Analyzes transcription text and audio duration for speaking rate, fillers, and fluency.
            4. Records the transcribed answer into the active interview state.
            5. If an evaluation score is provided, updates performance metrics and adapts difficulty.
            6. Returns a structured VoiceAnswerResult.

        Args:
            question: The interview question presented to the candidate.
            audio_path: File path of the candidate's audio recording.
            score: Optional performance score on a 0-100 scale.
            topic: Optional topic associated with the evaluated answer.
            state: Optional InterviewState instance. If omitted, uses self.state.
            stt_provider: Optional SpeechToTextProvider override. If omitted, uses self.stt_provider.
            speech_analytics: Optional SpeechAnalytics override. If omitted, uses self.speech_analytics.

        Returns:
            A VoiceAnswerResult containing transcription, analytics, and optional adaptive result.

        Raises:
            ValueError: If session is not in progress, audio path is empty/invalid,
                        question is empty, or score is outside [0, 100].
            FileNotFoundError: If audio file does not exist (when validated by provider).
            RuntimeError: If speech-to-text transcription fails.
        """
        target_state = self._resolve_state(state)

        if target_state.status != SessionStatus.IN_PROGRESS.value:
            raise ValueError(
                f"Cannot process voice answer: session is not in progress (current status: '{target_state.status}')."
            )

        if question is None or not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        clean_question = question.strip()

        if audio_path is None or not isinstance(audio_path, (str, Path)):
            raise ValueError(
                f"Audio path must be a string or Path, got {type(audio_path).__name__ if audio_path is not None else 'None'}"
            )
        audio_path_str = str(audio_path).strip()
        if not audio_path_str:
            raise ValueError("Audio path cannot be empty.")

        if score is not None:
            self._validate_score(score)

        provider = stt_provider if stt_provider is not None else self.stt_provider
        analytics_engine = speech_analytics if speech_analytics is not None else self.speech_analytics

        try:
            transcription = provider.transcribe(audio_path_str)
        except (ValueError, FileNotFoundError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Voice answer transcription failed: {exc}") from exc

        if not isinstance(transcription, TranscriptionResult):
            raise RuntimeError(
                f"Speech-to-text provider returned invalid result type: {type(transcription).__name__}"
            )

        analytics = analytics_engine.analyze(
            text=transcription.text,
            duration_seconds=transcription.duration_seconds,
        )

        self.record_answer(
            question=clean_question,
            answer=transcription.text,
            state=target_state,
        )

        adaptive_result: AdaptiveResult | None = None
        if score is not None:
            adaptive_result = self.process_answer(
                score=score,
                topic=topic,
                state=target_state,
            )

        return VoiceAnswerResult(
            audio_path=audio_path_str,
            transcription=transcription,
            analytics=analytics,
            adaptive_result=adaptive_result,
        )

    def get_next_difficulty(
        self,
        current_difficulty: str | Difficulty,
        score: float,
    ) -> str:
        """Calculate the next difficulty level based on performance score.

        Adaptive rules:
            - score >= 80: increase difficulty (capped at 'hard')
            - score 50-79: keep difficulty unchanged
            - score < 50: decrease difficulty (capped at 'easy')

        Boundary limits:
            - 'easy' cannot decrease further
            - 'hard' cannot increase further

        Args:
            current_difficulty: Current difficulty level ('easy', 'medium', 'hard').
            score: Candidate score on a 0-100 scale.

        Returns:
            The next difficulty level string ('easy', 'medium', 'hard').

        Raises:
            ValueError: If score is out of [0, 100] range or difficulty is invalid.
        """
        self._validate_score(score)
        validated_diff = Difficulty.from_string(current_difficulty)
        current_idx = self.DIFFICULTY_LEVELS.index(validated_diff.value)

        if score >= self.HIGH_PERFORMANCE_THRESHOLD:
            # Increase difficulty, bounded by 'hard'
            next_idx = min(current_idx + 1, len(self.DIFFICULTY_LEVELS) - 1)
        elif score >= self.MEDIUM_PERFORMANCE_THRESHOLD:
            # Keep unchanged
            next_idx = current_idx
        else:
            # Decrease difficulty, bounded by 'easy'
            next_idx = max(current_idx - 1, 0)

        return self.DIFFICULTY_LEVELS[next_idx]

    def get_next_question(
        self,
        state: InterviewState | None = None,
        is_follow_up: bool = False,
        topic: str | None = None,
    ) -> Question | None:
        """Retrieve the next adaptive question based on current interview state.

        Delegates selection to QuestionSelector to find the best unasked question
        matching the current difficulty level and weak topic focus, tracks the selected
        question ID, increments question count, and manages session lifecycle and limits.

        Args:
            state: Optional InterviewState. If omitted, uses self.state.
            is_follow_up: Whether to request a follow-up question.
            topic: Optional explicit topic constraint or preference.

        Returns:
            A Question object, or None if max questions limit is reached or bank is exhausted.

        Raises:
            ValueError: If no active interview state is found.
        """
        target_state = self._resolve_state(state)

        # If already completed
        if target_state.status == SessionStatus.COMPLETED.value:
            return None

        # Auto-transition not_started to in_progress if needed
        if target_state.status == SessionStatus.NOT_STARTED.value:
            target_state.status = SessionStatus.IN_PROGRESS.value

        # Check if max_questions limit has been reached
        if target_state.max_questions is not None and target_state.question_number >= target_state.max_questions:
            target_state.status = SessionStatus.COMPLETED.value
            return None

        question = self.selector.select_next_question(
            state=target_state,
            is_follow_up=is_follow_up,
            topic=topic,
        )

        if question is None:
            # Bank exhausted
            target_state.status = SessionStatus.COMPLETED.value
            return None

        if question.id not in target_state.questions_asked:
            target_state.questions_asked.append(question.id)
        target_state.question_number += 1

        return question

    def _resolve_state(self, state: InterviewState | None) -> InterviewState:
        """Resolve the target state from explicit argument or internal state.

        Args:
            state: Explicit state if provided.

        Returns:
            The active InterviewState instance.

        Raises:
            ValueError: If no valid state is available.
        """
        target_state = state if state is not None else self.state
        if target_state is None:
            raise ValueError(
                "No active interview state found. Call start_interview() first or pass a state instance."
            )
        return target_state

    @staticmethod
    def _validate_score(score: float) -> None:
        """Validate that score is numeric and within bounds [0, 100].

        Args:
            score: The score to validate.

        Raises:
            ValueError: If score is not between 0 and 100.
        """
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"Score must be a number, got {type(score).__name__}")
        if score < 0.0 or score > 100.0:
            raise ValueError(f"Score must be between 0 and 100, got {score}")
