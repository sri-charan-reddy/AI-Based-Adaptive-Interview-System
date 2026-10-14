"""Data models for the Adaptive Interview Engine.

This module defines the core data structures used to maintain and track the state
of an ongoing adaptive interview session, question items, evaluation results, and session lifecycle statuses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from adaptive_interview.speech.models import (
    SpeechAnalyticsResult,
    TranscriptionResult,
    VoiceAnswerResult,
)


class Difficulty(str, Enum):
    """Supported difficulty levels for the interview questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

    @classmethod
    def from_string(cls, value: str | Difficulty) -> Difficulty:
        """Parse and normalize a difficulty level string or enum.

        Args:
            value: Difficulty as string or Difficulty enum instance.

        Returns:
            Normalized Difficulty enum.

        Raises:
            ValueError: If the string does not match a valid difficulty level.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise ValueError(f"Difficulty must be a string or Difficulty enum, got {type(value).__name__}")

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        valid_levels = [m.value for m in cls]
        raise ValueError(f"Invalid difficulty '{value}'. Supported levels are: {valid_levels}")


class SessionStatus(str, Enum):
    """Lifecycle status states for an interview session."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

    @classmethod
    def from_string(cls, value: str | SessionStatus) -> SessionStatus:
        """Parse and normalize a session status string or enum.

        Args:
            value: Session status as string or SessionStatus enum instance.

        Returns:
            Normalized SessionStatus enum.

        Raises:
            ValueError: If the string does not match a valid session status.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            raise ValueError(f"Session status must be a string or SessionStatus enum, got {type(value).__name__}")

        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member

        valid_statuses = [m.value for m in cls]
        raise ValueError(f"Invalid session status '{value}'. Supported statuses are: {valid_statuses}")


@dataclass
class Question:
    """Represents an interview question item.

    Attributes:
        id: Unique identifier for the question (e.g. 'py_01').
        text: The question text / prompt.
        topic: The topic category (e.g. 'Python', 'OOP', 'Data Structures').
        difficulty: Difficulty level ('easy', 'medium', 'hard').
        follow_up: Boolean flag indicating if this is a follow-up question.
    """

    id: str
    text: str
    topic: str
    difficulty: str = Difficulty.MEDIUM.value
    follow_up: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize question attributes."""
        if not self.id or not str(self.id).strip():
            raise ValueError("Question id must be a non-empty string.")
        if not self.text or not str(self.text).strip():
            raise ValueError("Question text must be a non-empty string.")
        if not self.topic or not str(self.topic).strip():
            raise ValueError("Question topic must be a non-empty string.")

        self.id = str(self.id).strip()
        self.text = str(self.text).strip()
        self.topic = str(self.topic).strip()
        validated_diff = Difficulty.from_string(self.difficulty)
        self.difficulty = validated_diff.value
        self.follow_up = bool(self.follow_up)

    def to_dict(self) -> dict[str, Any]:
        """Convert the question to a serializable dictionary.

        Returns:
            Dictionary containing question attributes.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """Create a Question instance from a dictionary.

        Args:
            data: Dictionary containing question fields.

        Returns:
            A new Question instance.
        """
        return cls(
            id=data["id"],
            text=data["text"],
            topic=data["topic"],
            difficulty=data.get("difficulty", Difficulty.MEDIUM.value),
            follow_up=data.get("follow_up", False),
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. question['text'])."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


@dataclass
class AdaptiveResult:
    """Represents the structured outcome of candidate answer processing.

    Attributes:
        score: Evaluated performance score (0-100 scale).
        topic: Topic associated with the answer (or None).
        difficulty: Resulting difficulty level ('easy', 'medium', 'hard').
        is_weak_topic: Boolean indicating whether the evaluated topic is currently weak.
        should_follow_up: Boolean indicating whether an adaptive follow-up is triggered.
    """

    score: float
    topic: str | None
    difficulty: str
    is_weak_topic: bool
    should_follow_up: bool

    def __post_init__(self) -> None:
        """Validate and normalize result attributes."""
        validated_diff = Difficulty.from_string(self.difficulty)
        self.difficulty = validated_diff.value
        self.score = float(self.score)
        self.topic = str(self.topic).strip() if self.topic is not None and str(self.topic).strip() else None
        self.is_weak_topic = bool(self.is_weak_topic)
        self.should_follow_up = bool(self.should_follow_up)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a serializable dictionary.

        Returns:
            Dictionary containing evaluation and adaptation details.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveResult:
        """Create an AdaptiveResult instance from a dictionary.

        Args:
            data: Dictionary containing result fields.

        Returns:
            A new AdaptiveResult instance.
        """
        return cls(
            score=float(data["score"]),
            topic=data.get("topic"),
            difficulty=str(data["difficulty"]),
            is_weak_topic=bool(data.get("is_weak_topic", False)),
            should_follow_up=bool(data.get("should_follow_up", False)),
        )

    def __getitem__(self, item: str) -> Any:
        """Allow dictionary-style subscripting (e.g. result['should_follow_up'])."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


@dataclass
class InterviewState:
    """Represents the complete state of an adaptive interview session.

    Attributes:
        role: The target role for the interview (e.g. 'Software Engineer').
        difficulty: Current difficulty level ('easy', 'medium', 'hard').
        question_number: The count of questions asked so far.
        questions_asked: Ordered list of question prompts / IDs asked to the candidate.
        answers: Ordered list of answers submitted by the candidate.
        scores: Ordered list of evaluation scores received for answers (0-100 scale).
        topics: List of topics covered or targeted in the interview.
        weak_topics: List of topics where the candidate underperformed.
        status: Current lifecycle status of the session ('not_started', 'in_progress', 'completed').
        max_questions: Optional maximum number of questions to be asked in the session.
    """

    role: str
    difficulty: str = Difficulty.MEDIUM.value
    question_number: int = 0
    questions_asked: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    status: str = SessionStatus.NOT_STARTED.value
    max_questions: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize difficulty level, status, and max_questions."""
        validated_diff = Difficulty.from_string(self.difficulty)
        self.difficulty = validated_diff.value

        validated_status = SessionStatus.from_string(self.status)
        self.status = validated_status.value

        if self.max_questions is not None:
            if not isinstance(self.max_questions, int) or isinstance(self.max_questions, bool) or self.max_questions <= 0:
                raise ValueError("max_questions must be a positive integer.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the interview state to a serializable dictionary.

        Returns:
            Dictionary containing all state attributes.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterviewState:
        """Create an InterviewState instance from a dictionary.

        Args:
            data: Dictionary containing interview state fields.

        Returns:
            A new InterviewState instance.
        """
        return cls(
            role=data["role"],
            difficulty=data.get("difficulty", Difficulty.MEDIUM.value),
            question_number=data.get("question_number", 0),
            questions_asked=list(data.get("questions_asked", [])),
            answers=list(data.get("answers", [])),
            scores=list(data.get("scores", [])),
            topics=list(data.get("topics", [])),
            weak_topics=list(data.get("weak_topics", [])),
            status=data.get("status", SessionStatus.NOT_STARTED.value),
            max_questions=data.get("max_questions"),
        )
