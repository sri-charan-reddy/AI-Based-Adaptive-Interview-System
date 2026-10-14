"""Unit tests for Interview Session Control & Lifecycle (Task 4).

Tests session statuses, lifecycle transitions (start_session, complete_session, is_session_complete),
max_questions length limits, question progression, bank exhaustion, and answer recording constraints.
"""

from __future__ import annotations

import unittest

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import Difficulty, InterviewState, Question, SessionStatus


class TestSessionLifecycle(unittest.TestCase):
    """Test suite for interview session lifecycle and control."""

    def setUp(self) -> None:
        """Create a fresh engine instance before each test."""
        self.engine = AdaptiveInterviewEngine()

    # 1. Default status & SessionStatus validation
    def test_default_session_status(self) -> None:
        """Test default status of InterviewState is 'not_started'."""
        state = InterviewState(role="Backend Engineer")
        self.assertEqual(state.status, "not_started")
        self.assertEqual(state.status, SessionStatus.NOT_STARTED.value)
        self.assertIsNone(state.max_questions)

    def test_session_status_enum_validation(self) -> None:
        """Test SessionStatus from_string validation and case normalization."""
        self.assertEqual(SessionStatus.from_string("NOT_STARTED"), SessionStatus.NOT_STARTED)
        self.assertEqual(SessionStatus.from_string("In_Progress"), SessionStatus.IN_PROGRESS)
        self.assertEqual(SessionStatus.from_string("completed"), SessionStatus.COMPLETED)

        with self.assertRaises(ValueError):
            SessionStatus.from_string("invalid_status")

        with self.assertRaises(ValueError):
            SessionStatus.from_string(123)  # type: ignore

    # 2. Lifecycle transitions: start_session & complete_session
    def test_start_session_transition(self) -> None:
        """Test start_session transitions not_started -> in_progress."""
        state = InterviewState(role="Backend Engineer", status="not_started")
        self.engine.state = state

        updated = self.engine.start_session()
        self.assertEqual(updated.status, "in_progress")
        self.assertFalse(self.engine.is_session_complete())

    def test_complete_session_transition(self) -> None:
        """Test complete_session transitions in_progress -> completed."""
        state = self.engine.start_interview(role="DevOps Engineer")
        self.assertEqual(state.status, "in_progress")
        self.assertFalse(self.engine.is_session_complete())

        completed_state = self.engine.complete_session()
        self.assertEqual(completed_state.status, "completed")
        self.assertTrue(self.engine.is_session_complete())

    def test_invalid_lifecycle_transitions(self) -> None:
        """Test invalid lifecycle transitions raise ValueError."""
        # 1. Cannot start an already completed session
        completed_state = InterviewState(role="Engineer", status="completed")
        self.engine.state = completed_state
        with self.assertRaises(ValueError):
            self.engine.start_session()

        # 2. Cannot complete a session that is not in_progress (e.g. not_started)
        not_started_state = InterviewState(role="Engineer", status="not_started")
        self.engine.state = not_started_state
        with self.assertRaises(ValueError):
            self.engine.complete_session()

        # 3. Cannot complete a session that is already completed
        with self.assertRaises(ValueError):
            self.engine.complete_session(state=completed_state)

    # 3. max_questions validation & storage
    def test_max_questions_storage_and_serialization(self) -> None:
        """Test storing and serializing max_questions in InterviewState."""
        state = self.engine.start_interview(
            role="Data Engineer",
            max_questions=5,
        )
        self.assertEqual(state.max_questions, 5)
        self.assertEqual(state.status, "in_progress")

        state_dict = state.to_dict()
        self.assertEqual(state_dict["max_questions"], 5)
        self.assertEqual(state_dict["status"], "in_progress")

        reconstructed = InterviewState.from_dict(state_dict)
        self.assertEqual(reconstructed.max_questions, 5)
        self.assertEqual(reconstructed.status, "in_progress")

    def test_max_questions_validation(self) -> None:
        """Test max_questions rejects non-positive or non-integer values."""
        with self.assertRaises(ValueError):
            self.engine.start_interview(role="Dev", max_questions=0)

        with self.assertRaises(ValueError):
            self.engine.start_interview(role="Dev", max_questions=-3)

        with self.assertRaises(ValueError):
            self.engine.start_interview(role="Dev", max_questions=3.5)  # type: ignore

        with self.assertRaises(ValueError):
            self.engine.start_interview(role="Dev", max_questions=True)  # type: ignore

    # 4. Question progression and limit enforcement
    def test_question_progression_and_limit_enforcement(self) -> None:
        """Test progression and completion when max_questions limit is reached."""
        state = self.engine.start_interview(role="Software Engineer", max_questions=2)
        self.assertEqual(state.question_number, 0)
        self.assertEqual(state.status, "in_progress")

        # Question 1
        q1 = self.engine.get_next_question()
        self.assertIsNotNone(q1)
        self.assertEqual(state.question_number, 1)
        self.assertEqual(state.status, "in_progress")
        self.assertIn(q1.id, state.questions_asked)

        # Question 2 (Final allowed question)
        q2 = self.engine.get_next_question()
        self.assertIsNotNone(q2)
        self.assertEqual(state.question_number, 2)
        self.assertEqual(state.status, "in_progress")
        self.assertIn(q2.id, state.questions_asked)

        # 3rd request: Limit reached -> returns None and marks completed
        q3 = self.engine.get_next_question()
        self.assertIsNone(q3)
        self.assertEqual(state.status, "completed")
        self.assertTrue(self.engine.is_session_complete())

        # 4th request: Already completed -> returns None
        q4 = self.engine.get_next_question()
        self.assertIsNone(q4)
        self.assertEqual(state.status, "completed")

    # 5. Bank exhaustion
    def test_bank_exhaustion_marks_session_completed(self) -> None:
        """Test bank exhaustion marks session completed and does not add to questions_asked."""
        small_bank = [
            Question(id="q1", text="Text 1", topic="Python", difficulty="easy"),
        ]
        engine = AdaptiveInterviewEngine(question_bank=small_bank)
        state = engine.start_interview(role="Dev")

        q1 = engine.get_next_question()
        self.assertIsNotNone(q1)
        self.assertEqual(state.questions_asked, ["q1"])
        self.assertEqual(state.question_number, 1)
        self.assertEqual(state.status, "in_progress")

        # Bank now exhausted
        q2 = engine.get_next_question()
        self.assertIsNone(q2)
        self.assertEqual(state.status, "completed")
        self.assertTrue(engine.is_session_complete())
        self.assertEqual(state.questions_asked, ["q1"])  # Nothing added

    # 6. Answer recording lifecycle enforcement
    def test_record_answer_while_in_progress(self) -> None:
        """Test recording answer succeeds while session is in_progress."""
        self.engine.start_interview(role="Frontend Dev")
        q = self.engine.get_next_question()

        updated_state = self.engine.record_answer(question=q.text, answer="Clean response")
        self.assertEqual(updated_state.answers, ["Clean response"])
        self.assertEqual(updated_state.question_number, 1)

    def test_record_answer_rejected_when_not_in_progress(self) -> None:
        """Test recording answer raises ValueError when session is not in progress."""
        # 1. not_started
        not_started_state = InterviewState(role="Dev", status="not_started")
        self.engine.state = not_started_state
        with self.assertRaises(ValueError):
            self.engine.record_answer(question="Q?", answer="A.")

        # 2. completed
        self.engine.start_interview(role="Dev")
        self.engine.complete_session()
        with self.assertRaises(ValueError):
            self.engine.record_answer(question="Q?", answer="A.")


if __name__ == "__main__":
    unittest.main()
