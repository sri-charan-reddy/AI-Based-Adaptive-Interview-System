"""Unit tests for Adaptive Interview Module (Member 2).

Tests state tracking, answer recording, difficulty scaling rules, boundary conditions,
and error handling.
"""

from __future__ import annotations

import unittest

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import Difficulty, InterviewState


class TestAdaptiveInterviewEngine(unittest.TestCase):
    """Test suite for AdaptiveInterviewEngine and InterviewState."""

    def setUp(self) -> None:
        """Set up a fresh engine instance before each test."""
        self.engine = AdaptiveInterviewEngine()

    # 1. Starting an interview
    def test_start_interview_default_difficulty(self) -> None:
        """Test starting an interview with default medium difficulty."""
        state = self.engine.start_interview(role="Backend Developer")

        self.assertIsInstance(state, InterviewState)
        self.assertEqual(state.role, "Backend Developer")
        self.assertEqual(state.difficulty, "medium")
        self.assertEqual(state.question_number, 0)
        self.assertEqual(state.questions_asked, [])
        self.assertEqual(state.answers, [])
        self.assertEqual(state.scores, [])
        self.assertEqual(state.topics, [])
        self.assertEqual(state.weak_topics, [])
        self.assertIs(self.engine.state, state)

    def test_start_interview_custom_difficulty_and_topics(self) -> None:
        """Test starting an interview with specified difficulty and initial topics."""
        initial_topics = ["Data Structures", "System Design"]
        state = self.engine.start_interview(
            role="Senior Engineer",
            initial_difficulty="easy",
            topics=initial_topics,
        )

        self.assertEqual(state.role, "Senior Engineer")
        self.assertEqual(state.difficulty, "easy")
        self.assertEqual(state.topics, ["Data Structures", "System Design"])

    def test_start_interview_empty_role_raises_error(self) -> None:
        """Test that starting an interview with an empty role raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.start_interview(role="")

        with self.assertRaises(ValueError):
            self.engine.start_interview(role="   ")

    def test_start_interview_invalid_difficulty_raises_error(self) -> None:
        """Test that starting an interview with invalid difficulty raises ValueError."""
        with self.assertRaises(ValueError):
            self.engine.start_interview(role="DevOps", initial_difficulty="expert")

    # 2. Recording an answer
    def test_record_answer(self) -> None:
        """Test recording questions and candidate answers."""
        self.engine.start_interview(role="Python Developer")

        q1 = "What is GIL in Python?"
        a1 = "Global Interpreter Lock prevents multiple native threads from executing Python bytecodes simultaneously."
        updated_state = self.engine.record_answer(question=q1, answer=a1)

        self.assertEqual(updated_state.question_number, 1)
        self.assertEqual(updated_state.questions_asked, [q1])
        self.assertEqual(updated_state.answers, [a1])

        q2 = "What are Python decorators?"
        a2 = "Decorators are functions that modify the behavior of another function."
        updated_state = self.engine.record_answer(question=q2, answer=a2)

        self.assertEqual(updated_state.question_number, 2)
        self.assertEqual(updated_state.questions_asked, [q1, q2])
        self.assertEqual(updated_state.answers, [a1, a2])

    def test_record_answer_without_active_state_raises_error(self) -> None:
        """Test that recording an answer without starting an interview raises ValueError."""
        fresh_engine = AdaptiveInterviewEngine()
        with self.assertRaises(ValueError):
            fresh_engine.record_answer(question="Question?", answer="Answer.")

    # 3. Increasing difficulty (score >= 80)
    def test_increase_difficulty_from_easy_to_medium(self) -> None:
        """Test that a score >= 80 increases difficulty from easy to medium."""
        self.engine.start_interview(role="Frontend Developer", initial_difficulty="easy")

        next_diff = self.engine.get_next_difficulty(current_difficulty="easy", score=80.0)
        self.assertEqual(next_diff, "medium")

        # Via update_performance
        result = self.engine.update_performance(score=85.0)
        self.assertEqual(result, "medium")
        self.assertEqual(self.engine.state.difficulty, "medium")
        self.assertEqual(self.engine.state.scores, [85.0])

    def test_increase_difficulty_from_medium_to_hard(self) -> None:
        """Test that a score >= 80 increases difficulty from medium to hard."""
        self.engine.start_interview(role="Frontend Developer", initial_difficulty="medium")

        next_diff = self.engine.get_next_difficulty(current_difficulty="medium", score=95.0)
        self.assertEqual(next_diff, "hard")

        result = self.engine.update_performance(score=95.0)
        self.assertEqual(result, "hard")
        self.assertEqual(self.engine.state.difficulty, "hard")

    def test_increase_difficulty_boundary_at_80(self) -> None:
        """Test the exact boundary score of 80 triggers an increase."""
        next_diff = self.engine.get_next_difficulty(current_difficulty="easy", score=80.0)
        self.assertEqual(next_diff, "medium")

    # 4. Keeping difficulty unchanged (score 50-79)
    def test_keep_difficulty_unchanged_for_medium(self) -> None:
        """Test that scores between 50 and 79 maintain current difficulty."""
        self.engine.start_interview(role="Data Engineer", initial_difficulty="medium")

        for test_score in [50.0, 65.5, 79.0, 79.9]:
            with self.subTest(score=test_score):
                next_diff = self.engine.get_next_difficulty(
                    current_difficulty="medium",
                    score=test_score,
                )
                self.assertEqual(next_diff, "medium")

    def test_keep_difficulty_unchanged_for_easy(self) -> None:
        """Test that scores between 50 and 79 keep 'easy' difficulty unchanged."""
        next_diff = self.engine.get_next_difficulty(current_difficulty="easy", score=60.0)
        self.assertEqual(next_diff, "easy")

    def test_keep_difficulty_unchanged_for_hard(self) -> None:
        """Test that scores between 50 and 79 keep 'hard' difficulty unchanged."""
        next_diff = self.engine.get_next_difficulty(current_difficulty="hard", score=75.0)
        self.assertEqual(next_diff, "hard")

    # 5. Decreasing difficulty (score < 50)
    def test_decrease_difficulty_from_hard_to_medium(self) -> None:
        """Test that a score < 50 decreases difficulty from hard to medium."""
        self.engine.start_interview(role="ML Engineer", initial_difficulty="hard")

        next_diff = self.engine.get_next_difficulty(current_difficulty="hard", score=45.0)
        self.assertEqual(next_diff, "medium")

        result = self.engine.update_performance(score=45.0)
        self.assertEqual(result, "medium")
        self.assertEqual(self.engine.state.difficulty, "medium")

    def test_decrease_difficulty_from_medium_to_easy(self) -> None:
        """Test that a score < 50 decreases difficulty from medium to easy."""
        self.engine.start_interview(role="ML Engineer", initial_difficulty="medium")

        next_diff = self.engine.get_next_difficulty(current_difficulty="medium", score=30.0)
        self.assertEqual(next_diff, "easy")

        result = self.engine.update_performance(score=30.0)
        self.assertEqual(result, "easy")
        self.assertEqual(self.engine.state.difficulty, "easy")

    def test_decrease_difficulty_boundary_below_50(self) -> None:
        """Test that a score just below 50 (e.g. 49.9) decreases difficulty."""
        next_diff = self.engine.get_next_difficulty(current_difficulty="hard", score=49.9)
        self.assertEqual(next_diff, "medium")

    # 6. Preventing difficulty above hard
    def test_prevent_difficulty_above_hard(self) -> None:
        """Test that difficulty cannot exceed 'hard' on high scores."""
        self.engine.start_interview(role="Security Engineer", initial_difficulty="hard")

        next_diff = self.engine.get_next_difficulty(current_difficulty="hard", score=100.0)
        self.assertEqual(next_diff, "hard")

        result = self.engine.update_performance(score=98.0)
        self.assertEqual(result, "hard")
        self.assertEqual(self.engine.state.difficulty, "hard")

    # 7. Preventing difficulty below easy
    def test_prevent_difficulty_below_easy(self) -> None:
        """Test that difficulty cannot drop below 'easy' on low scores."""
        self.engine.start_interview(role="QA Engineer", initial_difficulty="easy")

        next_diff = self.engine.get_next_difficulty(current_difficulty="easy", score=0.0)
        self.assertEqual(next_diff, "easy")

        result = self.engine.update_performance(score=25.0)
        self.assertEqual(result, "easy")
        self.assertEqual(self.engine.state.difficulty, "easy")

    # 8. Weak topics and topic tracking
    def test_track_weak_topic_on_low_score(self) -> None:
        """Test that score < 50 adds topic to weak_topics and topics list."""
        self.engine.start_interview(role="Fullstack Engineer")

        self.engine.update_performance(score=40.0, topic="Concurrency")
        self.assertIn("Concurrency", self.engine.state.topics)
        self.assertIn("Concurrency", self.engine.state.weak_topics)

    def test_strong_score_does_not_add_to_weak_topics(self) -> None:
        """Test that score >= 50 does not add topic to weak_topics."""
        self.engine.start_interview(role="Fullstack Engineer")

        self.engine.update_performance(score=85.0, topic="Algorithms")
        self.assertIn("Algorithms", self.engine.state.topics)
        self.assertNotIn("Algorithms", self.engine.state.weak_topics)

    def test_weak_topics_avoid_duplicates(self) -> None:
        """Test that weak topics list avoids duplicate entries."""
        self.engine.start_interview(role="Fullstack Engineer")

        self.engine.update_performance(score=30.0, topic="SQL")
        self.engine.update_performance(score=20.0, topic="SQL")

        self.assertEqual(self.engine.state.weak_topics.count("SQL"), 1)

    # 9. Validation and Error Handling
    def test_score_validation_out_of_bounds(self) -> None:
        """Test that scores outside [0, 100] raise ValueError."""
        with self.assertRaises(ValueError):
            self.engine.get_next_difficulty(current_difficulty="medium", score=-1.0)

        with self.assertRaises(ValueError):
            self.engine.get_next_difficulty(current_difficulty="medium", score=100.1)

    def test_score_validation_non_numeric(self) -> None:
        """Test that non-numeric scores raise ValueError."""
        with self.assertRaises(ValueError):
            self.engine.get_next_difficulty(current_difficulty="medium", score="high")  # type: ignore

    def test_difficulty_case_insensitivity(self) -> None:
        """Test that difficulty strings are handled case-insensitively."""
        self.assertEqual(self.engine.get_next_difficulty("EASY", 90.0), "medium")
        self.assertEqual(self.engine.get_next_difficulty("Medium", 40.0), "easy")
        self.assertEqual(self.engine.get_next_difficulty("HARD", 60.0), "hard")

    # 10. State serialization (to_dict / from_dict)
    def test_state_serialization(self) -> None:
        """Test converting InterviewState to and from dictionary."""
        original_state = InterviewState(
            role="Cloud Architect",
            difficulty="hard",
            question_number=3,
            questions_asked=["Q1", "Q2", "Q3"],
            answers=["A1", "A2", "A3"],
            scores=[85.0, 90.0, 75.0],
            topics=["AWS", "Kubernetes"],
            weak_topics=["Networking"],
        )

        state_dict = original_state.to_dict()
        self.assertEqual(state_dict["role"], "Cloud Architect")
        self.assertEqual(state_dict["difficulty"], "hard")
        self.assertEqual(state_dict["question_number"], 3)

        reconstructed_state = InterviewState.from_dict(state_dict)
        self.assertEqual(reconstructed_state.role, original_state.role)
        self.assertEqual(reconstructed_state.difficulty, original_state.difficulty)
        self.assertEqual(reconstructed_state.question_number, original_state.question_number)
        self.assertEqual(reconstructed_state.questions_asked, original_state.questions_asked)
        self.assertEqual(reconstructed_state.answers, original_state.answers)
        self.assertEqual(reconstructed_state.scores, original_state.scores)
        self.assertEqual(reconstructed_state.topics, original_state.topics)
        self.assertEqual(reconstructed_state.weak_topics, original_state.weak_topics)


if __name__ == "__main__":
    unittest.main()
