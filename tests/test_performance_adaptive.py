"""Unit tests for Performance-Based Adaptive Interview Logic (Task 3).

Tests performance score validation, weak-topic remediation, difficulty adjustment,
follow-up decision making (should_ask_follow_up), and unified answer processing (process_answer).
"""

from __future__ import annotations

import unittest

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import AdaptiveResult, Difficulty, InterviewState


class TestPerformanceAdaptiveLogic(unittest.TestCase):
    """Test suite for performance evaluation and adaptive answer processing."""

    def setUp(self) -> None:
        """Set up a fresh engine and interview state for each test."""
        self.engine = AdaptiveInterviewEngine()
        self.state = self.engine.start_interview(
            role="Fullstack Engineer",
            initial_difficulty="medium",
            topics=["Python", "OOP", "Data Structures"],
        )

    # 1. Score validation
    def test_score_validation_boundaries(self) -> None:
        """Test score validation on boundaries (0 and 100)."""
        # Valid boundaries
        self.assertEqual(self.engine.update_performance(score=0.0), "easy")
        self.assertEqual(self.engine.update_performance(score=100.0), "medium")

    def test_score_validation_out_of_range(self) -> None:
        """Test that invalid score ranges raise ValueError."""
        with self.assertRaises(ValueError):
            self.engine.update_performance(score=-0.1)

        with self.assertRaises(ValueError):
            self.engine.update_performance(score=100.1)

        with self.assertRaises(ValueError):
            self.engine.should_ask_follow_up(score=-5.0)

        with self.assertRaises(ValueError):
            self.engine.process_answer(score=150.0)

    def test_score_validation_invalid_types(self) -> None:
        """Test non-numeric score types raise ValueError."""
        with self.assertRaises(ValueError):
            self.engine.process_answer(score="85")  # type: ignore

        with self.assertRaises(ValueError):
            self.engine.process_answer(score=True)  # type: ignore

    # 2. Difficulty adaptation rules & boundaries
    def test_high_score_increases_difficulty(self) -> None:
        """Test that scores >= 80 increase difficulty."""
        self.state.difficulty = "easy"
        new_diff = self.engine.update_performance(score=80.0)
        self.assertEqual(new_diff, "medium")
        self.assertEqual(self.state.difficulty, "medium")

        new_diff2 = self.engine.update_performance(score=85.0)
        self.assertEqual(new_diff2, "hard")
        self.assertEqual(self.state.difficulty, "hard")

    def test_medium_score_keeps_difficulty(self) -> None:
        """Test that scores between 50 and 79 preserve current difficulty."""
        self.state.difficulty = "medium"
        self.assertEqual(self.engine.update_performance(score=50.0), "medium")
        self.assertEqual(self.engine.update_performance(score=65.0), "medium")
        self.assertEqual(self.engine.update_performance(score=79.9), "medium")
        self.assertEqual(self.state.difficulty, "medium")

    def test_low_score_decreases_difficulty(self) -> None:
        """Test that scores < 50 decrease difficulty."""
        self.state.difficulty = "hard"
        new_diff = self.engine.update_performance(score=49.0)
        self.assertEqual(new_diff, "medium")
        self.assertEqual(self.state.difficulty, "medium")

        new_diff2 = self.engine.update_performance(score=20.0)
        self.assertEqual(new_diff2, "easy")
        self.assertEqual(self.state.difficulty, "easy")

    def test_difficulty_boundaries_capped(self) -> None:
        """Test that difficulty cannot exceed hard or drop below easy."""
        self.state.difficulty = "hard"
        self.assertEqual(self.engine.update_performance(score=95.0), "hard")
        self.assertEqual(self.state.difficulty, "hard")

        self.state.difficulty = "easy"
        self.assertEqual(self.engine.update_performance(score=15.0), "easy")
        self.assertEqual(self.state.difficulty, "easy")

    # 3. Weak topic creation, duplicate prevention, and topic normalization
    def test_weak_topic_creation_on_low_score(self) -> None:
        """Test that score < 50 adds normalized topic to weak_topics."""
        self.engine.update_performance(score=45.0, topic="  Python  ")
        self.assertIn("Python", self.state.weak_topics)
        self.assertIn("Python", self.state.topics)
        self.assertEqual(self.state.scores, [45.0])

    def test_no_duplicate_weak_topics(self) -> None:
        """Test that repeated low scores on the same topic do not create duplicates."""
        self.engine.update_performance(score=30.0, topic="OOP")
        self.engine.update_performance(score=40.0, topic="oop")
        self.engine.update_performance(score=25.0, topic=" OOP ")

        self.assertEqual(len(self.state.weak_topics), 1)
        self.assertEqual(self.state.weak_topics[0].lower(), "oop")

    def test_strong_score_removes_previously_weak_topic(self) -> None:
        """Test that a score >= 80 removes a topic from weak_topics (remediation)."""
        # First, mark Python as weak
        self.engine.update_performance(score=40.0, topic="Python")
        self.assertIn("Python", self.state.weak_topics)

        # Candidate now achieves a strong score on Python
        self.engine.update_performance(score=85.0, topic="python")
        self.assertNotIn("Python", self.state.weak_topics)
        self.assertNotIn("python", self.state.weak_topics)
        self.assertEqual(len(self.state.weak_topics), 0)

    def test_moderate_score_preserves_weak_topic(self) -> None:
        """Test that a score between 50 and 79 does not remove an existing weak topic."""
        self.engine.update_performance(score=35.0, topic="Data Structures")
        self.assertIn("Data Structures", self.state.weak_topics)

        # Moderate score (65) should keep it in weak_topics until score >= 80 is achieved
        self.engine.update_performance(score=65.0, topic="Data Structures")
        self.assertIn("Data Structures", self.state.weak_topics)

    # 4. Adaptive Follow-up Decision (should_ask_follow_up)
    def test_should_ask_follow_up_low_score(self) -> None:
        """Test follow-up decision is True for low scores (<50)."""
        self.assertTrue(self.engine.should_ask_follow_up(score=45.0, topic="Python"))
        self.assertTrue(self.engine.should_ask_follow_up(score=0.0))
        self.assertTrue(self.engine.should_ask_follow_up(score=49.9))

    def test_should_ask_follow_up_high_score(self) -> None:
        """Test follow-up decision is False for high scores (>=80)."""
        self.assertFalse(self.engine.should_ask_follow_up(score=80.0, topic="Python"))
        self.assertFalse(self.engine.should_ask_follow_up(score=95.0, topic="Python"))
        self.assertFalse(self.engine.should_ask_follow_up(score=100.0))

    def test_should_ask_follow_up_medium_score(self) -> None:
        """Test follow-up decision for medium scores (50-79)."""
        # Normal non-weak topic -> False
        self.assertFalse(self.engine.should_ask_follow_up(score=65.0, topic="Python"))

        # When topic is currently in weak_topics -> True
        self.state.weak_topics = ["Python"]
        self.assertTrue(self.engine.should_ask_follow_up(score=65.0, topic="Python"))
        self.assertTrue(self.engine.should_ask_follow_up(score=75.0, topic="python"))  # case-insensitive

        # For a different non-weak topic -> False
        self.assertFalse(self.engine.should_ask_follow_up(score=65.0, topic="OOP"))

    # 5. High-Level Process Answer (process_answer)
    def test_process_answer_low_score_creates_weak_topic_and_triggers_follow_up(self) -> None:
        """Test process_answer when candidate struggles (<50)."""
        result = self.engine.process_answer(score=45.0, topic="Python")

        self.assertIsInstance(result, AdaptiveResult)
        self.assertEqual(result.score, 45.0)
        self.assertEqual(result.topic, "Python")
        self.assertEqual(result.difficulty, "easy")
        self.assertTrue(result.is_weak_topic)
        self.assertTrue(result.should_follow_up)

        # Dict subscripting support
        self.assertEqual(result["difficulty"], "easy")
        self.assertTrue(result["should_follow_up"])

        # Serialization to dict
        result_dict = result.to_dict()
        self.assertEqual(result_dict["score"], 45.0)
        self.assertEqual(result_dict["topic"], "Python")
        self.assertEqual(result_dict["difficulty"], "easy")
        self.assertTrue(result_dict["is_weak_topic"])
        self.assertTrue(result_dict["should_follow_up"])

    def test_process_answer_high_score_clears_weak_topic_and_scales_up(self) -> None:
        """Test process_answer on strong score clears weak topic and scales difficulty up."""
        # Initial weak state
        self.state.weak_topics = ["Python"]
        self.state.difficulty = "easy"

        result = self.engine.process_answer(score=90.0, topic="Python")

        self.assertEqual(result.difficulty, "medium")
        self.assertFalse(result.is_weak_topic)
        self.assertFalse(result.should_follow_up)
        self.assertEqual(self.state.weak_topics, [])

    def test_process_answer_without_topic(self) -> None:
        """Test process_answer when no topic is provided."""
        result = self.engine.process_answer(score=70.0)

        self.assertEqual(result.score, 70.0)
        self.assertIsNone(result.topic)
        self.assertEqual(result.difficulty, "medium")
        self.assertFalse(result.is_weak_topic)
        self.assertFalse(result.should_follow_up)

    def test_end_to_end_adaptive_session_loop(self) -> None:
        """Test end-to-end adaptive evaluation cycle integrated with question selection."""
        # 1. Start interview
        engine = AdaptiveInterviewEngine()
        state = engine.start_interview(role="Software Engineer", initial_difficulty="medium", topics=["Python", "OOP"])

        # 2. Get initial question
        q1 = engine.get_next_question()
        self.assertIsNotNone(q1)

        # 3. Process candidate response: poor score on Python
        engine.record_answer(question=q1.text, answer="Incomplete Python answer")
        res1 = engine.process_answer(score=40.0, topic=q1.topic)

        self.assertEqual(res1.difficulty, "easy")
        self.assertTrue(res1.should_follow_up)
        self.assertTrue(res1.is_weak_topic)

        # 4. Fetch follow-up question targeting weak topic
        q_fu = engine.get_next_question(is_follow_up=res1.should_follow_up, topic=res1.topic)
        self.assertIsNotNone(q_fu)

        # 5. Candidate masters follow-up question
        engine.record_answer(question=q_fu.text, answer="Excellent follow-up answer")
        res2 = engine.process_answer(score=90.0, topic=q_fu.topic)

        self.assertEqual(res2.difficulty, "medium")
        self.assertFalse(res2.should_follow_up)
        self.assertFalse(res2.is_weak_topic)
        self.assertEqual(state.weak_topics, [])


if __name__ == "__main__":
    unittest.main()
