"""Unit tests for QuestionSelector and Adaptive Question/Follow-up Logic (Task 2).

Tests difficulty-based selection, topic matching, weak-topic prioritization,
preventing duplicate questions, follow-up selection, fallback mechanics, and engine integration.
"""

from __future__ import annotations

import unittest

from adaptive_interview.engine import AdaptiveInterviewEngine
from adaptive_interview.models import Difficulty, InterviewState, Question
from adaptive_interview.question_bank import (
    DEFAULT_QUESTION_BANK,
    get_default_question_bank,
)
from adaptive_interview.selector import QuestionSelector


class TestQuestionSelector(unittest.TestCase):
    """Test suite for Question, QuestionSelector, and engine question retrieval."""

    def setUp(self) -> None:
        """Create custom test questions and default selector instance."""
        self.custom_bank = [
            Question(id="q_py_easy", text="Python Easy Text", topic="Python", difficulty="easy", follow_up=False),
            Question(id="q_py_med", text="Python Med Text", topic="Python", difficulty="medium", follow_up=False),
            Question(id="q_py_hard", text="Python Hard Text", topic="Python", difficulty="hard", follow_up=False),
            Question(id="q_py_fu", text="Python Follow-up Text", topic="Python", difficulty="medium", follow_up=True),
            Question(id="q_oop_easy", text="OOP Easy Text", topic="OOP", difficulty="easy", follow_up=False),
            Question(id="q_oop_med", text="OOP Med Text", topic="OOP", difficulty="medium", follow_up=False),
            Question(id="q_oop_hard", text="OOP Hard Text", topic="OOP", difficulty="hard", follow_up=False),
            Question(id="q_ds_easy", text="DS Easy Text", topic="Data Structures", difficulty="easy", follow_up=False),
            Question(id="q_ds_med", text="DS Med Text", topic="Data Structures", difficulty="medium", follow_up=False),
            Question(id="q_ds_hard", text="DS Hard Text", topic="Data Structures", difficulty="hard", follow_up=False),
        ]
        self.selector = QuestionSelector(question_bank=self.custom_bank)

    # 1. Question Model Tests
    def test_question_model_attributes_and_subscript(self) -> None:
        """Test Question dataclass fields, validation, and dict-like subscripting."""
        q = Question(
            id="test_01",
            text="What is polymorphism?",
            topic="OOP",
            difficulty="medium",
            follow_up=False,
        )
        self.assertEqual(q.id, "test_01")
        self.assertEqual(q.text, "What is polymorphism?")
        self.assertEqual(q.topic, "OOP")
        self.assertEqual(q.difficulty, "medium")
        self.assertFalse(q.follow_up)

        # Dictionary access support
        self.assertEqual(q["id"], "test_01")
        self.assertEqual(q["text"], "What is polymorphism?")

        # Serialization to and from dict
        q_dict = q.to_dict()
        self.assertIsInstance(q_dict, dict)
        self.assertEqual(q_dict["id"], "test_01")

        reconstructed = Question.from_dict(q_dict)
        self.assertEqual(reconstructed.id, q.id)
        self.assertEqual(reconstructed.text, q.text)

    def test_question_model_validation(self) -> None:
        """Test validation on invalid Question fields."""
        with self.assertRaises(ValueError):
            Question(id="", text="Text", topic="Python")

        with self.assertRaises(ValueError):
            Question(id="1", text="", topic="Python")

        with self.assertRaises(ValueError):
            Question(id="1", text="Text", topic="")

        with self.assertRaises(ValueError):
            Question(id="1", text="Text", topic="Python", difficulty="extreme")

    # 2. Difficulty-based Selection
    def test_difficulty_based_selection_easy(self) -> None:
        """Test selecting question when state difficulty is easy."""
        state = InterviewState(role="Backend Dev", difficulty="easy", topics=["Python"])
        selected = self.selector.select_next_question(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.difficulty, "easy")
        self.assertEqual(selected.id, "q_py_easy")

    def test_difficulty_based_selection_medium(self) -> None:
        """Test selecting question when state difficulty is medium."""
        state = InterviewState(role="Backend Dev", difficulty="medium", topics=["Python"])
        selected = self.selector.select_next_question(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.difficulty, "medium")
        self.assertEqual(selected.id, "q_py_med")

    def test_difficulty_based_selection_hard(self) -> None:
        """Test selecting question when state difficulty is hard."""
        state = InterviewState(role="Backend Dev", difficulty="hard", topics=["Python"])
        selected = self.selector.select_next_question(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.difficulty, "hard")
        self.assertEqual(selected.id, "q_py_hard")

    # 3. Topic Matching
    def test_topic_matching_from_session_topics(self) -> None:
        """Test selector honors topics configured in interview state."""
        state_oop = InterviewState(role="Backend Dev", difficulty="easy", topics=["OOP"])
        selected_oop = self.selector.select_next_question(state_oop)
        self.assertEqual(selected_oop.topic, "OOP")
        self.assertEqual(selected_oop.id, "q_oop_easy")

        state_ds = InterviewState(role="Backend Dev", difficulty="hard", topics=["Data Structures"])
        selected_ds = self.selector.select_next_question(state_ds)
        self.assertEqual(selected_ds.topic, "Data Structures")
        self.assertEqual(selected_ds.id, "q_ds_hard")

    def test_explicit_topic_parameter_override(self) -> None:
        """Test passing an explicit topic constraint to select_next_question."""
        state = InterviewState(role="Backend Dev", difficulty="medium", topics=["Python"])
        selected = self.selector.select_next_question(state, topic="OOP")

        self.assertIsNotNone(selected)
        self.assertEqual(selected.topic, "OOP")
        self.assertEqual(selected.difficulty, "medium")
        self.assertEqual(selected.id, "q_oop_med")

    # 4. Weak-Topic Prioritization
    def test_weak_topic_prioritization(self) -> None:
        """Test that weak topics are prioritized over normal session topics."""
        state = InterviewState(
            role="Fullstack Dev",
            difficulty="easy",
            topics=["Python", "OOP", "Data Structures"],
            weak_topics=["OOP"],
        )
        selected = self.selector.select_next_question(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.topic, "OOP")
        self.assertEqual(selected.difficulty, "easy")
        self.assertEqual(selected.id, "q_oop_easy")

    def test_weak_topic_prioritization_with_case_insensitivity(self) -> None:
        """Test that weak topic matching works case-insensitively."""
        state = InterviewState(
            role="Engineer",
            difficulty="medium",
            topics=["Python"],
            weak_topics=["data structures"],
        )
        selected = self.selector.select_next_question(state)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.topic, "Data Structures")
        self.assertEqual(selected.id, "q_ds_med")

    # 5. No Duplicate Questions
    def test_no_duplicate_questions_by_id_and_text(self) -> None:
        """Test that already asked questions (by ID or Text) are never repeated."""
        state = InterviewState(
            role="Backend Dev",
            difficulty="easy",
            topics=["Python"],
            questions_asked=["q_py_easy"],  # asked by ID
        )
        selected1 = self.selector.select_next_question(state)
        # Should not return q_py_easy; should pick another easy or next available
        self.assertNotEqual(selected1.id, "q_py_easy")

        # Now mark q_oop_easy asked by text
        state.questions_asked.append("OOP Easy Text")
        selected2 = self.selector.select_next_question(state)
        self.assertNotIn(selected2.id, ["q_py_easy", "q_oop_easy"])
        self.assertEqual(selected2.id, "q_ds_easy")

    # 6. Follow-up Selection
    def test_follow_up_selection(self) -> None:
        """Test selecting a dedicated follow-up question."""
        state = InterviewState(role="Backend Dev", difficulty="medium", topics=["Python"])
        selected = self.selector.select_next_question(state, is_follow_up=True)

        self.assertIsNotNone(selected)
        self.assertTrue(selected.follow_up)
        self.assertEqual(selected.id, "q_py_fu")

    def test_select_follow_up_convenience_method(self) -> None:
        """Test the select_follow_up helper method."""
        state = InterviewState(role="Backend Dev", difficulty="medium", topics=["Python"])
        selected = self.selector.select_follow_up(state)

        self.assertIsNotNone(selected)
        self.assertTrue(selected.follow_up)
        self.assertEqual(selected.id, "q_py_fu")

    # 7. Fallback when Exact Difficulty is Unavailable
    def test_fallback_when_exact_difficulty_unavailable(self) -> None:
        """Test fallback to adjacent difficulty when exact difficulty is exhausted."""
        limited_bank = [
            Question(id="q_py_med", text="Med Text", topic="Python", difficulty="medium", follow_up=False),
            Question(id="q_py_hard", text="Hard Text", topic="Python", difficulty="hard", follow_up=False),
        ]
        limited_selector = QuestionSelector(question_bank=limited_bank)

        # Candidate is at "easy" difficulty, but only "medium" and "hard" exist
        state = InterviewState(role="Backend Dev", difficulty="easy", topics=["Python"])
        selected = limited_selector.select_next_question(state)

        self.assertIsNotNone(selected)
        # Should gracefully fall back to closest difficulty ("medium", dist 1) rather than hard (dist 2)
        self.assertEqual(selected.difficulty, "medium")
        self.assertEqual(selected.id, "q_py_med")

    def test_fallback_when_hard_difficulty_unavailable(self) -> None:
        """Test fallback to medium when hard difficulty is exhausted."""
        limited_bank = [
            Question(id="q_easy", text="Easy Text", topic="Python", difficulty="easy", follow_up=False),
            Question(id="q_med", text="Med Text", topic="Python", difficulty="medium", follow_up=False),
        ]
        limited_selector = QuestionSelector(question_bank=limited_bank)

        state = InterviewState(role="Backend Dev", difficulty="hard", topics=["Python"])
        selected = limited_selector.select_next_question(state)

        self.assertIsNotNone(selected)
        # Should fall back to "medium" (dist 1) before "easy" (dist 2)
        self.assertEqual(selected.difficulty, "medium")
        self.assertEqual(selected.id, "q_med")

    # 8. Behavior when No Questions Remain
    def test_behavior_when_no_questions_remain(self) -> None:
        """Test selector returns None when all questions in the bank are exhausted."""
        small_bank = [
            Question(id="q1", text="Text 1", topic="Python", difficulty="easy"),
            Question(id="q2", text="Text 2", topic="Python", difficulty="medium"),
        ]
        exhausted_selector = QuestionSelector(question_bank=small_bank)

        state = InterviewState(
            role="Backend Dev",
            difficulty="easy",
            questions_asked=["q1", "q2"],
        )
        selected = exhausted_selector.select_next_question(state)
        self.assertIsNone(selected)

    # 9. Default Question Bank Content
    def test_default_question_bank_structure(self) -> None:
        """Test that DEFAULT_QUESTION_BANK contains Python, OOP, and Data Structures across difficulties."""
        bank = get_default_question_bank()
        self.assertGreaterEqual(len(bank), 15)

        topics = {q.topic for q in bank}
        self.assertIn("Python", topics)
        self.assertIn("OOP", topics)
        self.assertIn("Data Structures", topics)

        difficulties = {q.difficulty for q in bank}
        self.assertIn("easy", difficulties)
        self.assertIn("medium", difficulties)
        self.assertIn("hard", difficulties)

        follow_ups = [q for q in bank if q.follow_up]
        self.assertGreater(len(follow_ups), 0)

    # 10. Engine Integration Tests
    def test_engine_get_next_question_integration(self) -> None:
        """Test integration between AdaptiveInterviewEngine and QuestionSelector."""
        engine = AdaptiveInterviewEngine(question_bank=self.custom_bank)
        state = engine.start_interview(role="Software Engineer", initial_difficulty="medium", topics=["Python"])

        # 1. Fetch initial question
        q1 = engine.get_next_question()
        self.assertIsNotNone(q1)
        self.assertEqual(q1.difficulty, "medium")
        self.assertEqual(q1.topic, "Python")

        # 2. Record answer and good performance (score 90) -> Difficulty increases to hard
        engine.record_answer(question=q1.text, answer="Great Python answer")
        new_diff = engine.update_performance(score=90.0, topic="Python")
        self.assertEqual(new_diff, "hard")

        # 3. Next question should be hard difficulty
        q2 = engine.get_next_question()
        self.assertIsNotNone(q2)
        self.assertEqual(q2.difficulty, "hard")
        self.assertNotEqual(q2.id, q1.id)

        # 4. Record answer and poor performance (score 35) -> Difficulty decreases, weak topic recorded
        engine.record_answer(question=q2.text, answer="Weak answer")
        new_diff = engine.update_performance(score=35.0, topic="OOP")
        self.assertEqual(new_diff, "medium")
        self.assertIn("OOP", state.weak_topics)

        # 5. Next question should target weak topic "OOP"
        q3 = engine.get_next_question()
        self.assertIsNotNone(q3)
        self.assertEqual(q3.topic, "OOP")
        self.assertEqual(q3.difficulty, "medium")

    def test_engine_get_next_question_without_active_state_raises_error(self) -> None:
        """Test engine raises ValueError when get_next_question is called without active state."""
        fresh_engine = AdaptiveInterviewEngine()
        with self.assertRaises(ValueError):
            fresh_engine.get_next_question()


class TestQuestionTrackingRegression(unittest.TestCase):
    """Regression test suite proving automatic tracking of selected questions in engine."""

    def test_first_call_records_selected_question_id(self) -> None:
        """Test that the first call to get_next_question automatically records the question ID in state."""
        engine = AdaptiveInterviewEngine()
        state = engine.start_interview(role="Backend Engineer", initial_difficulty="medium", topics=["Python"])

        self.assertEqual(state.questions_asked, [])
        q1 = engine.get_next_question()

        self.assertIsNotNone(q1)
        self.assertIn(q1.id, state.questions_asked)
        self.assertEqual(state.questions_asked, [q1.id])

    def test_subsequent_calls_do_not_return_duplicate_questions(self) -> None:
        """Test that calling get_next_question multiple times returns distinct questions."""
        engine = AdaptiveInterviewEngine()
        state = engine.start_interview(role="Python Developer", initial_difficulty="medium", topics=["Python"])

        q1 = engine.get_next_question()
        q2 = engine.get_next_question()
        q3 = engine.get_next_question()

        self.assertIsNotNone(q1)
        self.assertIsNotNone(q2)
        self.assertIsNotNone(q3)

        # Ensure no duplicates among any of the calls
        self.assertNotEqual(q1.id, q2.id)
        self.assertNotEqual(q2.id, q3.id)
        self.assertNotEqual(q1.id, q3.id)

        # Verify state.questions_asked contains all selected question IDs in order without duplicates
        self.assertEqual(state.questions_asked, [q1.id, q2.id, q3.id])
        self.assertEqual(len(state.questions_asked), 3)

    def test_no_duplicate_entries_when_recording_answer(self) -> None:
        """Test that question ID is not duplicated in questions_asked if record_answer is also called."""
        engine = AdaptiveInterviewEngine()
        state = engine.start_interview(role="Python Developer", initial_difficulty="medium", topics=["Python"])

        q1 = engine.get_next_question()
        self.assertEqual(state.questions_asked, [q1.id])

        # Record answer with question text or ID
        engine.record_answer(question=q1.text, answer="Answer 1")
        self.assertEqual(state.question_number, 1)

        # Second question should still be unique
        q2 = engine.get_next_question()
        self.assertNotEqual(q2.id, q1.id)

    def test_weak_topic_prioritization_with_automatic_tracking(self) -> None:
        """Test that weak-topic prioritization continues to function and tracks distinct questions."""
        custom_bank = [
            Question(id="py_m1", text="Python M1", topic="Python", difficulty="medium"),
            Question(id="py_m2", text="Python M2", topic="Python", difficulty="medium"),
            Question(id="oop_m1", text="OOP M1", topic="OOP", difficulty="medium"),
            Question(id="oop_m2", text="OOP M2", topic="OOP", difficulty="medium"),
        ]
        engine = AdaptiveInterviewEngine(question_bank=custom_bank)
        state = engine.start_interview(role="Engineer", initial_difficulty="medium", topics=["Python", "OOP"])
        state.weak_topics = ["OOP"]

        # Call 1: Should select first OOP question
        q1 = engine.get_next_question()
        self.assertIsNotNone(q1)
        self.assertEqual(q1.topic, "OOP")
        self.assertEqual(q1.id, "oop_m1")

        # Call 2: Should select second OOP question (not the first again)
        q2 = engine.get_next_question()
        self.assertIsNotNone(q2)
        self.assertEqual(q2.topic, "OOP")
        self.assertEqual(q2.id, "oop_m2")

        # Call 3: OOP questions exhausted, falls back to Python without returning oop_m1 or oop_m2
        q3 = engine.get_next_question()
        self.assertIsNotNone(q3)
        self.assertEqual(q3.topic, "Python")
        self.assertIn(q3.id, ["py_m1", "py_m2"])

    def test_follow_up_selection_with_automatic_tracking(self) -> None:
        """Test that follow-up questions are automatically tracked and not duplicated."""
        engine = AdaptiveInterviewEngine()
        state = engine.start_interview(role="Developer", initial_difficulty="medium", topics=["Python"])

        fu1 = engine.get_next_question(is_follow_up=True)
        self.assertIsNotNone(fu1)
        self.assertTrue(fu1.follow_up)
        self.assertIn(fu1.id, state.questions_asked)

        fu2 = engine.get_next_question(is_follow_up=True)
        self.assertIsNotNone(fu2)
        self.assertTrue(fu2.follow_up)
        self.assertNotEqual(fu1.id, fu2.id)
        self.assertEqual(state.questions_asked, [fu1.id, fu2.id])

    def test_exhaustion_with_automatic_tracking(self) -> None:
        """Test that when question bank is exhausted, get_next_question returns None and preserves state."""
        small_bank = [
            Question(id="q1", text="Text 1", topic="Python", difficulty="easy"),
            Question(id="q2", text="Text 2", topic="Python", difficulty="easy"),
        ]
        engine = AdaptiveInterviewEngine(question_bank=small_bank)
        state = engine.start_interview(role="Dev", initial_difficulty="easy", topics=["Python"])

        q1 = engine.get_next_question()
        self.assertEqual(q1.id, "q1")

        q2 = engine.get_next_question()
        self.assertEqual(q2.id, "q2")

        # 3rd call: No questions left
        q3 = engine.get_next_question()
        self.assertIsNone(q3)
        self.assertEqual(state.questions_asked, ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
