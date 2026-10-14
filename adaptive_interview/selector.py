"""Question selection logic for the Adaptive Interview Engine.

This module provides the QuestionSelector class, which deterministically chooses
the next question or follow-up question based on the candidate's current difficulty,
interview topics, weak topic remediation, and history of asked questions.
"""

from __future__ import annotations

from typing import Sequence

from adaptive_interview.models import Difficulty, InterviewState, Question
from adaptive_interview.question_bank import get_default_question_bank


class QuestionSelector:
    """Selects appropriate interview questions based on adaptive heuristics."""

    DIFFICULTY_INDEX: dict[str, int] = {
        Difficulty.EASY.value: 0,
        Difficulty.MEDIUM.value: 1,
        Difficulty.HARD.value: 2,
    }

    def __init__(self, question_bank: Sequence[Question] | None = None) -> None:
        """Initialize the selector with a question bank.

        Args:
            question_bank: Optional sequence of Question items. If None, uses the default bank.
        """
        if question_bank is not None:
            self.question_bank: list[Question] = list(question_bank)
        else:
            self.question_bank = get_default_question_bank()

    def get_unasked_questions(self, state: InterviewState) -> list[Question]:
        """Return all questions in the bank that have not yet been asked.

        Checks both question ID and text against state.questions_asked to avoid repeats.

        Args:
            state: The current InterviewState.

        Returns:
            List of unasked Question objects.
        """
        asked_set = set(state.questions_asked)
        return [
            q for q in self.question_bank
            if q.id not in asked_set and q.text not in asked_set
        ]

    def select_next_question(
        self,
        state: InterviewState,
        is_follow_up: bool = False,
        topic: str | None = None,
    ) -> Question | None:
        """Select the next best question for the candidate.

        Selection strategy (deterministic multi-factor priority ranking):
            1. Unasked questions only (no repeats by ID or text).
            2. Follow-up alignment: prefers questions matching is_follow_up flag.
            3. Difficulty matching: prefers exact difficulty matches, then falls back gracefully
               to adjacent difficulties.
            4. Weak topics / Topic focus: within the target difficulty, prioritizes weak topics
               if present, requested explicit topic, or session topics.
            5. Original bank order for stable, deterministic tie-breaking.

        Args:
            state: The active InterviewState.
            is_follow_up: Whether to select a follow-up question.
            topic: Optional explicit topic constraint or preference.

        Returns:
            The selected Question, or None if all questions in the bank have been asked.
        """
        if not isinstance(state, InterviewState):
            raise ValueError(f"state must be an InterviewState instance, got {type(state).__name__}")

        unasked = self.get_unasked_questions(state)
        if not unasked:
            return None

        # Determine target difficulty index
        curr_diff = Difficulty.from_string(state.difficulty).value
        target_diff_idx = self.DIFFICULTY_INDEX.get(curr_diff, 1)

        # Normalize weak topics and session topics for matching
        normalized_weak = [w.strip().lower() for w in state.weak_topics if w.strip()]
        normalized_session_topics = [t.strip().lower() for t in state.topics if t.strip()]
        normalized_target_topic = topic.strip().lower() if topic and topic.strip() else None

        def priority_key(q: Question) -> tuple[int, int, int, int]:
            # 1. Follow-up matching (0 = exact match, 1 = mismatch fallback)
            follow_up_rank = 0 if q.follow_up == is_follow_up else 1

            # 2. Difficulty distance (0 = exact match, 1 = adjacent, 2 = opposite)
            q_diff_idx = self.DIFFICULTY_INDEX.get(q.difficulty.lower(), 1)
            diff_distance = abs(target_diff_idx - q_diff_idx)

            # 3. Topic matching & Weak topic prioritization
            q_topic_norm = q.topic.strip().lower()
            if normalized_target_topic is not None:
                # Explicit topic requested
                topic_rank = 0 if q_topic_norm == normalized_target_topic else 10
            elif normalized_weak:
                # Prioritize weak topics
                if q_topic_norm in normalized_weak:
                    topic_rank = normalized_weak.index(q_topic_norm)
                elif normalized_session_topics and q_topic_norm in normalized_session_topics:
                    topic_rank = 10 + normalized_session_topics.index(q_topic_norm)
                else:
                    topic_rank = 50
            elif normalized_session_topics:
                # Prioritize general session topics
                if q_topic_norm in normalized_session_topics:
                    topic_rank = normalized_session_topics.index(q_topic_norm)
                else:
                    topic_rank = 20
            else:
                topic_rank = 0

            # 4. Bank index for deterministic tie-breaking
            bank_index = self.question_bank.index(q) if q in self.question_bank else 0

            return (follow_up_rank, diff_distance, topic_rank, bank_index)

        sorted_candidates = sorted(unasked, key=priority_key)
        return sorted_candidates[0]

    def select_follow_up(
        self,
        state: InterviewState,
        topic: str | None = None,
    ) -> Question | None:
        """Convenience method to select a follow-up question.

        Args:
            state: The active InterviewState.
            topic: Optional explicit topic preference.

        Returns:
            The selected follow-up Question, or None if unavailable.
        """
        return self.select_next_question(state=state, is_follow_up=True, topic=topic)
