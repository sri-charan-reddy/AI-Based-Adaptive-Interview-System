"""Unit tests for Speech Analytics foundation (Task 7).

Tests SpeechAnalyticsResult model validation, serialization, dictionary subscripting,
SpeechAnalytics class word extraction, whole-word/phrase filler detection without substring false positives,
speaking rate (WPM) calculation, text-derived fluency indicators, and comprehensive error handling.
"""

from __future__ import annotations

import unittest

from adaptive_interview.speech.analytics import (
    SpeechAnalytics,
    SpeechAnalyzer,
    analyze_speech,
)
from adaptive_interview.speech.models import SpeechAnalyticsResult


class TestSpeechAnalyticsResult(unittest.TestCase):
    """Test suite for SpeechAnalyticsResult model validation, serialization, and behavior."""

    def test_default_values(self) -> None:
        """Test default values for optional attributes."""
        result = SpeechAnalyticsResult(
            word_count=50,
            filler_words_count=2,
            filler_words={"um": 1, "like": 1},
        )
        self.assertEqual(result.word_count, 50)
        self.assertEqual(result.filler_words_count, 2)
        self.assertEqual(result.filler_words, {"um": 1, "like": 1})
        self.assertIsNone(result.duration_seconds)
        self.assertIsNone(result.words_per_minute)
        self.assertEqual(result.filler_ratio, 0.0)
        self.assertIsNone(result.pace_status)
        self.assertIsNone(result.fluency_score)
        self.assertEqual(result.fluency_indicators, {})

    def test_custom_values_and_aliases(self) -> None:
        """Test explicit custom values and convenience property aliases."""
        indicators = {"filler_ratio": 0.04, "pace_status": "moderate"}
        result = SpeechAnalyticsResult(
            word_count=100,
            filler_words_count=4,
            filler_words={"um": 2, "you know": 2},
            duration_seconds=45.0,
            words_per_minute=133.33,
            filler_ratio=0.04,
            pace_status="moderate",
            fluency_score=92.0,
            fluency_indicators=indicators,
        )
        self.assertEqual(result.word_count, 100)
        self.assertEqual(result.filler_words_count, 4)
        self.assertEqual(result.filler_count, 4)  # property alias
        self.assertEqual(result.duration_seconds, 45.0)
        self.assertEqual(result.words_per_minute, 133.33)
        self.assertEqual(result.wpm, 133.33)  # property alias
        self.assertEqual(result.filler_ratio, 0.04)
        self.assertEqual(result.pace_status, "moderate")
        self.assertEqual(result.fluency_score, 92.0)
        self.assertEqual(result.fluency_indicators, indicators)

    def test_validation_errors(self) -> None:
        """Test validation rejects invalid word counts, filler counts, durations, and scores."""
        # Negative word count
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=-1, filler_words_count=0, filler_words={})

        # Boolean word count
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=True, filler_words_count=0, filler_words={})  # type: ignore

        # Negative filler count
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=-1, filler_words={})

        # Non-dict filler_words
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words=["um"])  # type: ignore

        # Negative filler ratio
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, filler_ratio=-0.1)

        # Invalid duration (zero or negative)
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, duration_seconds=0)

        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, duration_seconds=-5.0)

        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, duration_seconds=True)  # type: ignore

        # Invalid score (< 0 or > 100)
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, fluency_score=-1.0)

        with self.assertRaises(ValueError):
            SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={}, fluency_score=105.0)

    def test_dict_style_access(self) -> None:
        """Test subscript access via result['field']."""
        result = SpeechAnalyticsResult(
            word_count=75,
            filler_words_count=3,
            filler_words={"like": 3},
            duration_seconds=30.0,
            words_per_minute=150.0,
        )
        self.assertEqual(result["word_count"], 75)
        self.assertEqual(result["filler_words_count"], 3)
        self.assertEqual(result["filler_words"], {"like": 3})
        self.assertEqual(result["duration_seconds"], 30.0)
        self.assertEqual(result["words_per_minute"], 150.0)

    def test_dict_style_access_invalid_key_raises_key_error(self) -> None:
        """Test accessing non-existent key raises KeyError."""
        result = SpeechAnalyticsResult(word_count=10, filler_words_count=0, filler_words={})
        with self.assertRaises(KeyError):
            _ = result["non_existent_field"]

    def test_serialization_to_dict(self) -> None:
        """Test serializing result to dictionary."""
        result = SpeechAnalyticsResult(
            word_count=40,
            filler_words_count=1,
            filler_words={"um": 1},
            duration_seconds=20.0,
            words_per_minute=120.0,
            filler_ratio=0.025,
            pace_status="moderate",
            fluency_score=95.0,
            fluency_indicators={"filler_ratio": 0.025, "pace_status": "moderate"},
        )
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["word_count"], 40)
        self.assertEqual(data["filler_words_count"], 1)
        self.assertEqual(data["filler_words"], {"um": 1})
        self.assertEqual(data["words_per_minute"], 120.0)

    def test_deserialization_from_dict(self) -> None:
        """Test deserializing result from dictionary."""
        data = {
            "word_count": 80,
            "filler_words_count": 2,
            "filler_words": {"uh": 2},
            "duration_seconds": 40.0,
            "words_per_minute": 120.0,
            "filler_ratio": 0.025,
            "pace_status": "moderate",
            "fluency_score": 95.0,
            "fluency_indicators": {"pace_status": "moderate"},
        }
        result = SpeechAnalyticsResult.from_dict(data)
        self.assertEqual(result.word_count, 80)
        self.assertEqual(result.filler_words_count, 2)
        self.assertEqual(result.filler_words, {"uh": 2})
        self.assertEqual(result.duration_seconds, 40.0)

    def test_from_dict_missing_required_keys(self) -> None:
        """Test from_dict raises KeyError when required keys are missing."""
        with self.assertRaises(KeyError):
            SpeechAnalyticsResult.from_dict({"filler_words_count": 0, "filler_words": {}})

        with self.assertRaises(KeyError):
            SpeechAnalyticsResult.from_dict({"word_count": 10, "filler_words": {}})

        with self.assertRaises(KeyError):
            SpeechAnalyticsResult.from_dict({"word_count": 10, "filler_words_count": 0})

    def test_from_dict_non_dict_raises_value_error(self) -> None:
        """Test from_dict raises ValueError if input is not a dictionary."""
        with self.assertRaises(ValueError):
            SpeechAnalyticsResult.from_dict("invalid")  # type: ignore

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict and from_dict preserve equality identically."""
        original = SpeechAnalyticsResult(
            word_count=60,
            filler_words_count=3,
            filler_words={"um": 2, "erm": 1},
            duration_seconds=30.0,
            words_per_minute=120.0,
            filler_ratio=0.05,
            pace_status="moderate",
            fluency_score=90.0,
            fluency_indicators={"filler_ratio": 0.05, "pace_status": "moderate"},
        )
        reconstructed = SpeechAnalyticsResult.from_dict(original.to_dict())
        self.assertEqual(original, reconstructed)


class TestSpeechAnalyticsModule(unittest.TestCase):
    """Test suite for SpeechAnalytics word counting, filler matching, pace, and fluency logic."""

    def setUp(self) -> None:
        """Create fresh analyzer instance."""
        self.analyzer = SpeechAnalytics()

    # 1. Word Count Tests
    def test_basic_word_count(self) -> None:
        """Test word count with standard conversational sentences."""
        text = "I have five years of backend development experience using Python."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.word_count, 10)

    def test_word_count_with_contractions_and_hyphens(self) -> None:
        """Test word count retains contractions and hyphenated words appropriately."""
        text = "It's a state-of-the-art microservice, and I'm very proud of it."
        result = self.analyzer.analyze(text)
        # "It's", "a", "state-of-the-art", "microservice", "and", "I'm", "very", "proud", "of", "it" (10 words)
        self.assertEqual(result.word_count, 10)

    def test_word_count_with_numbers(self) -> None:
        """Test word count counts numbers as words."""
        text = "We scaled to 5000 users across 3 regions in 2025."
        result = self.analyzer.analyze(text)
        # "We", "scaled", "to", "5000", "users", "across", "3", "regions", "in", "2025" (10 words)
        self.assertEqual(result.word_count, 10)

    def test_word_count_with_heavy_punctuation(self) -> None:
        """Test word count cleans punctuation correctly."""
        text = "Python, C++, Java... Docker; and Kubernetes! Right?"
        result = self.analyzer.analyze(text)
        self.assertEqual(result.word_count, 7)

    # 2. Filler Word Detection Tests
    def test_single_filler_words_detected(self) -> None:
        """Test detecting individual single-word fillers: um, uh, erm, like."""
        text = "Um, I think uh, it was erm, like, very challenging."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 4)
        self.assertEqual(result.filler_words.get("um"), 1)
        self.assertEqual(result.filler_words.get("uh"), 1)
        self.assertEqual(result.filler_words.get("erm"), 1)
        self.assertEqual(result.filler_words.get("like"), 1)

    def test_multi_word_filler_phrase_you_know(self) -> None:
        """Test detecting multi-word filler phrase 'you know'."""
        text = "The system was, you know, experiencing high concurrency."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 1)
        self.assertEqual(result.filler_words.get("you know"), 1)

    def test_filler_case_insensitivity(self) -> None:
        """Test filler matching is case-insensitive across uppercase, lowercase, and mixed case."""
        text = "UM, I LIKE Python, and YOU KNOW, UH, ERM, it's great."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words.get("um"), 1)
        self.assertEqual(result.filler_words.get("like"), 1)
        self.assertEqual(result.filler_words.get("you know"), 1)
        self.assertEqual(result.filler_words.get("uh"), 1)
        self.assertEqual(result.filler_words.get("erm"), 1)
        self.assertEqual(result.filler_words_count, 5)

    def test_repeated_fillers_counted_correctly(self) -> None:
        """Test multiple occurrences of the same filler are counted accurately."""
        text = "Um, I wanted to say... um, well, um, it worked."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words.get("um"), 3)
        self.assertEqual(result.filler_words_count, 3)

    # 3. Substring Avoidance Tests (Critical)
    def test_substring_avoidance_um(self) -> None:
        """Verify 'um' is NOT matched as substring inside 'umbrella', 'forum', 'maximum'."""
        text = "Under the umbrella in the public forum, we reached maximum efficiency."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 0)
        self.assertEqual(result.filler_words, {})

    def test_substring_avoidance_like(self) -> None:
        """Verify 'like' is NOT matched as substring inside 'unlike', 'likely', 'dislike'."""
        text = "Unlike other tools, this is likely to succeed without any dislike."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 0)
        self.assertEqual(result.filler_words, {})

    def test_substring_avoidance_erm(self) -> None:
        """Verify 'erm' is NOT matched as substring inside 'germany', 'determine', 'permission'."""
        text = "We will determine permission in Germany."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 0)
        self.assertEqual(result.filler_words, {})

    def test_substring_avoidance_uh(self) -> None:
        """Verify 'uh' is NOT matched inside 'uh-huh'."""
        text = "Uh-huh, I agree."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words.get("uh", 0), 0)

    def test_filler_alongside_similar_substrings(self) -> None:
        """Verify legitimate filler is counted even when substring word appears in same sentence."""
        text = "Um, under the umbrella, like, unlike others, I was uh, happy."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words.get("um"), 1)
        self.assertEqual(result.filler_words.get("like"), 1)
        self.assertEqual(result.filler_words.get("uh"), 1)
        self.assertEqual(result.filler_words_count, 3)

    def test_clean_text_zero_fillers(self) -> None:
        """Test clean text with no fillers returns 0 filler count and 0 ratio."""
        text = "Object-oriented programming utilizes classes, encapsulation, and polymorphism."
        result = self.analyzer.analyze(text)
        self.assertEqual(result.filler_words_count, 0)
        self.assertEqual(result.filler_words, {})
        self.assertEqual(result.filler_ratio, 0.0)

    # 4. Speaking Rate & Duration Tests
    def test_speaking_rate_with_valid_duration(self) -> None:
        """Test speaking rate (WPM) calculation when duration is provided."""
        # 30 words in 15 seconds -> 120 WPM
        text = "one two three four five six seven eight nine ten " * 3
        result = self.analyzer.analyze(text, duration_seconds=15.0)
        self.assertEqual(result.word_count, 30)
        self.assertEqual(result.duration_seconds, 15.0)
        self.assertEqual(result.words_per_minute, 120.0)
        self.assertEqual(result.pace_status, "moderate")

    def test_speaking_rate_without_duration(self) -> None:
        """Test speaking rate is None and pace is unknown when duration is omitted."""
        text = "A simple answer with no audio duration provided."
        result = self.analyzer.analyze(text, duration_seconds=None)
        self.assertIsNone(result.duration_seconds)
        self.assertIsNone(result.words_per_minute)
        self.assertEqual(result.pace_status, "unknown")

    def test_pace_categories(self) -> None:
        """Test pace categorization for slow (<110), moderate (110-165), and fast (>165) speech."""
        text = "word " * 100  # 100 words

        # 100 words in 75 seconds -> 80.0 WPM (slow)
        slow_result = self.analyzer.analyze(text, duration_seconds=75.0)
        self.assertEqual(slow_result.pace_status, "slow")

        # 100 words in 45 seconds -> 133.33 WPM (moderate)
        mod_result = self.analyzer.analyze(text, duration_seconds=45.0)
        self.assertEqual(mod_result.pace_status, "moderate")

        # 100 words in 30 seconds -> 200.0 WPM (fast)
        fast_result = self.analyzer.analyze(text, duration_seconds=30.0)
        self.assertEqual(fast_result.pace_status, "fast")

    # 5. Fluency Indicators Tests
    def test_fluency_indicators_structure(self) -> None:
        """Test that fluency_indicators dictionary contains expected metrics."""
        text = "Um, I think this is like a moderate pace response."
        result = self.analyzer.analyze(text, duration_seconds=5.0)
        indicators = result.fluency_indicators
        self.assertIn("filler_ratio", indicators)
        self.assertIn("filler_density_status", indicators)
        self.assertIn("pace_status", indicators)
        self.assertIn("fluency_score", indicators)
        self.assertIn("words_per_minute", indicators)
        self.assertIn("duration_seconds", indicators)
        self.assertGreaterEqual(result.fluency_score, 0.0)
        self.assertLessEqual(result.fluency_score, 100.0)

    # 6. Custom Filler Words Configuration
    def test_custom_filler_words_configuration(self) -> None:
        """Test instantiating analyzer with custom filler list."""
        custom_fillers = ["actually", "sort of", "basically"]
        analyzer = SpeechAnalytics(filler_words=custom_fillers)
        text = "Actually, I sort of basically know Python."
        result = analyzer.analyze(text)
        self.assertEqual(result.filler_words.get("actually"), 1)
        self.assertEqual(result.filler_words.get("sort of"), 1)
        self.assertEqual(result.filler_words.get("basically"), 1)
        self.assertEqual(result.filler_words_count, 3)

    def test_invalid_custom_filler_words_raises_value_error(self) -> None:
        """Test passing invalid filler word collections raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechAnalytics(filler_words=["valid", "", "another"])

        with self.assertRaises(ValueError):
            SpeechAnalytics(filler_words=["valid", 123])  # type: ignore

        with self.assertRaises(ValueError):
            SpeechAnalytics(filler_words="not a list or tuple")  # type: ignore

    # 7. Input Validation & Error Handling Tests
    def test_invalid_text_inputs_raise_value_error(self) -> None:
        """Test None, empty string, whitespace-only, non-string, or symbol-only text raises ValueError."""
        with self.assertRaises(ValueError):
            self.analyzer.analyze("")

        with self.assertRaises(ValueError):
            self.analyzer.analyze("   ")

        with self.assertRaises(ValueError):
            self.analyzer.analyze(None)  # type: ignore

        with self.assertRaises(ValueError):
            self.analyzer.analyze(12345)  # type: ignore

        with self.assertRaises(ValueError):
            self.analyzer.analyze("... ,,, ??? !!!")

    def test_invalid_duration_inputs_raise_value_error(self) -> None:
        """Test non-positive, boolean, or non-numeric duration raises ValueError."""
        with self.assertRaises(ValueError):
            self.analyzer.analyze("Valid speech text.", duration_seconds=0)

        with self.assertRaises(ValueError):
            self.analyzer.analyze("Valid speech text.", duration_seconds=-10.0)

        with self.assertRaises(ValueError):
            self.analyzer.analyze("Valid speech text.", duration_seconds=True)  # type: ignore

        with self.assertRaises(ValueError):
            self.analyzer.analyze("Valid speech text.", duration_seconds="fast")  # type: ignore

    # 8. Module-Level Convenience Functions and Aliases
    def test_analyze_speech_helper_function(self) -> None:
        """Test top-level analyze_speech() helper function works seamlessly."""
        text = "Um, this is a test using the standalone analyze_speech function."
        result = analyze_speech(text, duration_seconds=5.0)
        self.assertIsInstance(result, SpeechAnalyticsResult)
        self.assertEqual(result.filler_words.get("um"), 1)
        self.assertEqual(result.word_count, 10)
        self.assertAlmostEqual(result.words_per_minute, 120.0, delta=0.1)

    def test_speech_analyzer_alias(self) -> None:
        """Test SpeechAnalyzer alias works identically to SpeechAnalytics."""
        self.assertIs(SpeechAnalyzer, SpeechAnalytics)
        analyzer = SpeechAnalyzer()
        res = analyzer.analyze("Quick response test.")
        self.assertEqual(res.word_count, 3)


if __name__ == "__main__":
    unittest.main()
