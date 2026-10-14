"""Unit tests for Text-to-Speech foundation (Task 6).

Tests SpeechSynthesisResult model validation, serialization, dictionary access,
TextToSpeechProvider abstract base class, and Pyttsx3TextToSpeech provider including
lazy engine initialization, input/path validation, mocked speech synthesis, property handling,
and error reporting.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from adaptive_interview.speech.base import TextToSpeechProvider
from adaptive_interview.speech.models import SpeechSynthesisResult
from adaptive_interview.speech.pyttsx_tts import Pyttsx3TextToSpeech


class TestSpeechSynthesisResult(unittest.TestCase):
    """Test suite for SpeechSynthesisResult dataclass validation, serialization, and behavior."""

    def test_default_values(self) -> None:
        """Test default values for optional fields and default provider."""
        result = SpeechSynthesisResult(text="Welcome to the interview.", audio_path="output.wav")
        self.assertEqual(result.text, "Welcome to the interview.")
        self.assertEqual(result.audio_path, "output.wav")
        self.assertEqual(result.provider, "pyttsx3")
        self.assertIsNone(result.duration_seconds)
        self.assertIsNone(result.voice)

    def test_custom_values(self) -> None:
        """Test initialization with explicit custom attributes."""
        result = SpeechSynthesisResult(
            text="Can you describe your experience with Python?",
            audio_path="q1.wav",
            provider="custom_tts",
            duration_seconds=3.45,
            voice="english_voice_1",
        )
        self.assertEqual(result.text, "Can you describe your experience with Python?")
        self.assertEqual(result.audio_path, "q1.wav")
        self.assertEqual(result.provider, "custom_tts")
        self.assertEqual(result.duration_seconds, 3.45)
        self.assertEqual(result.voice, "english_voice_1")

    def test_whitespace_normalization(self) -> None:
        """Test that text, audio_path, provider, and voice are stripped of surrounding whitespace."""
        result = SpeechSynthesisResult(
            text="   Trimmed question text.   ",
            audio_path="  output.wav  ",
            provider="  pyttsx3  ",
            voice="  voice_id  ",
        )
        self.assertEqual(result.text, "Trimmed question text.")
        self.assertEqual(result.audio_path, "output.wav")
        self.assertEqual(result.provider, "pyttsx3")
        self.assertEqual(result.voice, "voice_id")

    def test_validation_invalid_text(self) -> None:
        """Test that empty, whitespace-only, non-string, or None text raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text=None, audio_path="out.wav")  # type: ignore

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="", audio_path="out.wav")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="    ", audio_path="out.wav")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text=123, audio_path="out.wav")  # type: ignore

    def test_validation_invalid_audio_path(self) -> None:
        """Test that empty, whitespace-only, non-string, or None audio_path raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path=None)  # type: ignore

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="   ")

    def test_validation_empty_provider(self) -> None:
        """Test that empty or whitespace-only provider raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", provider="")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", provider="   ")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", provider=None)  # type: ignore

    def test_validation_invalid_duration(self) -> None:
        """Test that negative, boolean, or non-numeric duration_seconds raise ValueError."""
        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", duration_seconds=-1.0)

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", duration_seconds=True)  # type: ignore

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", duration_seconds="invalid")  # type: ignore

    def test_validation_invalid_voice(self) -> None:
        """Test that empty or non-string voice raises ValueError."""
        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", voice="")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", voice="   ")

        with self.assertRaises(ValueError):
            SpeechSynthesisResult(text="Valid text", audio_path="out.wav", voice=123)  # type: ignore

    def test_dict_style_access(self) -> None:
        """Test dictionary-style subscript access (result['field'])."""
        result = SpeechSynthesisResult(
            text="Dictionary access test",
            audio_path="out.wav",
            provider="pyttsx3",
            duration_seconds=2.1,
            voice="default_voice",
        )
        self.assertEqual(result["text"], "Dictionary access test")
        self.assertEqual(result["audio_path"], "out.wav")
        self.assertEqual(result["provider"], "pyttsx3")
        self.assertEqual(result["duration_seconds"], 2.1)
        self.assertEqual(result["voice"], "default_voice")

    def test_dict_style_access_invalid_key(self) -> None:
        """Test accessing non-existent key via subscript raises KeyError."""
        result = SpeechSynthesisResult(text="Test", audio_path="out.wav")
        with self.assertRaises(KeyError):
            _ = result["non_existent_key"]

    def test_serialization_to_dict(self) -> None:
        """Test serializing SpeechSynthesisResult to dictionary."""
        result = SpeechSynthesisResult(
            text="Serialization test",
            audio_path="sample.wav",
            provider="pyttsx3",
            duration_seconds=1.8,
            voice="en_us",
        )
        expected = {
            "text": "Serialization test",
            "audio_path": "sample.wav",
            "provider": "pyttsx3",
            "duration_seconds": 1.8,
            "voice": "en_us",
        }
        self.assertEqual(result.to_dict(), expected)

    def test_deserialization_from_dict(self) -> None:
        """Test reconstructing SpeechSynthesisResult from a dictionary."""
        data = {
            "text": "Reconstruction test",
            "audio_path": "reconstruct.wav",
            "provider": "pyttsx3",
            "duration_seconds": 3.0,
            "voice": "voice_1",
        }
        result = SpeechSynthesisResult.from_dict(data)
        self.assertEqual(result.text, "Reconstruction test")
        self.assertEqual(result.audio_path, "reconstruct.wav")
        self.assertEqual(result.provider, "pyttsx3")
        self.assertEqual(result.duration_seconds, 3.0)
        self.assertEqual(result.voice, "voice_1")

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with optional fields omitted."""
        data = {"text": "Minimal TTS test", "audio_path": "minimal.wav"}
        result = SpeechSynthesisResult.from_dict(data)
        self.assertEqual(result.text, "Minimal TTS test")
        self.assertEqual(result.audio_path, "minimal.wav")
        self.assertIsNone(result.duration_seconds)
        self.assertIsNone(result.voice)
        self.assertEqual(result.provider, "pyttsx3")

    def test_from_dict_errors(self) -> None:
        """Test from_dict error cases for missing keys and invalid data types."""
        with self.assertRaises(KeyError):
            SpeechSynthesisResult.from_dict({"audio_path": "out.wav"})

        with self.assertRaises(KeyError):
            SpeechSynthesisResult.from_dict({"text": "Hello"})

        with self.assertRaises(ValueError):
            SpeechSynthesisResult.from_dict("not a dict")  # type: ignore

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict and from_dict preserve all properties identically."""
        original = SpeechSynthesisResult(
            text="Roundtrip TTS test",
            audio_path="roundtrip.wav",
            provider="pyttsx3",
            duration_seconds=5.2,
            voice="voice_test",
        )
        reconstructed = SpeechSynthesisResult.from_dict(original.to_dict())
        self.assertEqual(original, reconstructed)


class TestTextToSpeechProviderBase(unittest.TestCase):
    """Test suite for the abstract TextToSpeechProvider base class."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        """Test that TextToSpeechProvider cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            TextToSpeechProvider()  # type: ignore

    def test_concrete_subclass_instantiation(self) -> None:
        """Test that a subclass implementing synthesize() can be instantiated."""

        class DummyTTS(TextToSpeechProvider):
            def synthesize(self, text: str, output_path: str | Path, **kwargs) -> SpeechSynthesisResult:
                return SpeechSynthesisResult(text=text, audio_path=str(output_path), provider="dummy")

        provider = DummyTTS()
        result = provider.synthesize("test prompt", "out.wav")
        self.assertEqual(result.text, "test prompt")
        self.assertEqual(result.audio_path, "out.wav")
        self.assertEqual(result.provider, "dummy")


class TestPyttsx3TextToSpeech(unittest.TestCase):
    """Test suite for Pyttsx3TextToSpeech provider implementation."""

    def test_default_configuration(self) -> None:
        """Test default initialization values."""
        tts = Pyttsx3TextToSpeech()
        self.assertIsNone(tts.voice)
        self.assertIsNone(tts.rate)
        self.assertIsNone(tts.volume)
        self.assertIsNone(tts.driver_name)
        self.assertFalse(tts.is_engine_initialized)

    def test_custom_configuration(self) -> None:
        """Test initialization with custom parameters."""
        tts = Pyttsx3TextToSpeech(voice="david", rate=175, volume=0.85, driver_name="sapi5")
        self.assertEqual(tts.voice, "david")
        self.assertEqual(tts.rate, 175)
        self.assertEqual(tts.volume, 0.85)
        self.assertEqual(tts.driver_name, "sapi5")
        self.assertFalse(tts.is_engine_initialized)

    def test_validation_invalid_rate(self) -> None:
        """Test invalid rate values raise ValueError."""
        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(rate=-100)

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(rate=0)

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(rate="fast")  # type: ignore

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(rate=True)  # type: ignore

    def test_validation_invalid_volume(self) -> None:
        """Test invalid volume values raise ValueError."""
        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(volume=-0.1)

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(volume=1.1)

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(volume="loud")  # type: ignore

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(volume=True)  # type: ignore

    def test_validation_invalid_voice(self) -> None:
        """Test invalid voice values raise ValueError."""
        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(voice="")

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(voice="   ")

        with self.assertRaises(ValueError):
            Pyttsx3TextToSpeech(voice=123)  # type: ignore

    def test_lazy_loading_not_initialized_on_init(self) -> None:
        """Verify engine is NOT initialized when Pyttsx3TextToSpeech is instantiated."""
        tts = Pyttsx3TextToSpeech(rate=150)
        self.assertFalse(tts.is_engine_initialized)
        self.assertIsNone(tts._engine)

    def test_injected_engine_initialization(self) -> None:
        """Test dependency injection of a mock pyttsx3 engine."""
        mock_engine = MagicMock()
        tts = Pyttsx3TextToSpeech(engine=mock_engine)
        self.assertTrue(tts.is_engine_initialized)
        self.assertIs(tts.load_engine(), mock_engine)

    # Input & Output Validation Tests
    def test_empty_text_raises_value_error(self) -> None:
        """Test empty text raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with self.assertRaises(ValueError):
            tts.synthesize("", "out.wav")

        with self.assertRaises(ValueError):
            tts.synthesize("   ", "out.wav")

    def test_non_string_text_raises_value_error(self) -> None:
        """Test non-string text raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with self.assertRaises(ValueError):
            tts.synthesize(None, "out.wav")  # type: ignore

        with self.assertRaises(ValueError):
            tts.synthesize(12345, "out.wav")  # type: ignore

    def test_empty_output_path_raises_value_error(self) -> None:
        """Test empty output path raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with self.assertRaises(ValueError):
            tts.synthesize("Hello", "")

        with self.assertRaises(ValueError):
            tts.synthesize("Hello", "   ")

    def test_invalid_output_path_type_raises_value_error(self) -> None:
        """Test non-string/non-Path output path raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with self.assertRaises(ValueError):
            tts.synthesize("Hello", None)  # type: ignore

        with self.assertRaises(ValueError):
            tts.synthesize("Hello", 999)  # type: ignore

    def test_directory_output_path_raises_value_error(self) -> None:
        """Test output path pointing to an existing directory raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError) as ctx:
                tts.synthesize("Hello", temp_dir)
            self.assertIn("directory", str(ctx.exception).lower())

    def test_non_wav_extension_raises_value_error(self) -> None:
        """Test output file with non-.wav extension raises ValueError."""
        tts = Pyttsx3TextToSpeech()
        with self.assertRaises(ValueError) as ctx:
            tts.synthesize("Hello", "output.mp3")
        self.assertIn(".wav", str(ctx.exception).lower())

        with self.assertRaises(ValueError):
            tts.synthesize("Hello", "output.txt")

    def test_pathlib_output_path_accepted(self) -> None:
        """Test pathlib.Path instance is accepted during output validation."""
        mock_engine = MagicMock()
        tts = Pyttsx3TextToSpeech(engine=mock_engine)
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "test_question.wav"
            result = tts.synthesize("What is recursion?", target_path)
            self.assertEqual(result.text, "What is recursion?")
            self.assertEqual(result.audio_path, str(target_path))
            mock_engine.save_to_file.assert_called_once_with("What is recursion?", str(target_path))
            mock_engine.runAndWait.assert_called_once()

    # Pyttsx3 Package & Engine Loading Errors
    def test_pyttsx3_not_installed_raises_runtime_error(self) -> None:
        """Test useful error message when pyttsx3 package is not installed."""
        with patch.dict("sys.modules", {"pyttsx3": None}):
            tts = Pyttsx3TextToSpeech()
            with self.assertRaises(RuntimeError) as ctx:
                tts.load_engine()
            err_msg = str(ctx.exception).lower()
            self.assertIn("pyttsx3", err_msg)
            self.assertIn("not installed", err_msg)

    def test_engine_initialization_failure_raises_runtime_error(self) -> None:
        """Test failure during pyttsx3.init raises RuntimeError."""
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = RuntimeError("COM SAPI5 failure")
        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            tts = Pyttsx3TextToSpeech()
            with self.assertRaises(RuntimeError) as ctx:
                tts.load_engine()
            self.assertIn("Failed to initialize pyttsx3 engine", str(ctx.exception))

    # Mocked Successful Synthesis Tests
    def test_synthesize_success_calls_engine_and_returns_result(self) -> None:
        """Test successful synthesis invokes engine.save_to_file and runAndWait."""
        mock_engine = MagicMock()
        tts = Pyttsx3TextToSpeech(rate=160, volume=0.9, voice="english_female", engine=mock_engine)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "output.wav"
            result = tts.synthesize("Explain object-oriented programming concepts.", output_file)

            self.assertIsInstance(result, SpeechSynthesisResult)
            self.assertEqual(result.text, "Explain object-oriented programming concepts.")
            self.assertEqual(result.audio_path, str(output_file))
            self.assertEqual(result.provider, "pyttsx3")
            self.assertEqual(result.voice, "english_female")

            mock_engine.setProperty.assert_any_call("rate", 160)
            mock_engine.setProperty.assert_any_call("volume", 0.9)
            mock_engine.setProperty.assert_any_call("voice", "english_female")
            mock_engine.save_to_file.assert_called_once_with(
                "Explain object-oriented programming concepts.", str(output_file)
            )
            mock_engine.runAndWait.assert_called_once()

    def test_synthesize_with_runtime_property_overrides(self) -> None:
        """Test that rate, volume, and voice can be overridden per synthesis call."""
        mock_engine = MagicMock()
        tts = Pyttsx3TextToSpeech(rate=150, volume=0.5, voice="base_voice", engine=mock_engine)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "custom.wav"
            result = tts.synthesize(
                "Fast speech test.",
                output_file,
                rate=220,
                volume=1.0,
                voice="overridden_voice",
            )

            self.assertEqual(result.voice, "overridden_voice")
            mock_engine.setProperty.assert_any_call("rate", 220)
            mock_engine.setProperty.assert_any_call("volume", 1.0)
            mock_engine.setProperty.assert_any_call("voice", "overridden_voice")

    def test_synthesize_creates_parent_directories_if_missing(self) -> None:
        """Test that missing parent directories are created automatically."""
        mock_engine = MagicMock()
        tts = Pyttsx3TextToSpeech(engine=mock_engine)

        with tempfile.TemporaryDirectory() as temp_dir:
            nested_output = Path(temp_dir) / "nested" / "subfolder" / "speech.wav"
            self.assertFalse(nested_output.parent.exists())

            result = tts.synthesize("Directory creation test.", nested_output)
            self.assertTrue(nested_output.parent.exists())
            self.assertEqual(result.audio_path, str(nested_output))

    # Mocked Synthesis Failure Tests
    def test_synthesize_failure_raises_runtime_error(self) -> None:
        """Test that an exception during engine execution raises RuntimeError with clear message."""
        mock_engine = MagicMock()
        mock_engine.save_to_file.side_effect = RuntimeError("Disk write error")

        tts = Pyttsx3TextToSpeech(engine=mock_engine)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "fail.wav"
            with self.assertRaises(RuntimeError) as ctx:
                tts.synthesize("This will fail.", output_file)
            self.assertIn("Speech synthesis failed", str(ctx.exception))
            self.assertIn("Disk write error", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
