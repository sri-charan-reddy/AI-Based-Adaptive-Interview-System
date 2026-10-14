"""Unit tests for Speech-to-Text foundation (Task 5).

Tests TranscriptionResult model validation, serialization, dictionary access,
SpeechToTextProvider abstract base class, and WhisperSpeechToText provider including
lazy loading, audio path validation, mocked transcription, and error handling.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from adaptive_interview.speech.base import SpeechToTextProvider
from adaptive_interview.speech.models import TranscriptionResult
from adaptive_interview.speech.whisper_stt import WhisperSpeechToText


class TestTranscriptionResult(unittest.TestCase):
    """Test suite for TranscriptionResult dataclass validation, serialization, and behavior."""

    def test_default_values(self) -> None:
        """Test default values for optional fields and default provider."""
        result = TranscriptionResult(text="Hello world")
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.provider, "whisper")
        self.assertIsNone(result.language)
        self.assertIsNone(result.duration_seconds)

    def test_custom_values(self) -> None:
        """Test initialization with explicit custom attributes."""
        result = TranscriptionResult(
            text="Experienced software engineer with Python expertise.",
            language="en",
            duration_seconds=5.75,
            provider="custom_whisper",
        )
        self.assertEqual(result.text, "Experienced software engineer with Python expertise.")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.duration_seconds, 5.75)
        self.assertEqual(result.provider, "custom_whisper")

    def test_whitespace_normalization(self) -> None:
        """Test that text, provider, and language are stripped of surrounding whitespace."""
        result = TranscriptionResult(
            text="   Clean transcription.   ",
            language="  en  ",
            provider="  whisper  ",
        )
        self.assertEqual(result.text, "Clean transcription.")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.provider, "whisper")

    def test_validation_invalid_text_type(self) -> None:
        """Test that non-string or None text raises ValueError."""
        with self.assertRaises(ValueError):
            TranscriptionResult(text=None)  # type: ignore

        with self.assertRaises(ValueError):
            TranscriptionResult(text=123)  # type: ignore

    def test_validation_empty_provider(self) -> None:
        """Test that empty or whitespace-only provider raises ValueError."""
        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", provider="")

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", provider="   ")

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", provider=None)  # type: ignore

    def test_validation_invalid_language(self) -> None:
        """Test that invalid language values raise ValueError."""
        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", language="")

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", language="   ")

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", language=123)  # type: ignore

    def test_validation_invalid_duration(self) -> None:
        """Test that negative, boolean, or non-numeric duration_seconds raise ValueError."""
        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", duration_seconds=-1.0)

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", duration_seconds=True)  # type: ignore

        with self.assertRaises(ValueError):
            TranscriptionResult(text="Valid text", duration_seconds="invalid")  # type: ignore

    def test_dict_style_access(self) -> None:
        """Test dictionary-style subscript access (result['field'])."""
        result = TranscriptionResult(
            text="Dictionary access test",
            language="es",
            duration_seconds=3.2,
            provider="whisper",
        )
        self.assertEqual(result["text"], "Dictionary access test")
        self.assertEqual(result["language"], "es")
        self.assertEqual(result["duration_seconds"], 3.2)
        self.assertEqual(result["provider"], "whisper")

    def test_dict_style_access_invalid_key(self) -> None:
        """Test accessing non-existent key via subscript raises KeyError."""
        result = TranscriptionResult(text="Test")
        with self.assertRaises(KeyError):
            _ = result["non_existent_key"]

    def test_serialization_to_dict(self) -> None:
        """Test serializing TranscriptionResult to dictionary."""
        result = TranscriptionResult(
            text="Serialization test",
            language="de",
            duration_seconds=4.0,
            provider="whisper",
        )
        expected = {
            "text": "Serialization test",
            "language": "de",
            "duration_seconds": 4.0,
            "provider": "whisper",
        }
        self.assertEqual(result.to_dict(), expected)

    def test_deserialization_from_dict(self) -> None:
        """Test reconstructing TranscriptionResult from a dictionary."""
        data = {
            "text": "Reconstruction test",
            "language": "fr",
            "duration_seconds": 2.5,
            "provider": "whisper",
        }
        result = TranscriptionResult.from_dict(data)
        self.assertEqual(result.text, "Reconstruction test")
        self.assertEqual(result.language, "fr")
        self.assertEqual(result.duration_seconds, 2.5)
        self.assertEqual(result.provider, "whisper")

    def test_from_dict_defaults(self) -> None:
        """Test from_dict with optional fields omitted."""
        data = {"text": "Minimal test"}
        result = TranscriptionResult.from_dict(data)
        self.assertEqual(result.text, "Minimal test")
        self.assertIsNone(result.language)
        self.assertIsNone(result.duration_seconds)
        self.assertEqual(result.provider, "whisper")

    def test_from_dict_errors(self) -> None:
        """Test from_dict error cases for missing keys and invalid data types."""
        with self.assertRaises(KeyError):
            TranscriptionResult.from_dict({"language": "en"})

        with self.assertRaises(ValueError):
            TranscriptionResult.from_dict("not a dict")  # type: ignore

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict and from_dict preserve all properties identically."""
        original = TranscriptionResult(
            text="Roundtrip test",
            language="ja",
            duration_seconds=8.1,
            provider="whisper",
        )
        reconstructed = TranscriptionResult.from_dict(original.to_dict())
        self.assertEqual(original, reconstructed)


class TestSpeechToTextProviderBase(unittest.TestCase):
    """Test suite for the abstract SpeechToTextProvider base class."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        """Test that SpeechToTextProvider cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            SpeechToTextProvider()  # type: ignore

    def test_concrete_subclass_instantiation(self) -> None:
        """Test that a subclass implementing transcribe() can be instantiated."""

        class DummyProvider(SpeechToTextProvider):
            def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
                return TranscriptionResult(text="dummy", provider="dummy")

        provider = DummyProvider()
        result = provider.transcribe("sample.wav")
        self.assertEqual(result.text, "dummy")
        self.assertEqual(result.provider, "dummy")


class TestWhisperSpeechToText(unittest.TestCase):
    """Test suite for WhisperSpeechToText provider implementation."""

    def test_default_model_name(self) -> None:
        """Test default model name is 'base'."""
        stt = WhisperSpeechToText()
        self.assertEqual(stt.model_name, "base")
        self.assertFalse(stt.is_model_loaded)
        self.assertIsNone(stt.device)

    def test_custom_model_and_device(self) -> None:
        """Test initializing with custom model name and device."""
        stt = WhisperSpeechToText(model_name="tiny", device="cpu")
        self.assertEqual(stt.model_name, "tiny")
        self.assertEqual(stt.device, "cpu")
        self.assertFalse(stt.is_model_loaded)

    def test_invalid_model_name_raises_value_error(self) -> None:
        """Test invalid model_name values raise ValueError."""
        with self.assertRaises(ValueError):
            WhisperSpeechToText(model_name="")

        with self.assertRaises(ValueError):
            WhisperSpeechToText(model_name="   ")

        with self.assertRaises(ValueError):
            WhisperSpeechToText(model_name=None)  # type: ignore

    def test_lazy_loading_not_loaded_on_init(self) -> None:
        """Verify model is NOT loaded when WhisperSpeechToText is instantiated."""
        stt = WhisperSpeechToText(model_name="tiny")
        self.assertFalse(stt.is_model_loaded)
        self.assertIsNone(stt._model)

    def test_injected_model_init(self) -> None:
        """Test dependency injection of a pre-loaded or mock model."""
        mock_model = MagicMock()
        stt = WhisperSpeechToText(model=mock_model)
        self.assertTrue(stt.is_model_loaded)
        self.assertIs(stt.load_model(), mock_model)

    # Audio Path Validation Tests
    def test_missing_audio_file_raises_file_not_found(self) -> None:
        """Test non-existent audio file raises FileNotFoundError."""
        stt = WhisperSpeechToText()
        with self.assertRaises(FileNotFoundError):
            stt.transcribe("non_existent_audio_file_12345.wav")

    def test_empty_audio_path_raises_value_error(self) -> None:
        """Test empty audio path string raises ValueError."""
        stt = WhisperSpeechToText()
        with self.assertRaises(ValueError):
            stt.transcribe("")

        with self.assertRaises(ValueError):
            stt.transcribe("   ")

    def test_invalid_audio_path_type_raises_value_error(self) -> None:
        """Test non-string/non-Path audio path raises ValueError."""
        stt = WhisperSpeechToText()
        with self.assertRaises(ValueError):
            stt.transcribe(None)  # type: ignore

        with self.assertRaises(ValueError):
            stt.transcribe(12345)  # type: ignore

    def test_directory_audio_path_raises_value_error(self) -> None:
        """Test path pointing to a directory raises ValueError."""
        stt = WhisperSpeechToText()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError) as ctx:
                stt.transcribe(temp_dir)
            self.assertIn("directory", str(ctx.exception).lower())

    def test_unsupported_audio_extension_raises_value_error(self) -> None:
        """Test audio file with unsupported extension raises ValueError."""
        stt = WhisperSpeechToText()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                stt.transcribe(tmp_path)
            self.assertIn("unsupported", str(ctx.exception).lower())
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_valid_pathlib_path_accepted(self) -> None:
        """Test pathlib.Path instance is accepted during path validation."""
        stt = WhisperSpeechToText()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            validated = stt.validate_audio_path(tmp_path)
            self.assertEqual(validated, tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # Whisper Package & Model Loading Errors
    def test_whisper_not_installed_raises_runtime_error(self) -> None:
        """Test useful error message when openai-whisper package is not installed."""
        with patch.dict("sys.modules", {"whisper": None}):
            stt = WhisperSpeechToText()
            with self.assertRaises(RuntimeError) as ctx:
                stt.load_model()
            err_msg = str(ctx.exception)
            self.assertIn("openai-whisper", err_msg.lower())
            self.assertIn("not installed", err_msg.lower())

    def test_model_loading_failure_raises_runtime_error(self) -> None:
        """Test failure during whisper.load_model raises RuntimeError."""
        mock_whisper = MagicMock()
        mock_whisper.load_model.side_effect = RuntimeError("Out of memory on device")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            stt = WhisperSpeechToText(model_name="base")
            with self.assertRaises(RuntimeError) as ctx:
                stt.load_model()
            self.assertIn("Failed to load Whisper model 'base'", str(ctx.exception))

    # Mocked Successful Transcription Tests
    def test_transcribe_success_with_segments_duration(self) -> None:
        """Test successful transcription extracting duration from segments."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "  I have built multiple scalable microservices with Python and FastAPI.  ",
            "language": "en",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "I have built multiple scalable"},
                {"id": 1, "start": 2.5, "end": 5.4, "text": "microservices with Python and FastAPI."},
            ],
        }

        stt = WhisperSpeechToText(model=mock_model)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = stt.transcribe(tmp_path)
            self.assertIsInstance(result, TranscriptionResult)
            self.assertEqual(
                result.text,
                "I have built multiple scalable microservices with Python and FastAPI.",
            )
            self.assertEqual(result.language, "en")
            self.assertEqual(result.duration_seconds, 5.4)
            self.assertEqual(result.provider, "whisper")
            mock_model.transcribe.assert_called_once_with(str(tmp_path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_transcribe_success_with_explicit_duration(self) -> None:
        """Test successful transcription extracting duration from top-level key."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Respuesta en espanol para la entrevista.",
            "language": "es",
            "duration": 4.25,
        }

        stt = WhisperSpeechToText(model=mock_model)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = stt.transcribe(tmp_path)
            self.assertEqual(result.text, "Respuesta en espanol para la entrevista.")
            self.assertEqual(result.language, "es")
            self.assertEqual(result.duration_seconds, 4.25)
            self.assertEqual(result.provider, "whisper")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_transcribe_success_without_duration(self) -> None:
        """Test successful transcription when duration is unavailable."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Quick answer without segments.",
            "language": "fr",
        }

        stt = WhisperSpeechToText(model=mock_model)
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = stt.transcribe(tmp_path)
            self.assertEqual(result.text, "Quick answer without segments.")
            self.assertEqual(result.language, "fr")
            self.assertIsNone(result.duration_seconds)
            self.assertEqual(result.provider, "whisper")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_transcribe_forwards_additional_kwargs(self) -> None:
        """Test optional kwargs (e.g. language, temperature) are forwarded to whisper.transcribe."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "kwargs forwarded", "language": "en"}

        stt = WhisperSpeechToText(model=mock_model)
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stt.transcribe(tmp_path, temperature=0.0, fp16=False)
            mock_model.transcribe.assert_called_once_with(
                str(tmp_path),
                temperature=0.0,
                fp16=False,
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    # Mocked Transcription Failure Tests
    def test_transcribe_failure_raises_runtime_error(self) -> None:
        """Test that an exception during transcription raises RuntimeError with informative message."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("Audio decoding failed")

        stt = WhisperSpeechToText(model=mock_model)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                stt.transcribe(tmp_path)
            self.assertIn("Whisper transcription failed", str(ctx.exception))
            self.assertIn("Audio decoding failed", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
