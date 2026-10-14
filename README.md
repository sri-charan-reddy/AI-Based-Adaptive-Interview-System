# Member 2: Adaptive Interview Engine

An independent, modular Python engine for managing adaptive interview flow, session state, dynamic difficulty scaling, and targeted question & follow-up selection for the **AI Interview Simulator**.

This module is designed for clean integration with:
- **Member 1 (Evaluation Module)**: Feeds answer evaluation scores and topic performance into `update_performance()` or `process_answer()`.
- **Member 3 (LLM & Backend Module)**: Consumes `AdaptiveResult`, `InterviewState`, and `Question` objects to dynamically serve appropriate interview questions tailored to the candidate's current difficulty and weak areas.

---

## 📁 Project Structure

```text
member2-adaptive-interview/
│
├── adaptive_interview/
│   ├── __init__.py          # Public package exports
│   ├── models.py            # InterviewState, Question, AdaptiveResult dataclasses, and Difficulty enum
│   ├── question_bank.py     # Local question bank for Software Developer topics & follow-ups
│   ├── selector.py          # QuestionSelector with deterministic priority logic
│   ├── engine.py            # AdaptiveInterviewEngine with difficulty logic & question retrieval
│   └── speech/              # Speech processing foundation (Tasks 5, 6, & 7)
│       ├── __init__.py      # STT, TTS, & Analytics module exports
│       ├── models.py        # TranscriptionResult, SpeechSynthesisResult, & SpeechAnalyticsResult
│       ├── base.py          # Abstract SpeechToTextProvider & TextToSpeechProvider interfaces
│       ├── whisper_stt.py   # Local WhisperSpeechToText provider implementation
│       ├── pyttsx_tts.py    # Local Pyttsx3TextToSpeech provider implementation
│       └── analytics.py     # SpeechAnalytics transcript analyzer & fluency metrics
│
├── tests/
│   ├── test_adaptive_interview.py      # State and difficulty scaling test suite
│   ├── test_performance_adaptive.py    # Performance updates, follow-up decisions, and process_answer
│   ├── test_question_selector.py        # Question selection, follow-up, and fallback test suite
│   ├── test_session_lifecycle.py        # Session lifecycle and progression test suite
│   ├── test_speech_to_text.py           # Speech-to-Text models, validation, and Whisper provider tests
│   ├── test_text_to_speech.py           # Text-to-Speech models, validation, and Pyttsx3 provider tests
│   ├── test_speech_analytics.py        # Speech Analytics models, filler matching, and WPM tests
│   └── test_voice_answer_processing.py # Voice-answer pipeline and integration test suite
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Key Features & Models

### 1. `InterviewState` ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/models.py))
Tracks the full lifecycle of an interview session:
- `role`: Target job role (e.g., `"Software Engineer"`).
- `difficulty`: Current difficulty level (`"easy"`, `"medium"`, `"hard"`).
- `question_number`: Count of questions presented to the candidate.
- `questions_asked`: History of question prompts / IDs.
- `answers`: History of candidate responses.
- `scores`: Numerical evaluation scores (0–100 scale).
- `topics`: List of topics covered during the session.
- `weak_topics`: Identified weak areas where score fell below 50 (cleared on subsequent scores $\ge 80$).
- `to_dict()` / `from_dict()`: Seamless dictionary serialization for APIs/data stores.

### 2. `Question` ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/models.py))
Structured question item:
- `id`: Unique identifier (e.g., `"py_easy_01"`).
- `text`: Question prompt.
- `topic`: Topic category (`"Python"`, `"OOP"`, `"Data Structures"`).
- `difficulty`: Difficulty level (`"easy"`, `"medium"`, `"hard"`).
- `follow_up`: Boolean flag indicating if this is a follow-up question.
- Subscriptable `question["text"]` and serializable via `question.to_dict()`.

### 3. `AdaptiveResult` ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/models.py))
Structured result returned by `engine.process_answer()`:
- `score`: Evaluated performance score (0–100).
- `topic`: Associated topic.
- `difficulty`: Resulting difficulty level (`"easy"`, `"medium"`, `"hard"`).
- `is_weak_topic`: Boolean flag indicating if topic is currently weak.
- `should_follow_up`: Boolean flag indicating if an adaptive follow-up question should be asked.

### 4. `AdaptiveInterviewEngine` ([engine.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/engine.py))
Core business logic for session orchestration and adaptation:
- `start_interview(role, initial_difficulty='medium', topics=None)`: Starts a new interview session.
- `get_next_question(state=None, is_follow_up=False, topic=None)`: Retrieves the next best question from the selector and tracks its ID.
- `record_answer(question, answer, state=None)`: Stores questions and answers, incrementing the question counter.
- `update_performance(score, topic=None, state=None)`: Records score, updates weak topics (adds for $<50$, removes for $\ge 80$), and adjusts difficulty.
- `should_ask_follow_up(score, topic=None, state=None)`: Evaluates if poor score or weak topic warrants a follow-up.
- `process_answer(score, topic=None, state=None)`: High-level unified method performing validation, performance update, difficulty adjustment, and returning an `AdaptiveResult`.

### 5. Speech-to-Text Foundation ([speech/](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech))
Modular foundation for transcribing candidate verbal answers via local speech recognition:
- **`TranscriptionResult`** ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/models.py)): Dataclass encapsulating transcribed text, detected/specified `language`, `duration_seconds`, and `provider` attribution (defaults to `"whisper"`). Supports input validation, dictionary subscripting (`result['text']`), and `to_dict()` / `from_dict()` serialization.
- **`SpeechToTextProvider`** ([base.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/base.py)): Abstract base class defining `transcribe(audio_path) -> TranscriptionResult`.
- **`WhisperSpeechToText`** ([whisper_stt.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/whisper_stt.py)): Local implementation using the OpenAI Whisper library. Features:
  - Model selection with lightweight default (`"base"`, `"tiny"`, etc.).
  - Lazy model loading so importing and instantiating does not trigger immediate model downloads or memory allocations.
  - Strict audio path validation (checking file existence, non-empty paths, regular file checks, and supported audio formats).
  - Clean duration calculation and language detection.
  - Informative error reporting when Whisper is not installed or when transcription fails.
  - 100% local operation without requiring any API keys.

### 6. Text-to-Speech Foundation ([speech/](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech))
Modular foundation for synthesizing spoken interview prompts and question audio via local text-to-speech:
- **`SpeechSynthesisResult`** ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/models.py)): Dataclass encapsulating synthesized `text`, output `audio_path`, `provider` attribution (defaults to `"pyttsx3"`), `duration_seconds`, and `voice`. Supports input validation, dictionary subscripting (`result['audio_path']`), and `to_dict()` / `from_dict()` serialization.
- **`TextToSpeechProvider`** ([base.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/base.py)): Abstract base class defining `synthesize(text, output_path, **kwargs) -> SpeechSynthesisResult`.
- **`Pyttsx3TextToSpeech`** ([pyttsx_tts.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/pyttsx_tts.py)): Local offline implementation using the `pyttsx3` library (native SAPI5 on Windows). Features:
  - Lazy engine initialization so importing and instantiating does not invoke audio drivers or COM subsystems prematurely.
  - Strict input validation for non-empty text, valid `.wav` destination path, speaking rate, volume, and voice selection.
  - Runtime property overrides on individual `synthesize()` calls (e.g. custom `rate`, `volume`, or `voice`).
  - Automatic creation of target parent directories if missing.
  - Clear error reporting when `pyttsx3` is missing or when synthesis operations fail.
  - 100% offline and local (zero cloud dependencies, microphone/speaker requirements, or API keys).

### 7. Speech Analytics Foundation ([speech/](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech))
Modular text-level speech analytics evaluating candidate spoken responses:
- **`SpeechAnalyticsResult`** ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/models.py)): Structured dataclass holding `word_count`, `filler_words_count`, `filler_words` (frequency map), `duration_seconds`, `words_per_minute` (WPM), `filler_ratio`, `pace_status` (`"slow"`, `"moderate"`, `"fast"`), `fluency_score` (0–100), and `fluency_indicators` dictionary. Supports `to_dict()`, `from_dict()`, and dictionary subscripting (`result['word_count']`).
- **`SpeechAnalytics`** ([analytics.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/analytics.py)): Transcript analysis engine:
  - Total word count extraction supporting contractions (`"it's"`) and hyphenated terms (`"state-of-the-art"`).
  - Whole-word and whole-phrase filler detection (defaulting to `"um"`, `"uh"`, `"erm"`, `"like"`, and `"you know"`), case-insensitively.
  - Substring avoidance: prevents false positives inside words like `"umbrella"`, `"unlike"`, `"likely"`, `"dislike"`, or `"uh-huh"`.
  - Speaking pace calculation (Words Per Minute) when a positive duration is provided.
  - Text-derived fluency scoring and density analysis.
  - Convenience helper function: `analyze_speech(text, duration_seconds=None)`.

> [!NOTE]
> **Acoustic Limitation**: This module performs text-level transcript analysis. It cannot measure acoustic properties such as actual silence/pause durations, pitch variation, pronunciation clarity, vocal tone, or emotional inflection.

### 8. Voice-Answer Processing Pipeline ([engine.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/engine.py))
End-to-end integration coordinating Speech-to-Text, verbal fluency analytics, interview answer recording, and adaptive difficulty scaling:
- **`VoiceAnswerResult`** ([models.py](file:///c:/Projects/ai-interview-simulator/member2-adaptive-interview/adaptive_interview/speech/models.py)): Structured dataclass holding `audio_path`, `transcription` (`TranscriptionResult`), `analytics` (`SpeechAnalyticsResult`), and optional `adaptive_result` (`AdaptiveResult | None`). Supports `to_dict()`, `from_dict()`, and subscript access (`result['analytics']`).
- **`AdaptiveInterviewEngine.process_voice_answer()`**:
  1. **Lifecycle & Input Validation**: Ensures the interview session is active (`in_progress`), validates non-empty question and audio path, and validates performance score bounds.
  2. **Speech-to-Text Transcription**: Invokes an injected `SpeechToTextProvider` (defaults to local Whisper) to convert audio to text.
  3. **Fluency & Pace Analysis**: Evaluates speaking rate (WPM), filler occurrences, and fluency score using `SpeechAnalytics`.
  4. **State Recording**: Stores the transcribed answer into `InterviewState.answers` via `record_answer()`.
  5. **Adaptive Performance**: If an evaluation score is provided, executes `process_answer()` to adjust session difficulty and update weak topics.
- **Dependency Injection**: `AdaptiveInterviewEngine` accepts custom or mock STT providers and analytics engines via `__init__`, properties, or runtime method parameters, ensuring Whisper models are never loaded during engine initialization or unit tests.

---

## ⚙️ Difficulty Scaling & Selection Logic

### Difficulty Transitions:
| Score Range | Action | Transition Example |
|---|---|---|
| **Score $\ge$ 80** | Increase difficulty | `easy` $\to$ `medium`, `medium` $\to$ `hard` |
| **Score 50 – 79** | Keep difficulty unchanged | `medium` $\to$ `medium` |
| **Score < 50** | Decrease difficulty | `hard` $\to$ `medium`, `medium` $\to$ `easy` |

### Follow-up Decision Rules:
- **Score < 50**: `should_follow_up = True` (triggers immediate remediation).
- **Score 50 – 79**: `should_follow_up = True` if the topic is currently marked weak; otherwise `False`.
- **Score $\ge$ 80**: `should_follow_up = False` (clears weak status and progresses).

---

## 💻 Usage Example

```python
from adaptive_interview import AdaptiveInterviewEngine

# 1. Initialize Engine and start interview
engine = AdaptiveInterviewEngine()
state = engine.start_interview(
    role="Backend Engineer",
    initial_difficulty="medium",
    topics=["Python", "OOP", "Data Structures"]
)

# 2. Retrieve first question
question = engine.get_next_question()
print(f"Question ({question.difficulty}, {question.topic}): {question.text}")

# 3. Candidate answers and answer is processed
engine.record_answer(
    question=question.text,
    answer="Candidate response here..."
)
result = engine.process_answer(score=40.0, topic=question.topic)

print(f"New Difficulty: {result.difficulty}")         # 'easy'
print(f"Should Follow-up: {result.should_follow_up}")  # True
print(f"Is Weak Topic: {result.is_weak_topic}")        # True

# 4. Serve targeted follow-up question
next_q = engine.get_next_question(
    is_follow_up=result.should_follow_up,
    topic=result.topic
)
print(f"Follow-up: {next_q.text}")
```

### Speech-to-Text Transcription Example:

```python
from adaptive_interview.speech import WhisperSpeechToText

# 1. Initialize Whisper STT provider (model loads lazily on first call)
stt = WhisperSpeechToText(model_name="base")

# 2. Transcribe candidate's recorded audio answer
result = stt.transcribe("candidate_response.wav")

print(f"Transcribed Text: {result.text}")
print(f"Detected Language: {result.language}")          # e.g. 'en'
print(f"Duration: {result.duration_seconds} seconds")   # e.g. 5.4
print(f"Provider: {result.provider}")                   # 'whisper'

# Subscript and serialization support
print(f"Direct access: {result['text']}")
serialized_dict = result.to_dict()
```

### Text-to-Speech Synthesis Example:

```python
from adaptive_interview.speech import Pyttsx3TextToSpeech

# 1. Initialize local TTS provider (lazy engine initialization)
tts = Pyttsx3TextToSpeech(rate=165, volume=0.9)

# 2. Synthesize an interview question into an audio .wav file
result = tts.synthesize(
    text="Can you explain the difference between a list and a tuple in Python?",
    output_path="interview_question_01.wav"
)

print(f"Synthesized Text : {result.text}")
print(f"Generated Audio  : {result.audio_path}")
print(f"Provider         : {result.provider}")

# Subscript and serialization support
print(f"Direct access: {result['audio_path']}")
serialized_dict = result.to_dict()
```

### Speech Analytics Transcript Evaluation Example:

```python
from adaptive_interview.speech import SpeechAnalytics, analyze_speech

# 1. Analyze candidate response with duration (45 seconds)
candidate_transcript = (
    "Um, I have, you know, built scalable backend systems in Python, like microservices."
)

analytics = SpeechAnalytics()
result = analytics.analyze(text=candidate_transcript, duration_seconds=45.0)

print(f"Total Words        : {result.word_count}")            # e.g. 13
print(f"Filler Count       : {result.filler_words_count}")     # 3
print(f"Filler Breakdown   : {result.filler_words}")           # {'um': 1, 'you know': 1, 'like': 1}
print(f"Speaking Rate      : {result.words_per_minute} WPM")   # e.g. 17.33
print(f"Pace Status        : {result.pace_status}")            # e.g. 'slow'
print(f"Fluency Score      : {result.fluency_score} / 100")    # e.g. 85.0

# 2. Or use the quick helper function:
quick_result = analyze_speech("I like Python because it's concise.", duration_seconds=3.0)
print(f"Quick WPM: {quick_result.wpm}")
```

### Voice Answer Processing Pipeline Example:

```python
from adaptive_interview import AdaptiveInterviewEngine

# 1. Initialize engine (zero Whisper model downloads on init)
engine = AdaptiveInterviewEngine()
state = engine.start_interview(
    role="Backend Engineer",
    initial_difficulty="medium",
    topics=["Python", "Data Structures"]
)

# 2. Get current question
question = engine.get_next_question()

# 3. Process candidate's audio answer file with optional evaluation score
voice_result = engine.process_voice_answer(
    question=question.text,
    audio_path="candidate_answer.wav",
    score=45.0,            # Evaluated score (0-100)
    topic=question.topic    # "Python"
)

# 4. Inspect results across STT, speech analytics, and adaptive adjustment
print(f"Transcribed Text : {voice_result.transcription.text}")
print(f"Speaking Rate    : {voice_result.analytics.words_per_minute} WPM")
print(f"Filler Count     : {voice_result.analytics.filler_words_count}")
print(f"Fluency Score    : {voice_result.analytics.fluency_score} / 100")
print(f"New Difficulty   : {voice_result.adaptive_result.difficulty}")         # 'easy'
print(f"Needs Follow-up  : {voice_result.adaptive_result.should_follow_up}")  # True
```

---

## 🧪 Running Unit Tests

The test suite runs with zero external dependencies using Python's built-in `unittest` framework (all STT, TTS, and model dependencies are fully mocked during testing):

```bash
# Run all unit tests
python -m unittest discover -s tests -p "test_*.py" -v
```

Or using `pytest` if installed:

```bash
pytest tests/ -v
```

---

## 📦 Dependencies & Installation

- **Core Engine Runtime**: Standard Python Library (`3.10+`) — zero external dependencies.
- **Speech-to-Text (Local STT)**: `openai-whisper>=20231117` (executes locally, no API key required).
- **Text-to-Speech (Local TTS)**: `pyttsx3>=2.90` (offline local synthesis, native SAPI5 on Windows).
- **Testing (Optional)**: `pytest>=8.0.0`.

### System Prerequisites for Speech Processing (FFmpeg)

OpenAI Whisper requires **FFmpeg** for audio decoding and processing. Ensure FFmpeg is installed and accessible on your system `PATH`:

- **Windows**:
  ```powershell
  winget install Gyan.FFmpeg
  # or using Chocolatey:
  choco install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

### Python Package Installation

Install the Python dependencies into your virtual environment:

```bash
pip install -r requirements.txt
```
