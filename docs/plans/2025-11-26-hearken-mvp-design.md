# Hearken v0.1.0 MVP Design

**Package:** `hearken` (formerly `sr_pipeline`)
**Author:** Nick Hehr
**Date:** November 26, 2025
**Status:** Design Complete
**Version:** 0.1.0 MVP

---

## Executive Summary

This document describes the design for `hearken` v0.1.0, a production-ready speech recognition pipeline that prevents audio drops during transcription. The package evolves the validated POC (`sr_pipeline_poc/`) into a clean, extensible API with proper abstractions and dual-mode operation (active and passive listening).

**Key Design Principles:**
- Clean abstractions for audio sources, transcribers, and VAD
- Non-blocking capture thread (never drops audio)
- Full 4-state FSM for robust speech detection
- Both active (`wait_for_speech()`) and passive (callbacks) modes
- Minimal dependencies with optional extras

---

## 1. Architecture Overview

### 1.1 Three-Thread Pipeline

```
Microphone → [Capture Thread] → Queue → [Detect Thread] → Queue → [Transcribe Thread] → Callback
                   ↓                          ↓                         ↓
            AudioChunk (30ms)      SpeechSegment (complete)    Text transcription
```

**Threads:**
1. **Capture Thread** - Reads audio at fixed intervals, never blocks on downstream
2. **Detect Thread** - Runs VAD + FSM to segment audio into complete utterances
3. **Transcribe Thread** - Transcribes segments (only runs in passive mode with `on_transcript`)

**Queues:**
- `capture_queue` - AudioChunks from microphone (size: 100)
- `segment_queue` - Complete SpeechSegments (size: 10)

### 1.2 Core Abstractions

**AudioSource** - Abstract interface for audio input devices
- `open()` / `close()` - Resource lifecycle
- `read(num_samples)` - Read audio data
- Properties: `sample_rate`, `sample_width`

**Transcriber** - Abstract interface for speech-to-text
- `transcribe(segment) -> str` - Convert audio to text

**VAD** - Abstract interface for voice activity detection
- `process(chunk) -> VADResult` - Detect speech in chunk
- `reset()` - Clear state between utterances

---

## 2. Package Structure

```
hearken/
├── __init__.py              # Public API exports
├── listener.py              # Listener class (main entry point)
├── types.py                 # AudioChunk, SpeechSegment, DetectorConfig, etc.
├── interfaces.py            # AudioSource, Transcriber, VAD base classes
├── vad/
│   ├── __init__.py
│   ├── base.py              # VAD ABC (re-exported from interfaces)
│   └── energy.py            # EnergyVAD implementation
├── detector.py              # SpeechDetector FSM implementation
├── adapters/
│   ├── __init__.py
│   └── sr.py                # SpeechRecognitionSource, SRTranscriber adapters
└── _internal/
    └── logging.py           # Logger setup utilities

tests/
├── test_vad.py              # EnergyVAD with mock audio
├── test_detector.py         # FSM transitions
├── test_listener.py         # End-to-end with mocks
└── test_adapters.py         # SR adapter tests (skip if not installed)

examples/
├── basic_usage.py           # Passive mode with auto-transcription
└── active_mode.py           # Active mode with manual control
```

---

## 3. Core Components

### 3.1 Data Types (`types.py`)

```python
@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    data: bytes
    timestamp: float          # time.monotonic() when captured
    sample_rate: int
    sample_width: int         # bytes per sample (2 for 16-bit)


@dataclass
class SpeechSegment:
    """A complete speech segment ready for transcription."""
    audio_data: bytes         # Raw PCM audio
    sample_rate: int
    sample_width: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VADResult:
    """Result from voice activity detection."""
    is_speech: bool
    confidence: float = 1.0   # 0.0 to 1.0


class DetectorState(Enum):
    """FSM states for utterance segmentation."""
    IDLE = auto()             # Waiting for speech
    SPEECH_STARTING = auto()  # Speech detected, confirming it's not noise
    SPEAKING = auto()         # Confirmed speech, accumulating audio
    TRAILING_SILENCE = auto() # Speech may have ended, waiting to confirm


@dataclass
class DetectorConfig:
    """Configuration for utterance detection FSM."""
    min_speech_duration: float = 0.25  # seconds
    max_speech_duration: float = 30.0  # seconds
    silence_timeout: float = 0.8       # seconds
    speech_padding: float = 0.3        # seconds
    frame_duration_ms: int = 30        # milliseconds
```

### 3.2 Interfaces (`interfaces.py`)

```python
class AudioSource(ABC):
    """Abstract interface for audio input devices."""

    @abstractmethod
    def open(self) -> None:
        """Open the audio source for reading."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the audio source and release resources."""
        ...

    @abstractmethod
    def read(self, num_samples: int) -> bytes:
        """Read audio samples from the source."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate in Hz (e.g., 16000)."""
        ...

    @property
    @abstractmethod
    def sample_width(self) -> int:
        """Bytes per sample (e.g., 2 for 16-bit)."""
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class Transcriber(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    def transcribe(self, segment: SpeechSegment) -> str:
        """Transcribe audio to text. May raise exceptions for API errors."""
        ...


class VAD(ABC):
    """Voice Activity Detection interface."""

    @abstractmethod
    def process(self, chunk: AudioChunk) -> VADResult:
        """Process audio chunk and return speech detection result."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state between utterances."""
        ...

    @property
    def required_sample_rate(self) -> int | None:
        """Required sample rate, or None if flexible."""
        return None

    @property
    def required_frame_duration_ms(self) -> int | None:
        """Required frame duration in ms, or None if flexible."""
        return None
```

### 3.3 EnergyVAD (`vad/energy.py`)

```python
class EnergyVAD(VAD):
    """
    Simple energy-based voice activity detection.

    Uses RMS (root mean square) energy threshold to detect speech.
    Optionally adapts threshold based on ambient noise during calibration.
    """

    def __init__(
        self,
        threshold: float = 300.0,
        dynamic: bool = True,
        calibration_samples: int = 50,  # ~1.5s at 30ms frames
    ):
        self.base_threshold = threshold
        self.dynamic = dynamic
        self.calibration_samples = calibration_samples

        self._ambient_energy: Optional[float] = None
        self._samples_seen = 0

    def process(self, chunk: AudioChunk) -> VADResult:
        # Convert bytes to int16 samples
        samples = np.frombuffer(chunk.data, dtype=np.int16)

        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        # Dynamic threshold adjustment during calibration
        if self.dynamic and self._samples_seen < self.calibration_samples:
            if self._ambient_energy is None:
                self._ambient_energy = energy
            else:
                # Exponential moving average
                self._ambient_energy = 0.9 * self._ambient_energy + 0.1 * energy
            self._samples_seen += 1

            # Use 1.5x ambient energy as threshold during calibration
            effective_threshold = max(self.base_threshold, self._ambient_energy * 1.5)
        else:
            effective_threshold = self.base_threshold

        is_speech = energy > effective_threshold
        confidence = min(1.0, energy / effective_threshold) if is_speech else 0.0

        return VADResult(is_speech=is_speech, confidence=confidence)

    def reset(self) -> None:
        """Reset between utterances. Don't reset ambient calibration."""
        pass
```

**Design Note:** Auto-calibration happens during the first ~1.5 seconds when `dynamic=True`. This matches `speech_recognition` behavior. WebRTC and Silero VADs won't need calibration.

### 3.4 SpeechDetector FSM (`detector.py`)

The 4-state finite state machine segments continuous audio into discrete speech segments:

**State Transitions:**
- `IDLE → SPEECH_STARTING` - Speech detected, start accumulating with padding
- `SPEECH_STARTING → SPEAKING` - Duration exceeds `min_speech_duration` (confirmed speech)
- `SPEECH_STARTING → IDLE` - Silence exceeds `silence_timeout` before min duration (false start)
- `SPEAKING → TRAILING_SILENCE` - Silence detected
- `TRAILING_SILENCE → SPEAKING` - Speech resumes (handles pauses)
- `TRAILING_SILENCE → IDLE` - Silence exceeds `silence_timeout` (emit segment)

**Key Features:**
- Padding buffer (ring buffer) provides pre-roll audio before speech starts
- False start detection prevents noise bursts from triggering
- Max duration enforcement prevents unbounded segments

---

## 4. Listener Implementation

### 4.1 API Design

**Active Mode:**
```python
listener = Listener(source=mic, transcriber=google)
listener.start()

segment = listener.wait_for_speech()  # Blocks until speech detected
text = google.transcribe(segment)
```

**Passive Mode (Auto-transcription):**
```python
listener = Listener(
    source=mic,
    transcriber=google,
    on_transcript=lambda text, seg: print(text)
)
listener.start()
listener.wait()  # Runs until stopped
```

**Passive Mode (Manual transcription):**
```python
def handle_speech(segment):
    text = google.transcribe(segment)
    print(text)

listener = Listener(source=mic, on_speech=handle_speech)
listener.start()
listener.wait()
```

### 4.2 Callback Execution

**Critical Design Decision:** Callbacks (`on_speech`, `on_transcript`) execute in separate threads to prevent user code from blocking the pipeline.

```python
def _handle_segment(self, segment: SpeechSegment) -> None:
    """Handle detected speech segment."""
    # Fire callback asynchronously (don't block detect thread)
    if self.on_speech:
        threading.Thread(
            target=self._safe_callback,
            args=(self.on_speech, segment),
            daemon=True,
            name="hearken-callback"
        ).start()

    # Queue for active mode or transcription
    try:
        self._segment_queue.put_nowait(segment)
    except queue.Full:
        logger.warning(f"Segment queue full, dropping {segment.duration:.1f}s segment")
```

**Rationale:** Slow user callbacks (database writes, complex processing) should not block VAD or transcription threads.

### 4.3 Thread Lifecycle

**Start sequence:**
1. `listener.start()` opens audio source
2. Spawns capture, detect, and (optionally) transcribe threads
3. All threads marked as daemon (won't prevent program exit)

**Stop sequence:**
1. `listener.stop()` sets stop flag and event
2. Sends "poison pills" (None) to unblock queue.get()
3. Joins threads with timeout
4. Closes audio source

---

## 5. Speech Recognition Adapters

### 5.1 SpeechRecognitionSource

Adapts `speech_recognition.Microphone` to `AudioSource` interface:

```python
class SpeechRecognitionSource(AudioSource):
    def __init__(self, microphone: sr.Microphone):
        self.microphone = microphone
        self._context_manager = None

    def open(self) -> None:
        self._context_manager = self.microphone.__enter__()

    def close(self) -> None:
        if self._context_manager is not None:
            self.microphone.__exit__(None, None, None)

    def read(self, num_samples: int) -> bytes:
        return self.microphone.stream.read(num_samples, exception_on_overflow=False)

    @property
    def sample_rate(self) -> int:
        return self.microphone.SAMPLE_RATE

    @property
    def sample_width(self) -> int:
        return self.microphone.SAMPLE_WIDTH
```

### 5.2 SRTranscriber

Wraps any `speech_recognition.Recognizer.recognize_*` method:

```python
class SRTranscriber(Transcriber):
    def __init__(
        self,
        recognizer: sr.Recognizer,
        method: str = 'recognize_google',
        **kwargs
    ):
        self.recognizer = recognizer
        self._recognize_func = getattr(recognizer, method)
        self.kwargs = kwargs

    def transcribe(self, segment: SpeechSegment) -> str:
        # Convert SpeechSegment to sr.AudioData
        audio_data = sr.AudioData(
            segment.audio_data,
            segment.sample_rate,
            segment.sample_width
        )

        # Call recognition method (raises sr.UnknownValueError or sr.RequestError)
        return self._recognize_func(audio_data, **self.kwargs)
```

**Design Note:** Adapters live in `hearken.adapters.sr` and require `hearken[sr]` installation.

---

## 6. Configuration & Defaults

### 6.1 Sensible Defaults

The API works out of the box with no configuration:

```python
listener = Listener(source=mic, transcriber=google)
```

Defaults:
- `vad` → `EnergyVAD(threshold=300.0, dynamic=True)`
- `detector_config` → `DetectorConfig()` with values from POC
- `on_error` → Logs to `logging.getLogger('hearken')`

### 6.2 Escape Hatches

Advanced users can tune everything:

```python
listener = Listener(
    source=mic,
    transcriber=google,
    vad=EnergyVAD(threshold=500.0, dynamic=False),
    detector_config=DetectorConfig(
        min_speech_duration=0.5,
        silence_timeout=1.0,
        speech_padding=0.5,
    ),
    on_error=custom_error_handler,
)
```

---

## 7. Observability

### 7.1 Logging

All internal logging uses Python's `logging` module with logger name `'hearken'`:

```python
import logging

# Configure hearken logging
logging.getLogger('hearken').setLevel(logging.DEBUG)

# Or just root logger
logging.basicConfig(level=logging.INFO)
```

**Log levels:**
- `DEBUG` - FSM transitions, thread lifecycle
- `INFO` - Segment detected, pipeline start/stop
- `WARNING` - Queue full, dropped chunks/segments
- `ERROR` - Capture errors, transcription failures

### 7.2 Error Callback

The `on_error` callback provides programmatic error handling:

```python
def handle_error(error: Exception):
    sentry.capture_exception(error)

listener = Listener(source=mic, transcriber=google, on_error=handle_error)
```

Default behavior: `logging.error(f"Pipeline error: {error}", exc_info=True)`

---

## 8. Dependencies

### 8.1 Core Dependencies

```toml
dependencies = [
    "numpy>=1.20",  # For RMS energy calculation
]
```

### 8.2 Optional Dependencies

```toml
[project.optional-dependencies]
sr = [
    "SpeechRecognition>=3.8",
    "PyAudio>=0.2.11",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
    "ruff>=0.1.0",
]
all = ["hearken[sr]"]
```

### 8.3 Installation

```bash
# Core package (bring your own audio source & transcriber)
pip install hearken

# With speech_recognition adapters (recommended)
pip install hearken[sr]

# Development
uv sync --all-extras
```

---

## 9. Testing Strategy

### 9.1 Unit Tests (MVP Scope)

**Mock audio data approach:**

```python
def create_mock_chunk(is_speech: bool, sample_rate: int = 16000) -> AudioChunk:
    """Create mock audio chunk with known energy profile."""
    if is_speech:
        # High energy signal
        samples = np.random.randint(-5000, 5000, size=480, dtype=np.int16)
    else:
        # Low energy signal
        samples = np.random.randint(-100, 100, size=480, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )
```

**Test coverage:**
- `test_vad.py` - EnergyVAD with synthetic audio (silence, speech, noise)
- `test_detector.py` - FSM transitions with controlled input sequences
- `test_listener.py` - End-to-end with mock AudioSource and Transcriber
- `test_adapters.py` - SR adapters (skip if SpeechRecognition not installed)

### 9.2 Integration Tests (Deferred to v0.2)

Real microphone tests require:
- PyAudio mocking in CI/CD
- Audio playback for known utterances
- Platform-specific microphone access

**Decision:** Rely on manual testing (laptop + Pi 4) for MVP, add integration test infrastructure in v0.2.

---

## 10. Migration from POC

### 10.1 What Changes

| POC | MVP (hearken) |
|-----|---------------|
| `sr_pipeline_poc/` | `hearken/` |
| `Utterance` | `SpeechSegment` |
| `AudioPipeline` | `Listener` |
| Hardcoded config | `DetectorConfig` |
| `PipelineMetrics` | Removed (logging only) |
| 2-state FSM | 4-state FSM |
| Monolithic Recognizer | `AudioSource` + `Transcriber` abstractions |

### 10.2 What Stays the Same

- Three-thread architecture
- Queue-based communication
- Non-blocking capture thread
- EnergyVAD algorithm
- Frame duration (30ms default)

### 10.3 POC Disposition

Archive `sr_pipeline_poc/` as reference implementation. Start fresh with `hearken/` directory structure.

---

## 11. Example Usage

### 11.1 Basic Usage

```python
import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

# Setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

# Create listener
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    on_transcript=lambda text, seg: print(f"[{seg.duration:.1f}s] {text}")
)

# Run
listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    listener.stop()
```

### 11.2 Active Mode

```python
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
)

listener.start()

while True:
    print("Waiting for speech...")
    segment = listener.wait_for_speech()

    if segment:
        try:
            text = listener.transcriber.transcribe(segment)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Could not understand")
```

---

## 12. Roadmap

### v0.1.0 (MVP) ✓
- Core abstractions (`AudioSource`, `Transcriber`, `VAD`)
- `Listener` with 3-thread architecture
- Full 4-state FSM
- `EnergyVAD` implementation
- Speech recognition adapters
- Active and passive modes
- Unit tests with mocks

### v0.2.0
- WebRTC VAD support
- Sample rate conversion utilities
- Integration tests with real microphone
- CI/CD pipeline

### v0.3.0
- Async transcriber interface (`AsyncTranscriber`)
- Async callback support
- aiohttp-based transcribers

### v0.4.0
- Silero VAD (neural network)
- Multiprocessing support for CPU-bound VAD
- Streaming recognition support

---

## 13. Success Criteria

**Functional:**
- ✅ Listener works in both active and passive modes
- ✅ EnergyVAD detects speech with <1% false positive rate on test audio
- ✅ FSM correctly handles false starts and mid-utterance pauses
- ✅ Speech recognition adapters work with all `sr.Recognizer.recognize_*` methods

**Non-Functional:**
- ✅ Drop rate <1% during normal speech (validated in POC)
- ✅ Drop rate <1% even with slow transcription (validated in stress test)
- ✅ Works on MacOS and Raspberry Pi 4
- ✅ Unit test coverage >80%

**API Quality:**
- ✅ Works with sensible defaults (no config required)
- ✅ Escape hatches available for advanced tuning
- ✅ Clear abstractions for custom audio sources and transcribers
- ✅ Type hints throughout

---

## 14. Open Questions & Future Work

### 14.1 Resolved in Design

- ✅ Package name: `hearken`
- ✅ Main class name: `Listener`
- ✅ Calibration: Auto-calibration in EnergyVAD only
- ✅ Metrics: Removed in favor of logging
- ✅ Callback execution: Async threads to prevent blocking

### 14.2 Deferred

- Sample rate conversion (needed for WebRTC VAD in v0.2)
- Async transcriber pattern (v0.3)
- Silero VAD multiprocessing (v0.4)
- Streaming recognition (v0.4+)

---

*End of Design Document*
