# Hearken v0.1.0 MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build production-ready hearken package with core abstractions, EnergyVAD, 4-state FSM detector, and speech_recognition adapters.

**Architecture:** Three-thread pipeline (capture, detect, transcribe) with clean abstractions for audio sources, transcribers, and VAD. Non-blocking callbacks, logging-based observability.

**Tech Stack:** Python 3.11+, numpy, speech_recognition (optional), pytest, uv

---

## Pre-Implementation Setup

### Task 0: Create Package Structure

**Files:**
- Create: `hearken/__init__.py`
- Create: `hearken/interfaces.py`
- Create: `hearken/types.py`
- Create: `hearken/listener.py`
- Create: `hearken/detector.py`
- Create: `hearken/vad/__init__.py`
- Create: `hearken/vad/energy.py`
- Create: `hearken/adapters/__init__.py`
- Create: `hearken/adapters/sr.py`
- Create: `tests/__init__.py`
- Create: `examples/basic_usage.py`
- Create: `examples/active_mode.py`
- Create: `pyproject.toml`
- Create: `README.md`

**Step 1: Create directory structure**

```bash
mkdir -p hearken/vad hearken/adapters tests examples
touch hearken/__init__.py hearken/interfaces.py hearken/types.py
touch hearken/listener.py hearken/detector.py
touch hearken/vad/__init__.py hearken/vad/energy.py
touch hearken/adapters/__init__.py hearken/adapters/sr.py
touch tests/__init__.py
touch examples/basic_usage.py examples/active_mode.py
```

**Step 2: Create pyproject.toml**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hearken"
version = "0.1.0"
description = "Robust speech recognition pipeline that prevents audio drops"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [
    { name = "Nick Hehr", email = "headhipster@hipsterbrown.com" },
]
keywords = ["speech-recognition", "voice", "audio", "vad", "transcription"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
]

dependencies = [
    "numpy>=1.20",
]

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
all = [
    "hearken[sr]",
]

[project.urls]
Homepage = "https://github.com/hipsterbrown/hearken"
Documentation = "https://github.com/hipsterbrown/hearken#readme"
Repository = "https://github.com/hipsterbrown/hearken"
Issues = "https://github.com/hipsterbrown/hearken/issues"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
    "ruff>=0.1.0",
]
```

**Step 3: Install dependencies**

Run: `uv sync --all-extras`
Expected: Dependencies installed successfully

**Step 4: Commit**

```bash
git add hearken/ tests/ examples/ pyproject.toml
git commit -m "feat: create hearken package structure and configuration"
```

---

## Part 1: Core Data Types

### Task 1: Implement Core Data Types

**Files:**
- Create: `hearken/types.py`
- Create: `tests/test_types.py`

**Step 1: Write test for AudioChunk**

Create `tests/test_types.py`:

```python
import time
from hearken.types import AudioChunk


def test_audio_chunk_creation():
    """Test AudioChunk dataclass creation."""
    data = b'\x00\x01' * 480  # 30ms at 16kHz, 16-bit
    timestamp = time.monotonic()

    chunk = AudioChunk(
        data=data,
        timestamp=timestamp,
        sample_rate=16000,
        sample_width=2,
    )

    assert chunk.data == data
    assert chunk.timestamp == timestamp
    assert chunk.sample_rate == 16000
    assert chunk.sample_width == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py::test_audio_chunk_creation -v`
Expected: FAIL with "cannot import name 'AudioChunk'"

**Step 3: Implement AudioChunk**

Create `hearken/types.py`:

```python
"""Core data types for hearken pipeline."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    data: bytes
    timestamp: float          # time.monotonic() when captured
    sample_rate: int
    sample_width: int         # bytes per sample (2 for 16-bit)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py::test_audio_chunk_creation -v`
Expected: PASS

**Step 5: Write test for SpeechSegment**

Add to `tests/test_types.py`:

```python
from hearken.types import SpeechSegment


def test_speech_segment_creation():
    """Test SpeechSegment dataclass creation."""
    audio_data = b'\x00\x01' * 8000  # 1 second at 16kHz

    segment = SpeechSegment(
        audio_data=audio_data,
        sample_rate=16000,
        sample_width=2,
        start_time=1.0,
        end_time=2.0,
    )

    assert segment.audio_data == audio_data
    assert segment.sample_rate == 16000
    assert segment.sample_width == 2
    assert segment.start_time == 1.0
    assert segment.end_time == 2.0
    assert segment.duration == 1.0
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_types.py::test_speech_segment_creation -v`
Expected: FAIL with "cannot import name 'SpeechSegment'"

**Step 7: Implement SpeechSegment**

Add to `hearken/types.py`:

```python
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
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_types.py::test_speech_segment_creation -v`
Expected: PASS

**Step 9: Write tests for VADResult and DetectorState**

Add to `tests/test_types.py`:

```python
from hearken.types import VADResult, DetectorState


def test_vad_result_creation():
    """Test VADResult dataclass creation."""
    result = VADResult(is_speech=True, confidence=0.9)

    assert result.is_speech is True
    assert result.confidence == 0.9


def test_vad_result_default_confidence():
    """Test VADResult default confidence."""
    result = VADResult(is_speech=False)

    assert result.is_speech is False
    assert result.confidence == 1.0


def test_detector_state_enum():
    """Test DetectorState enum values."""
    assert DetectorState.IDLE
    assert DetectorState.SPEECH_STARTING
    assert DetectorState.SPEAKING
    assert DetectorState.TRAILING_SILENCE
```

**Step 10: Run tests to verify they fail**

Run: `pytest tests/test_types.py -v -k "vad_result or detector_state"`
Expected: FAIL with import errors

**Step 11: Implement VADResult and DetectorState**

Add to `hearken/types.py`:

```python
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
```

**Step 12: Run tests to verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: All tests PASS

**Step 13: Write test for DetectorConfig**

Add to `tests/test_types.py`:

```python
from hearken.types import DetectorConfig


def test_detector_config_defaults():
    """Test DetectorConfig default values."""
    config = DetectorConfig()

    assert config.min_speech_duration == 0.25
    assert config.max_speech_duration == 30.0
    assert config.silence_timeout == 0.8
    assert config.speech_padding == 0.3
    assert config.frame_duration_ms == 30


def test_detector_config_custom_values():
    """Test DetectorConfig with custom values."""
    config = DetectorConfig(
        min_speech_duration=0.5,
        silence_timeout=1.0,
    )

    assert config.min_speech_duration == 0.5
    assert config.silence_timeout == 1.0
    assert config.max_speech_duration == 30.0  # Default
```

**Step 14: Run test to verify it fails**

Run: `pytest tests/test_types.py::test_detector_config_defaults -v`
Expected: FAIL with import error

**Step 15: Implement DetectorConfig**

Add to `hearken/types.py`:

```python
@dataclass
class DetectorConfig:
    """Configuration for utterance detection FSM."""

    # Minimum speech duration to consider valid (filters transients)
    min_speech_duration: float = 0.25  # seconds

    # Maximum speech duration before forced segmentation
    max_speech_duration: float = 30.0  # seconds

    # Silence duration to end an utterance
    silence_timeout: float = 0.8  # seconds

    # Audio to prepend before detected speech start
    speech_padding: float = 0.3  # seconds

    # Frame duration for audio chunks
    frame_duration_ms: int = 30  # milliseconds
```

**Step 16: Run all type tests to verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: All tests PASS

**Step 17: Commit**

```bash
git add hearken/types.py tests/test_types.py
git commit -m "feat: implement core data types with tests

- AudioChunk for raw audio with metadata
- SpeechSegment for complete utterances
- VADResult for speech detection results
- DetectorState enum for FSM states
- DetectorConfig for detection parameters"
```

---

## Part 2: Abstract Interfaces

### Task 2: Implement Abstract Interfaces

**Files:**
- Create: `hearken/interfaces.py`
- Create: `tests/test_interfaces.py`

**Step 1: Write test for AudioSource interface**

Create `tests/test_interfaces.py`:

```python
from hearken.interfaces import AudioSource


class MockAudioSource(AudioSource):
    """Mock implementation for testing."""

    def __init__(self):
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read(self, num_samples: int) -> bytes:
        return b'\x00' * num_samples * 2

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def sample_width(self) -> int:
        return 2


def test_audio_source_context_manager():
    """Test AudioSource context manager protocol."""
    source = MockAudioSource()

    assert not source.is_open

    with source as s:
        assert source.is_open
        assert s is source

    assert not source.is_open


def test_audio_source_read():
    """Test AudioSource read method."""
    source = MockAudioSource()
    source.open()

    data = source.read(480)
    assert len(data) == 960  # 480 samples * 2 bytes

    source.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py::test_audio_source_context_manager -v`
Expected: FAIL with "cannot import name 'AudioSource'"

**Step 3: Implement AudioSource interface**

Create `hearken/interfaces.py`:

```python
"""Abstract interfaces for hearken components."""

from abc import ABC, abstractmethod


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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_interfaces.py -v -k audio_source`
Expected: All tests PASS

**Step 5: Write test for Transcriber interface**

Add to `tests/test_interfaces.py`:

```python
from hearken.interfaces import Transcriber
from hearken.types import SpeechSegment


class MockTranscriber(Transcriber):
    """Mock implementation for testing."""

    def transcribe(self, segment: SpeechSegment) -> str:
        return f"transcribed {segment.duration:.1f}s"


def test_transcriber_interface():
    """Test Transcriber interface."""
    transcriber = MockTranscriber()

    segment = SpeechSegment(
        audio_data=b'\x00' * 16000,
        sample_rate=16000,
        sample_width=2,
        start_time=0.0,
        end_time=1.0,
    )

    result = transcriber.transcribe(segment)
    assert result == "transcribed 1.0s"
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_interfaces.py::test_transcriber_interface -v`
Expected: FAIL with import error

**Step 7: Implement Transcriber interface**

Add to `hearken/interfaces.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import SpeechSegment


class Transcriber(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    def transcribe(self, segment: 'SpeechSegment') -> str:
        """Transcribe audio to text. May raise exceptions for API errors."""
        ...
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_interfaces.py::test_transcriber_interface -v`
Expected: PASS

**Step 9: Write test for VAD interface**

Add to `tests/test_interfaces.py`:

```python
from hearken.interfaces import VAD
from hearken.types import AudioChunk, VADResult


class MockVAD(VAD):
    """Mock implementation for testing."""

    def process(self, chunk: AudioChunk) -> VADResult:
        # Simple mock: always returns speech
        return VADResult(is_speech=True, confidence=1.0)

    def reset(self) -> None:
        pass


def test_vad_interface():
    """Test VAD interface."""
    vad = MockVAD()

    chunk = AudioChunk(
        data=b'\x00' * 960,
        timestamp=0.0,
        sample_rate=16000,
        sample_width=2,
    )

    result = vad.process(chunk)
    assert result.is_speech is True
    assert result.confidence == 1.0


def test_vad_default_properties():
    """Test VAD default property values."""
    vad = MockVAD()

    assert vad.required_sample_rate is None
    assert vad.required_frame_duration_ms is None
```

**Step 10: Run tests to verify they fail**

Run: `pytest tests/test_interfaces.py -v -k vad`
Expected: FAIL with import error

**Step 11: Implement VAD interface**

Add to `hearken/interfaces.py`:

```python
if TYPE_CHECKING:
    from .types import AudioChunk, VADResult


class VAD(ABC):
    """Voice Activity Detection interface."""

    @abstractmethod
    def process(self, chunk: 'AudioChunk') -> 'VADResult':
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

**Step 12: Run all interface tests to verify they pass**

Run: `pytest tests/test_interfaces.py -v`
Expected: All tests PASS

**Step 13: Commit**

```bash
git add hearken/interfaces.py tests/test_interfaces.py
git commit -m "feat: implement abstract interfaces with tests

- AudioSource for audio input devices
- Transcriber for speech-to-text
- VAD for voice activity detection"
```

---

## Part 3: EnergyVAD Implementation

### Task 3: Implement EnergyVAD

**Files:**
- Create: `hearken/vad/energy.py`
- Create: `tests/test_vad_energy.py`

**Step 1: Write test for basic EnergyVAD creation**

Create `tests/test_vad_energy.py`:

```python
import numpy as np
from hearken.vad.energy import EnergyVAD
from hearken.types import AudioChunk


def test_energy_vad_creation():
    """Test EnergyVAD initialization."""
    vad = EnergyVAD(threshold=300.0, dynamic=True)

    assert vad.base_threshold == 300.0
    assert vad.dynamic is True


def test_energy_vad_default_properties():
    """Test EnergyVAD default properties."""
    vad = EnergyVAD()

    assert vad.required_sample_rate is None
    assert vad.required_frame_duration_ms is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad_energy.py::test_energy_vad_creation -v`
Expected: FAIL with import error

**Step 3: Implement basic EnergyVAD structure**

Create `hearken/vad/energy.py`:

```python
"""Energy-based voice activity detection."""

import numpy as np
from typing import Optional

from ..interfaces import VAD
from ..types import AudioChunk, VADResult


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
        """
        Args:
            threshold: RMS energy threshold for speech detection
            dynamic: If True, adapt threshold based on ambient noise
            calibration_samples: Number of initial samples for calibration
        """
        self.base_threshold = threshold
        self.dynamic = dynamic
        self.calibration_samples = calibration_samples

        self._ambient_energy: Optional[float] = None
        self._samples_seen = 0

    def process(self, chunk: AudioChunk) -> VADResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset between utterances. Don't reset ambient calibration."""
        pass

    @property
    def required_sample_rate(self) -> Optional[int]:
        return None  # Works with any sample rate

    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        return None  # Works with any frame size
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vad_energy.py -v -k creation`
Expected: PASS

**Step 5: Write test for silence detection**

Add to `tests/test_vad_energy.py`:

```python
import time


def create_silence_chunk(sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Create a chunk of silence (low energy)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Low amplitude noise
    samples = np.random.randint(-100, 100, size=num_samples, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )


def test_energy_vad_detects_silence():
    """Test EnergyVAD detects silence correctly."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)

    chunk = create_silence_chunk()
    result = vad.process(chunk)

    assert result.is_speech is False
    assert result.confidence == 0.0
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_vad_energy.py::test_energy_vad_detects_silence -v`
Expected: FAIL with NotImplementedError

**Step 7: Write test for speech detection**

Add to `tests/test_vad_energy.py`:

```python
def create_speech_chunk(sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Create a chunk with speech-like energy."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # High amplitude signal
    samples = np.random.randint(-5000, 5000, size=num_samples, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )


def test_energy_vad_detects_speech():
    """Test EnergyVAD detects speech correctly."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)

    chunk = create_speech_chunk()
    result = vad.process(chunk)

    assert result.is_speech is True
    assert result.confidence > 0.0
```

**Step 8: Run test to verify it fails**

Run: `pytest tests/test_vad_energy.py::test_energy_vad_detects_speech -v`
Expected: FAIL with NotImplementedError

**Step 9: Implement EnergyVAD.process()**

Update `hearken/vad/energy.py`:

```python
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

    # Confidence: how far above/below threshold
    confidence = min(1.0, energy / effective_threshold) if is_speech else 0.0

    return VADResult(is_speech=is_speech, confidence=confidence)
```

**Step 10: Run all EnergyVAD tests to verify they pass**

Run: `pytest tests/test_vad_energy.py -v`
Expected: All tests PASS

**Step 11: Write test for dynamic calibration**

Add to `tests/test_vad_energy.py`:

```python
def test_energy_vad_dynamic_calibration():
    """Test EnergyVAD dynamic threshold adjustment."""
    vad = EnergyVAD(threshold=100.0, dynamic=True, calibration_samples=10)

    # Feed 10 silence chunks for calibration
    for _ in range(10):
        chunk = create_silence_chunk()
        result = vad.process(chunk)

    # After calibration, silence should still be silence
    chunk = create_silence_chunk()
    result = vad.process(chunk)
    assert result.is_speech is False

    # Speech should still be detected
    chunk = create_speech_chunk()
    result = vad.process(chunk)
    assert result.is_speech is True
```

**Step 12: Run calibration test to verify it passes**

Run: `pytest tests/test_vad_energy.py::test_energy_vad_dynamic_calibration -v`
Expected: PASS

**Step 13: Update vad __init__ to export EnergyVAD**

Create `hearken/vad/__init__.py`:

```python
"""Voice activity detection implementations."""

from .energy import EnergyVAD

__all__ = ['EnergyVAD']
```

**Step 14: Run all tests to verify**

Run: `pytest tests/test_vad_energy.py -v`
Expected: All tests PASS

**Step 15: Commit**

```bash
git add hearken/vad/energy.py hearken/vad/__init__.py tests/test_vad_energy.py
git commit -m "feat: implement EnergyVAD with tests

- RMS energy-based voice activity detection
- Dynamic threshold calibration
- Works with any sample rate/frame size"
```

---

## Part 4: SpeechDetector FSM

### Task 4: Implement SpeechDetector

**Files:**
- Create: `hearken/detector.py`
- Create: `tests/test_detector.py`

**Step 1: Write test for detector initialization**

Create `tests/test_detector.py`:

```python
from hearken.detector import SpeechDetector
from hearken.vad.energy import EnergyVAD
from hearken.types import DetectorState, DetectorConfig


def test_detector_initialization():
    """Test SpeechDetector initialization."""
    vad = EnergyVAD()
    config = DetectorConfig()

    detector = SpeechDetector(vad=vad, config=config)

    assert detector.state == DetectorState.IDLE
    assert detector.vad is vad
    assert detector.config is config
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_detector.py::test_detector_initialization -v`
Expected: FAIL with import error

**Step 3: Implement basic SpeechDetector structure**

Create `hearken/detector.py`:

```python
"""Speech detection finite state machine."""

import logging
import collections
from typing import Optional, Callable

from .types import AudioChunk, SpeechSegment, DetectorState, DetectorConfig, VADResult
from .interfaces import VAD

logger = logging.getLogger('hearken.detector')


class SpeechDetector:
    """
    Finite state machine for segmenting continuous audio into speech segments.

    Implements 4-state FSM to robustly detect speech boundaries:
    - IDLE: Waiting for speech
    - SPEECH_STARTING: Speech detected, confirming it's not transient noise
    - SPEAKING: Confirmed speech, accumulating audio
    - TRAILING_SILENCE: Speech may have ended, waiting to confirm
    """

    def __init__(
        self,
        vad: VAD,
        config: Optional[DetectorConfig] = None,
        on_segment: Optional[Callable[[SpeechSegment], None]] = None,
    ):
        """
        Args:
            vad: Voice activity detector instance
            config: Detection configuration (uses defaults if None)
            on_segment: Callback when complete segment detected
        """
        self.vad = vad
        self.config = config or DetectorConfig()
        self.on_segment = on_segment

        # FSM state
        self.state = DetectorState.IDLE

        # Ring buffer for speech padding (pre-roll)
        padding_frames = int(
            self.config.speech_padding * 1000 / self.config.frame_duration_ms
        )
        self.padding_buffer: collections.deque[AudioChunk] = collections.deque(
            maxlen=max(1, padding_frames)
        )

        # Current segment accumulator
        self.segment_chunks: list[AudioChunk] = []
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None

    def process(self, chunk: AudioChunk) -> None:
        """Process an audio chunk through the FSM."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.state = DetectorState.IDLE
        self.segment_chunks = []
        self.padding_buffer.clear()
        self.speech_start_time = None
        self.last_speech_time = None
        self.vad.reset()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_initialization -v`
Expected: PASS

**Step 5: Write test for IDLE → SPEECH_STARTING transition**

Add to `tests/test_detector.py`:

```python
import numpy as np
import time
from hearken.types import AudioChunk


def create_chunk(is_speech: bool, timestamp: float) -> AudioChunk:
    """Create mock audio chunk."""
    if is_speech:
        samples = np.random.randint(-5000, 5000, size=480, dtype=np.int16)
    else:
        samples = np.random.randint(-100, 100, size=480, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=timestamp,
        sample_rate=16000,
        sample_width=2,
    )


def test_detector_idle_to_speech_starting():
    """Test transition from IDLE to SPEECH_STARTING."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)
    detector = SpeechDetector(vad=vad)

    # Process silence - should stay in IDLE
    chunk = create_chunk(is_speech=False, timestamp=0.0)
    detector.process(chunk)
    assert detector.state == DetectorState.IDLE

    # Process speech - should transition to SPEECH_STARTING
    chunk = create_chunk(is_speech=True, timestamp=0.03)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEECH_STARTING
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_detector.py::test_detector_idle_to_speech_starting -v`
Expected: FAIL with NotImplementedError

**Step 7: Implement process() method with IDLE handler**

Update `hearken/detector.py`:

```python
def process(self, chunk: AudioChunk) -> None:
    """
    Process an audio chunk through the FSM.

    Args:
        chunk: Audio chunk to process
    """
    # Run VAD
    try:
        vad_result = self.vad.process(chunk)
    except Exception as e:
        logger.error(f"VAD processing failed: {e}")
        return

    is_speech = vad_result.is_speech
    now = chunk.timestamp

    # FSM transitions
    if self.state == DetectorState.IDLE:
        self._handle_idle(chunk, is_speech, now)
    elif self.state == DetectorState.SPEECH_STARTING:
        self._handle_speech_starting(chunk, is_speech, now)
    elif self.state == DetectorState.SPEAKING:
        self._handle_speaking(chunk, is_speech, now)
    elif self.state == DetectorState.TRAILING_SILENCE:
        self._handle_trailing_silence(chunk, is_speech, now)


def _handle_idle(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """IDLE state: waiting for speech."""
    self.padding_buffer.append(chunk)

    if is_speech:
        logger.debug("Speech detected, transitioning to SPEECH_STARTING")
        self.state = DetectorState.SPEECH_STARTING
        self.speech_start_time = now
        self.last_speech_time = now
        # Include padding buffer
        self.segment_chunks = list(self.padding_buffer)


def _handle_speech_starting(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """SPEECH_STARTING: confirming speech isn't transient noise."""
    pass  # Stub for now


def _handle_speaking(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """SPEAKING: confirmed speech, accumulating audio."""
    pass  # Stub for now


def _handle_trailing_silence(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """TRAILING_SILENCE: speech may have ended, waiting to confirm."""
    pass  # Stub for now
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_idle_to_speech_starting -v`
Expected: PASS

**Step 9: Write test for SPEECH_STARTING → SPEAKING transition**

Add to `tests/test_detector.py`:

```python
def test_detector_speech_starting_to_speaking():
    """Test transition from SPEECH_STARTING to SPEAKING."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)
    config = DetectorConfig(min_speech_duration=0.09)  # 3 frames at 30ms
    detector = SpeechDetector(vad=vad, config=config)

    # Transition to SPEECH_STARTING
    chunk = create_chunk(is_speech=True, timestamp=0.0)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEECH_STARTING

    # Process 2 more speech frames (total 90ms)
    chunk = create_chunk(is_speech=True, timestamp=0.03)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEECH_STARTING

    chunk = create_chunk(is_speech=True, timestamp=0.06)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEECH_STARTING

    # Next frame should transition to SPEAKING (>= min_speech_duration)
    chunk = create_chunk(is_speech=True, timestamp=0.09)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEAKING
```

**Step 10: Run test to verify it fails**

Run: `pytest tests/test_detector.py::test_detector_speech_starting_to_speaking -v`
Expected: FAIL (stays in SPEECH_STARTING)

**Step 11: Implement _handle_speech_starting()**

Update `hearken/detector.py`:

```python
def _handle_speech_starting(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """SPEECH_STARTING: confirming speech isn't transient noise."""
    self.segment_chunks.append(chunk)

    if is_speech:
        self.last_speech_time = now

        # Check if we've exceeded minimum speech duration
        speech_duration = now - self.speech_start_time
        if speech_duration >= self.config.min_speech_duration:
            logger.debug(f"Speech confirmed after {speech_duration:.2f}s, transitioning to SPEAKING")
            self.state = DetectorState.SPEAKING
    else:
        # Check if silence has exceeded timeout (false start)
        silence_duration = now - self.last_speech_time
        if silence_duration >= self.config.silence_timeout:
            logger.debug(f"False start detected, returning to IDLE")
            self.state = DetectorState.IDLE
            self.segment_chunks = []
            self.padding_buffer.clear()
            self.vad.reset()
```

**Step 12: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_speech_starting_to_speaking -v`
Expected: PASS

**Step 13: Write test for false start (SPEECH_STARTING → IDLE)**

Add to `tests/test_detector.py`:

```python
def test_detector_false_start():
    """Test false start detection (SPEECH_STARTING → IDLE)."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)
    config = DetectorConfig(silence_timeout=0.1)  # Very short timeout
    detector = SpeechDetector(vad=vad, config=config)

    # Brief speech (false start)
    chunk = create_chunk(is_speech=True, timestamp=0.0)
    detector.process(chunk)
    assert detector.state == DetectorState.SPEECH_STARTING

    # Silence exceeds timeout - should return to IDLE
    chunk = create_chunk(is_speech=False, timestamp=0.03)
    detector.process(chunk)
    chunk = create_chunk(is_speech=False, timestamp=0.06)
    detector.process(chunk)
    chunk = create_chunk(is_speech=False, timestamp=0.09)
    detector.process(chunk)
    chunk = create_chunk(is_speech=False, timestamp=0.12)
    detector.process(chunk)

    assert detector.state == DetectorState.IDLE
```

**Step 14: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_false_start -v`
Expected: PASS

**Step 15: Write test for complete segment emission**

Add to `tests/test_detector.py`:

```python
def test_detector_emits_segment():
    """Test complete segment emission."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)
    config = DetectorConfig(
        min_speech_duration=0.06,
        silence_timeout=0.12,
    )

    segments = []
    detector = SpeechDetector(
        vad=vad,
        config=config,
        on_segment=lambda seg: segments.append(seg)
    )

    # Speech pattern: speech → silence → IDLE
    t = 0.0
    for _ in range(4):  # 120ms of speech
        chunk = create_chunk(is_speech=True, timestamp=t)
        detector.process(chunk)
        t += 0.03

    assert detector.state == DetectorState.SPEAKING

    # Silence to trigger segment emission
    for _ in range(5):  # 150ms of silence
        chunk = create_chunk(is_speech=False, timestamp=t)
        detector.process(chunk)
        t += 0.03

    # Should have emitted one segment
    assert len(segments) == 1
    assert segments[0].duration > 0
    assert detector.state == DetectorState.IDLE
```

**Step 16: Run test to verify it fails**

Run: `pytest tests/test_detector.py::test_detector_emits_segment -v`
Expected: FAIL (no segment emitted)

**Step 17: Implement _handle_speaking() and _handle_trailing_silence()**

Update `hearken/detector.py`:

```python
def _handle_speaking(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """SPEAKING: confirmed speech, accumulating audio."""
    self.segment_chunks.append(chunk)

    if is_speech:
        self.last_speech_time = now

    # Check max duration (force split)
    speech_duration = now - self.speech_start_time
    if speech_duration >= self.config.max_speech_duration:
        logger.debug(f"Max duration ({self.config.max_speech_duration}s) reached, emitting segment")
        self._emit_segment(now)
        self.state = DetectorState.IDLE
    elif not is_speech:
        logger.debug("Silence detected, transitioning to TRAILING_SILENCE")
        self.state = DetectorState.TRAILING_SILENCE


def _handle_trailing_silence(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
    """TRAILING_SILENCE: speech may have ended, waiting to confirm."""
    self.segment_chunks.append(chunk)

    if is_speech:
        logger.debug("Speech resumed, returning to SPEAKING")
        self.last_speech_time = now
        self.state = DetectorState.SPEAKING
    else:
        # Check if silence timeout exceeded
        silence_duration = now - self.last_speech_time
        if silence_duration >= self.config.silence_timeout:
            logger.debug(f"Silence confirmed after {silence_duration:.2f}s, emitting segment")
            self._emit_segment(now)
            self.state = DetectorState.IDLE


def _emit_segment(self, end_time: float) -> None:
    """Emit a complete speech segment."""
    if not self.segment_chunks:
        return

    # Combine chunks into single audio blob
    audio_data = b''.join(c.data for c in self.segment_chunks)

    segment = SpeechSegment(
        audio_data=audio_data,
        sample_rate=self.segment_chunks[0].sample_rate,
        sample_width=self.segment_chunks[0].sample_width,
        start_time=self.speech_start_time,
        end_time=end_time,
    )

    logger.info(f"Speech segment detected: {segment.duration:.2f}s")

    # Reset for next segment
    self.segment_chunks = []
    self.padding_buffer.clear()
    self.vad.reset()

    # Invoke callback
    if self.on_segment:
        try:
            self.on_segment(segment)
        except Exception as e:
            logger.error(f"Segment callback failed: {e}")
```

**Step 18: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_emits_segment -v`
Expected: PASS

**Step 19: Write test for pause handling (TRAILING_SILENCE → SPEAKING)**

Add to `tests/test_detector.py`:

```python
def test_detector_handles_pause():
    """Test mid-utterance pause handling (TRAILING_SILENCE → SPEAKING)."""
    vad = EnergyVAD(threshold=300.0, dynamic=False)
    config = DetectorConfig(
        min_speech_duration=0.06,
        silence_timeout=0.12,
    )

    segments = []
    detector = SpeechDetector(
        vad=vad,
        config=config,
        on_segment=lambda seg: segments.append(seg)
    )

    t = 0.0
    # Speech
    for _ in range(4):
        chunk = create_chunk(is_speech=True, timestamp=t)
        detector.process(chunk)
        t += 0.03

    assert detector.state == DetectorState.SPEAKING

    # Brief pause (< silence_timeout)
    for _ in range(2):  # 60ms silence
        chunk = create_chunk(is_speech=False, timestamp=t)
        detector.process(chunk)
        t += 0.03

    assert detector.state == DetectorState.TRAILING_SILENCE

    # Speech resumes
    chunk = create_chunk(is_speech=True, timestamp=t)
    detector.process(chunk)

    assert detector.state == DetectorState.SPEAKING
    assert len(segments) == 0  # No segment emitted yet
```

**Step 20: Run test to verify it passes**

Run: `pytest tests/test_detector.py::test_detector_handles_pause -v`
Expected: PASS

**Step 21: Run all detector tests**

Run: `pytest tests/test_detector.py -v`
Expected: All tests PASS

**Step 22: Commit**

```bash
git add hearken/detector.py tests/test_detector.py
git commit -m "feat: implement SpeechDetector FSM with tests

- 4-state FSM for robust speech segmentation
- False start detection
- Mid-utterance pause handling
- Max duration enforcement
- Segment emission with callback"
```

---

## Part 5: Listener Implementation

### Task 5: Implement Listener (Part 1 - Structure)

**Files:**
- Create: `hearken/listener.py`
- Create: `tests/test_listener.py`

**Step 1: Write test for Listener initialization**

Create `tests/test_listener.py`:

```python
from hearken import Listener
from hearken.interfaces import AudioSource, Transcriber
from hearken.types import SpeechSegment
from hearken.vad.energy import EnergyVAD


class MockAudioSource(AudioSource):
    def __init__(self):
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read(self, num_samples: int) -> bytes:
        import time
        time.sleep(0.001)  # Simulate device latency
        return b'\x00' * num_samples * 2

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def sample_width(self) -> int:
        return 2


class MockTranscriber(Transcriber):
    def transcribe(self, segment: SpeechSegment) -> str:
        return f"mock transcription {segment.duration:.1f}s"


def test_listener_initialization():
    """Test Listener initialization."""
    source = MockAudioSource()
    transcriber = MockTranscriber()
    vad = EnergyVAD()

    listener = Listener(
        source=source,
        transcriber=transcriber,
        vad=vad,
    )

    assert listener.source is source
    assert listener.transcriber is transcriber
    assert listener.vad is vad


def test_listener_requires_transcriber_for_on_transcript():
    """Test Listener requires transcriber when on_transcript provided."""
    source = MockAudioSource()

    try:
        listener = Listener(
            source=source,
            on_transcript=lambda text, seg: None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "transcriber required" in str(e).lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_initialization -v`
Expected: FAIL with import error

**Step 3: Implement basic Listener structure**

Create `hearken/listener.py`:

```python
"""Main Listener class for hearken pipeline."""

import logging
import threading
import queue
import time
from typing import Optional, Callable

from .interfaces import AudioSource, Transcriber, VAD
from .types import AudioChunk, SpeechSegment, DetectorConfig
from .detector import SpeechDetector
from .vad.energy import EnergyVAD

logger = logging.getLogger('hearken')


class Listener:
    """
    Multi-threaded speech recognition pipeline.

    Decouples audio capture, voice activity detection, and transcription
    into independent threads to prevent audio drops during processing.
    """

    def __init__(
        self,
        source: AudioSource,
        transcriber: Optional[Transcriber] = None,
        vad: Optional[VAD] = None,
        detector_config: Optional[DetectorConfig] = None,
        on_speech: Optional[Callable[[SpeechSegment], None]] = None,
        on_transcript: Optional[Callable[[str, SpeechSegment], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        capture_queue_size: int = 100,
        segment_queue_size: int = 10,
    ):
        """
        Args:
            source: Audio input source
            transcriber: Transcription engine (required if on_transcript provided)
            vad: Voice activity detector (defaults to EnergyVAD)
            detector_config: Detection parameters (uses defaults if None)
            on_speech: Callback for raw speech segments (passive mode)
            on_transcript: Callback for transcribed segments (passive mode)
            on_error: Error callback (defaults to logging.error)
            capture_queue_size: Max chunks in capture queue
            segment_queue_size: Max segments in segment queue
        """
        self.source = source
        self.transcriber = transcriber
        self.vad = vad or EnergyVAD()
        self.detector_config = detector_config or DetectorConfig()
        self.on_speech = on_speech
        self.on_transcript = on_transcript
        self.on_error = on_error or self._default_error_handler

        # Validate configuration
        if on_transcript and not transcriber:
            raise ValueError("transcriber required when on_transcript is provided")

        # Queues
        self._capture_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue(
            maxsize=capture_queue_size
        )
        self._segment_queue: queue.Queue[Optional[SpeechSegment]] = queue.Queue(
            maxsize=segment_queue_size
        )

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start all pipeline threads."""
        raise NotImplementedError

    def stop(self, timeout: float = 2.0) -> None:
        """Stop all pipeline threads gracefully."""
        raise NotImplementedError

    def wait(self) -> None:
        """Block until stop() is called or threads exit."""
        raise NotImplementedError

    def wait_for_speech(self, timeout: Optional[float] = None) -> Optional[SpeechSegment]:
        """Block until a speech segment is detected (active mode)."""
        raise NotImplementedError

    def _default_error_handler(self, error: Exception) -> None:
        """Default error handler - just logs."""
        logger.error(f"Pipeline error: {error}", exc_info=True)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_listener.py -v -k initialization`
Expected: PASS

**Step 5: Write test for start/stop lifecycle**

Add to `tests/test_listener.py`:

```python
def test_listener_start_stop():
    """Test Listener start and stop lifecycle."""
    source = MockAudioSource()
    listener = Listener(source=source)

    assert not source.is_open

    listener.start()
    assert source.is_open

    listener.stop()
    assert not source.is_open
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_start_stop -v`
Expected: FAIL with NotImplementedError

**Step 7: Implement start() and stop() methods**

Update `hearken/listener.py`:

```python
def start(self) -> None:
    """Start all pipeline threads."""
    if self._running:
        raise RuntimeError("Listener already running")

    logger.info("Starting listener")
    self._running = True
    self._stop_event.clear()

    # Open audio source
    try:
        self.source.open()
    except Exception as e:
        self._running = False
        logger.error(f"Failed to open audio source: {e}")
        raise

    # Start threads
    self._threads = [
        threading.Thread(
            target=self._capture_loop,
            name="hearken-capture",
            daemon=True
        ),
        threading.Thread(
            target=self._detect_loop,
            name="hearken-detect",
            daemon=True
        ),
    ]

    # Only start transcribe thread if needed for passive mode
    if self.on_transcript:
        self._threads.append(
            threading.Thread(
                target=self._transcribe_loop,
                name="hearken-transcribe",
                daemon=True
            )
        )

    for t in self._threads:
        t.start()

    logger.info("Listener started")


def stop(self, timeout: float = 2.0) -> None:
    """Stop all pipeline threads gracefully."""
    if not self._running:
        return

    logger.info("Stopping listener")
    self._running = False
    self._stop_event.set()

    # Send poison pills
    try:
        self._capture_queue.put_nowait(None)
    except queue.Full:
        pass

    try:
        self._segment_queue.put_nowait(None)
    except queue.Full:
        pass

    # Wait for threads
    for t in self._threads:
        t.join(timeout=timeout)

    self._threads.clear()

    # Close audio source
    try:
        self.source.close()
    except Exception as e:
        logger.error(f"Error closing audio source: {e}")

    logger.info("Listener stopped")


def wait(self) -> None:
    """Block until stop() is called or threads exit."""
    while self._running and any(t.is_alive() for t in self._threads):
        time.sleep(0.1)


def _capture_loop(self) -> None:
    """Capture thread: reads audio chunks at fixed intervals."""
    pass  # Stub


def _detect_loop(self) -> None:
    """Detection thread: runs VAD and FSM to segment audio."""
    pass  # Stub


def _transcribe_loop(self) -> None:
    """Transcription thread: transcribes segments and invokes callback."""
    pass  # Stub
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_listener.py::test_listener_start_stop -v`
Expected: PASS

**Step 9: Commit**

```bash
git add hearken/listener.py tests/test_listener.py
git commit -m "feat: implement Listener basic structure

- Initialization with validation
- Start/stop lifecycle
- Thread management stubs"
```

### Task 6: Implement Listener (Part 2 - Capture Thread)

**Files:**
- Modify: `hearken/listener.py`
- Modify: `tests/test_listener.py`

**Step 1: Write test for capture thread**

Add to `tests/test_listener.py`:

```python
def test_listener_capture_thread():
    """Test capture thread reads audio chunks."""
    source = MockAudioSource()
    listener = Listener(source=source)

    listener.start()
    time.sleep(0.1)  # Let it capture some chunks

    # Queue should have chunks
    assert listener._capture_queue.qsize() > 0

    listener.stop()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_capture_thread -v`
Expected: FAIL (queue empty)

**Step 3: Implement _capture_loop()**

Update `hearken/listener.py`:

```python
def _capture_loop(self) -> None:
    """Capture thread: reads audio chunks at fixed intervals."""
    frame_duration_ms = self.vad.required_frame_duration_ms or self.detector_config.frame_duration_ms
    chunk_samples = int(self.source.sample_rate * frame_duration_ms / 1000)

    logger.debug(f"Capture thread started (frame_duration={frame_duration_ms}ms, samples={chunk_samples})")

    chunks_captured = 0
    chunks_dropped = 0

    while self._running:
        try:
            # Read audio - releases GIL during device read
            data = self.source.read(chunk_samples)

            chunk = AudioChunk(
                data=data,
                timestamp=time.monotonic(),
                sample_rate=self.source.sample_rate,
                sample_width=self.source.sample_width,
            )

            # Non-blocking put
            try:
                self._capture_queue.put_nowait(chunk)
                chunks_captured += 1
            except queue.Full:
                chunks_dropped += 1
                if chunks_dropped % 100 == 0:
                    drop_rate = chunks_dropped / (chunks_captured + chunks_dropped) * 100
                    logger.warning(f"Capture queue full, dropped {chunks_dropped} chunks ({drop_rate:.1f}%)")

        except Exception as e:
            if self._running:
                logger.error(f"Capture error: {e}")
                self.on_error(e)
            break

    logger.debug(f"Capture thread stopped (captured={chunks_captured}, dropped={chunks_dropped})")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_listener.py::test_listener_capture_thread -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hearken/listener.py tests/test_listener.py
git commit -m "feat: implement Listener capture thread

- Reads audio at fixed frame rate
- Non-blocking queue puts
- Drop tracking with warnings"
```

### Task 7: Implement Listener (Part 3 - Detect Thread)

**Files:**
- Modify: `hearken/listener.py`
- Modify: `tests/test_listener.py`

**Step 1: Write test for detect thread**

Add to `tests/test_listener.py`:

```python
import numpy as np


class SpeechAudioSource(AudioSource):
    """Mock source that generates speech-like audio."""

    def __init__(self):
        self.is_open = False
        self.frame_count = 0

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read(self, num_samples: int) -> bytes:
        # Generate high-energy audio (speech)
        samples = np.random.randint(-5000, 5000, size=num_samples, dtype=np.int16)
        self.frame_count += 1
        return samples.tobytes()

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def sample_width(self) -> int:
        return 2


def test_listener_detect_thread():
    """Test detect thread processes chunks and detects speech."""
    from hearken.types import DetectorConfig

    source = SpeechAudioSource()
    config = DetectorConfig(
        min_speech_duration=0.09,
        silence_timeout=0.12,
    )

    segments = []

    listener = Listener(
        source=source,
        detector_config=config,
        on_speech=lambda seg: segments.append(seg),
    )

    listener.start()
    time.sleep(0.5)  # Let it run for 500ms
    listener.stop()

    # Should have detected at least one segment
    assert len(segments) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_detect_thread -v`
Expected: FAIL (no segments detected)

**Step 3: Implement _detect_loop() and _handle_segment()**

Update `hearken/listener.py`:

```python
def _detect_loop(self) -> None:
    """Detection thread: runs VAD and FSM to segment audio."""
    detector = SpeechDetector(
        vad=self.vad,
        config=self.detector_config,
        on_segment=self._handle_segment,
    )

    logger.debug("Detection thread started")

    while self._running:
        try:
            chunk = self._capture_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if chunk is None:  # Poison pill
            break

        detector.process(chunk)

    logger.debug("Detection thread stopped")


def _handle_segment(self, segment: SpeechSegment) -> None:
    """Handle detected speech segment."""
    # Call on_speech callback asynchronously (don't block detect thread)
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


def _safe_callback(self, callback: Callable, *args) -> None:
    """Execute callback with error handling."""
    try:
        callback(*args)
    except Exception as e:
        logger.error(f"Callback failed: {e}", exc_info=True)
        self.on_error(e)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_listener.py::test_listener_detect_thread -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hearken/listener.py tests/test_listener.py
git commit -m "feat: implement Listener detect thread

- Runs SpeechDetector FSM on captured chunks
- Async callback execution for on_speech
- Queues segments for active mode/transcription"
```

### Task 8: Implement Listener (Part 4 - Transcribe Thread & Active Mode)

**Files:**
- Modify: `hearken/listener.py`
- Modify: `tests/test_listener.py`

**Step 1: Write test for transcribe thread**

Add to `tests/test_listener.py`:

```python
def test_listener_transcribe_thread():
    """Test transcribe thread processes segments."""
    from hearken.types import DetectorConfig

    source = SpeechAudioSource()
    transcriber = MockTranscriber()
    config = DetectorConfig(
        min_speech_duration=0.09,
        silence_timeout=0.12,
    )

    transcripts = []

    listener = Listener(
        source=source,
        transcriber=transcriber,
        detector_config=config,
        on_transcript=lambda text, seg: transcripts.append((text, seg)),
    )

    listener.start()
    time.sleep(0.5)
    listener.stop()

    # Should have transcribed at least one segment
    assert len(transcripts) > 0
    text, seg = transcripts[0]
    assert "mock transcription" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_transcribe_thread -v`
Expected: FAIL (no transcripts)

**Step 3: Implement _transcribe_loop()**

Update `hearken/listener.py`:

```python
def _transcribe_loop(self) -> None:
    """Transcription thread: transcribes segments and invokes callback."""
    logger.debug("Transcription thread started")

    while self._running:
        try:
            segment = self._segment_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if segment is None:  # Poison pill
            break

        try:
            # Transcribe - may release GIL during network I/O
            text = self.transcriber.transcribe(segment)

            # Fire callback asynchronously (don't block transcription)
            if self.on_transcript:
                threading.Thread(
                    target=self._safe_callback,
                    args=(self.on_transcript, text, segment),
                    daemon=True,
                    name="hearken-callback"
                ).start()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.on_error(e)

    logger.debug("Transcription thread stopped")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_listener.py::test_listener_transcribe_thread -v`
Expected: PASS

**Step 5: Write test for active mode (wait_for_speech)**

Add to `tests/test_listener.py`:

```python
def test_listener_wait_for_speech():
    """Test active mode with wait_for_speech()."""
    from hearken.types import DetectorConfig

    source = SpeechAudioSource()
    config = DetectorConfig(
        min_speech_duration=0.09,
        silence_timeout=0.12,
    )

    listener = Listener(source=source, detector_config=config)
    listener.start()

    # Wait for speech (with timeout)
    segment = listener.wait_for_speech(timeout=2.0)

    listener.stop()

    assert segment is not None
    assert segment.duration > 0
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_listener.py::test_listener_wait_for_speech -v`
Expected: FAIL with NotImplementedError

**Step 7: Implement wait_for_speech()**

Update `hearken/listener.py`:

```python
def wait_for_speech(self, timeout: Optional[float] = None) -> Optional[SpeechSegment]:
    """
    Block until a speech segment is detected (active mode).

    Args:
        timeout: Optional timeout in seconds (None = wait indefinitely)

    Returns:
        SpeechSegment if detected, None if timeout

    Raises:
        RuntimeError: If listener not running
    """
    if not self._running:
        raise RuntimeError("Listener not running")

    try:
        segment = self._segment_queue.get(timeout=timeout)
        return segment if segment is not None else None
    except queue.Empty:
        return None
```

**Step 8: Run test to verify it passes**

Run: `pytest tests/test_listener.py::test_listener_wait_for_speech -v`
Expected: PASS

**Step 9: Run all Listener tests**

Run: `pytest tests/test_listener.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add hearken/listener.py tests/test_listener.py
git commit -m "feat: implement Listener transcribe thread and active mode

- Transcribe thread with async callbacks
- wait_for_speech() for active mode
- Complete 3-thread pipeline"
```

---

## Part 6: Speech Recognition Adapters

### Task 9: Implement Speech Recognition Adapters

**Files:**
- Create: `hearken/adapters/sr.py`
- Create: `tests/test_adapters_sr.py`

**Step 1: Write test for SpeechRecognitionSource**

Create `tests/test_adapters_sr.py`:

```python
import pytest

# Skip all tests if SpeechRecognition not installed
sr = pytest.importorskip("speech_recognition")

from hearken.adapters.sr import SpeechRecognitionSource


def test_speech_recognition_source_lifecycle():
    """Test SpeechRecognitionSource open/close."""
    mic = sr.Microphone()
    source = SpeechRecognitionSource(mic)

    source.open()
    assert source.sample_rate > 0
    assert source.sample_width > 0
    source.close()


def test_speech_recognition_source_context_manager():
    """Test SpeechRecognitionSource context manager."""
    mic = sr.Microphone()
    source = SpeechRecognitionSource(mic)

    with source as s:
        assert s is source
        assert source.sample_rate > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapters_sr.py::test_speech_recognition_source_lifecycle -v`
Expected: FAIL with import error (or SKIP if SR not installed)

**Step 3: Implement SpeechRecognitionSource**

Create `hearken/adapters/sr.py`:

```python
"""
Adapters for speech_recognition library compatibility.

These adapters allow using speech_recognition's Microphone and Recognizer
with hearken's abstract interfaces.

Only available when SpeechRecognition is installed (hearken[sr]).
"""

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    # Define stubs so module can still be imported
    class sr:
        class Microphone: pass
        class Recognizer: pass
        class AudioData: pass


from ..interfaces import AudioSource, Transcriber
from ..types import SpeechSegment


class SpeechRecognitionSource(AudioSource):
    """
    Adapter for speech_recognition.Microphone.

    Example:
        import speech_recognition as sr
        from hearken.adapters.sr import SpeechRecognitionSource

        mic = sr.Microphone()
        source = SpeechRecognitionSource(mic)

        listener = Listener(source=source, ...)
    """

    def __init__(self, microphone: 'sr.Microphone'):
        """
        Args:
            microphone: speech_recognition Microphone instance
        """
        if not SR_AVAILABLE:
            raise ImportError(
                "SpeechRecognition not installed. "
                "Install with: pip install hearken[sr]"
            )

        self.microphone = microphone
        self._context_manager = None

    def open(self) -> None:
        """Open the microphone."""
        self._context_manager = self.microphone.__enter__()

    def close(self) -> None:
        """Close the microphone."""
        if self._context_manager is not None:
            self.microphone.__exit__(None, None, None)
            self._context_manager = None

    def read(self, num_samples: int) -> bytes:
        """Read audio samples from microphone."""
        return self.microphone.stream.read(num_samples, exception_on_overflow=False)

    @property
    def sample_rate(self) -> int:
        return self.microphone.SAMPLE_RATE

    @property
    def sample_width(self) -> int:
        return self.microphone.SAMPLE_WIDTH
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adapters_sr.py -v -k source`
Expected: PASS (or SKIP if SR not installed)

**Step 5: Write test for SRTranscriber**

Add to `tests/test_adapters_sr.py`:

```python
from hearken.adapters.sr import SRTranscriber
from hearken.types import SpeechSegment
import numpy as np


def test_sr_transcriber_initialization():
    """Test SRTranscriber initialization."""
    recognizer = sr.Recognizer()
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    assert transcriber.recognizer is recognizer
    assert transcriber.method_name == 'recognize_google'


def test_sr_transcriber_invalid_method():
    """Test SRTranscriber with invalid method name."""
    recognizer = sr.Recognizer()

    try:
        transcriber = SRTranscriber(recognizer, method='invalid_method')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no method" in str(e).lower()


def test_sr_transcriber_creates_audio_data():
    """Test SRTranscriber converts SpeechSegment to AudioData."""
    recognizer = sr.Recognizer()
    transcriber = SRTranscriber(recognizer, method='recognize_sphinx')  # Offline

    # Create dummy segment
    samples = np.random.randint(-100, 100, size=16000, dtype=np.int16)
    segment = SpeechSegment(
        audio_data=samples.tobytes(),
        sample_rate=16000,
        sample_width=2,
        start_time=0.0,
        end_time=1.0,
    )

    # This will likely fail to recognize, but we're testing the conversion
    try:
        transcriber.transcribe(segment)
    except sr.UnknownValueError:
        pass  # Expected for random noise
```

**Step 6: Run test to verify it fails**

Run: `pytest tests/test_adapters_sr.py::test_sr_transcriber_initialization -v`
Expected: FAIL with import error

**Step 7: Implement SRTranscriber**

Add to `hearken/adapters/sr.py`:

```python
class SRTranscriber(Transcriber):
    """
    Adapter for speech_recognition.Recognizer.

    Wraps any recognition method from speech_recognition (Google, Sphinx, etc.)

    Example:
        import speech_recognition as sr
        from hearken.adapters.sr import SRTranscriber

        recognizer = sr.Recognizer()

        # Adjust for ambient noise
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)

        # Use Google Speech API
        transcriber = SRTranscriber(
            recognizer,
            method='recognize_google',
            language='en-US'
        )

        listener = Listener(source=..., transcriber=transcriber)
    """

    def __init__(
        self,
        recognizer: 'sr.Recognizer',
        method: str = 'recognize_google',
        **kwargs
    ):
        """
        Args:
            recognizer: speech_recognition Recognizer instance
            method: Recognition method name (e.g., 'recognize_google')
            **kwargs: Additional arguments passed to recognition method
        """
        if not SR_AVAILABLE:
            raise ImportError(
                "SpeechRecognition not installed. "
                "Install with: pip install hearken[sr]"
            )

        self.recognizer = recognizer
        self.method_name = method
        self.kwargs = kwargs

        # Get the recognition method
        if not hasattr(recognizer, method):
            raise ValueError(f"Recognizer has no method '{method}'")

        self._recognize_func = getattr(recognizer, method)

    def transcribe(self, segment: SpeechSegment) -> str:
        """
        Transcribe a speech segment using speech_recognition.

        Args:
            segment: Speech segment to transcribe

        Returns:
            Transcribed text

        Raises:
            Exception: If transcription fails (network error, etc.)
        """
        # Convert SpeechSegment to sr.AudioData
        audio_data = sr.AudioData(
            segment.audio_data,
            segment.sample_rate,
            segment.sample_width
        )

        # Call recognition method
        # Note: recognize_* methods raise sr.UnknownValueError if no speech detected
        # and sr.RequestError for API failures. Let these propagate.
        return self._recognize_func(audio_data, **self.kwargs)
```

**Step 8: Run all adapter tests**

Run: `pytest tests/test_adapters_sr.py -v`
Expected: All tests PASS (or SKIP if SR not installed)

**Step 9: Update adapters __init__**

Create `hearken/adapters/__init__.py`:

```python
"""Adapters for third-party libraries."""

# Speech recognition adapters (optional dependency)
try:
    from .sr import SpeechRecognitionSource, SRTranscriber
    __all__ = ['SpeechRecognitionSource', 'SRTranscriber']
except ImportError:
    __all__ = []
```

**Step 10: Commit**

```bash
git add hearken/adapters/ tests/test_adapters_sr.py
git commit -m "feat: implement speech_recognition adapters

- SpeechRecognitionSource adapter for sr.Microphone
- SRTranscriber adapter for sr.Recognizer
- Optional import handling for hearken[sr]"
```

---

## Part 7: Public API & Documentation

### Task 10: Create Public API Exports

**Files:**
- Modify: `hearken/__init__.py`
- Create: `README.md`

**Step 1: Update hearken/__init__.py**

Update `hearken/__init__.py`:

```python
"""
Hearken - Robust speech recognition pipeline for Python.

Decouples audio capture, voice activity detection, and transcription
into independent threads to prevent audio drops during processing.
"""

__version__ = "0.1.0"

# Core components
from .listener import Listener
from .types import (
    AudioChunk,
    SpeechSegment,
    VADResult,
    DetectorConfig,
    DetectorState,
)

# Interfaces
from .interfaces import AudioSource, Transcriber, VAD

# VAD implementations
from .vad.energy import EnergyVAD

__all__ = [
    # Main class
    "Listener",

    # Data types
    "AudioChunk",
    "SpeechSegment",
    "VADResult",
    "DetectorConfig",
    "DetectorState",

    # Interfaces
    "AudioSource",
    "Transcriber",
    "VAD",

    # VAD implementations
    "EnergyVAD",
]
```

**Step 2: Test imports work**

Run: `python -c "from hearken import Listener, EnergyVAD; print('OK')"`
Expected: OK

**Step 3: Create README.md**

Create `README.md`:

```markdown
# Hearken

Robust speech recognition pipeline for Python that prevents audio drops during transcription.

## The Problem

In typical speech detection programs, audio capture is blocked during transcription. This causes dropped frames when network I/O is slow, resulting in missed speech.

## The Solution

Hearken decouples capture, voice activity detection (VAD), and transcription into independent threads with queue-based communication. The capture thread never blocks, preventing audio loss even during slow transcription.

## Installation

```bash
# Core package
pip install hearken

# With speech_recognition adapters (recommended for getting started)
pip install hearken[sr]
```

## Quick Start

```python
import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

# Setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Create listener
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    on_transcript=lambda text, seg: print(f"You said: {text}")
)

# Run
listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    listener.stop()
```

## Features

- **No dropped frames**: Capture thread never blocks on downstream processing
- **Flexible VAD**: Pluggable voice activity detection (Energy-based included, WebRTC coming soon)
- **Two modes**: Passive (callbacks) and active (`wait_for_speech()`)
- **Clean abstractions**: Bring your own audio source and transcriber
- **Production-ready FSM**: Robust 4-state detector filters false starts and handles pauses

## Architecture

```
Microphone → [Capture Thread] → Queue → [Detect Thread] → Queue → [Transcribe Thread] → Callback
                   ↓                          ↓                         ↓
            AudioChunk (30ms)      SpeechSegment (complete)    Text transcription
```

## Active Mode

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

## Documentation

See [examples/](examples/) for more usage patterns.

## Development

```bash
# Clone repository
git clone https://github.com/hipsterbrown/hearken.git
cd hearken

# Install with dev dependencies
uv sync --all-extras

# Run tests
pytest

# Run tests with coverage
pytest --cov=hearken --cov-report=term-missing

# Format code
black hearken/ tests/

# Type checking
mypy hearken/

# Linting
ruff check hearken/ tests/
```

## Roadmap

- v0.1: EnergyVAD, core pipeline ✓
- v0.2: WebRTC VAD support
- v0.3: Async transcriber support
- v0.4: Silero VAD (neural network)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Credits

Created by Nick Hehr (@hipsterbrown)
```

**Step 4: Commit**

```bash
git add hearken/__init__.py README.md
git commit -m "docs: create public API exports and README"
```

---

## Part 8: Examples

### Task 11: Create Example Scripts

**Files:**
- Create: `examples/basic_usage.py`
- Create: `examples/active_mode.py`

**Step 1: Create basic usage example**

Create `examples/basic_usage.py`:

```python
"""
Basic hearken usage example with speech_recognition adapters.

Demonstrates passive mode with automatic transcription.
"""

import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    # Setup speech_recognition components
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Adjust for ambient noise
    print("Calibrating for ambient noise... (1 second)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold set to: {recognizer.energy_threshold}")

    # Create hearken components
    audio_source = SpeechRecognitionSource(mic)
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    # Callback for transcripts
    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    # Create and start listener
    listener = Listener(
        source=audio_source,
        transcriber=transcriber,
        vad=EnergyVAD(dynamic=True),
        on_transcript=on_transcript,
    )

    print("\nListening... (Ctrl+C to stop)")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
```

**Step 2: Create active mode example**

Create `examples/active_mode.py`:

```python
"""
Active mode example - explicitly request speech segments.

Useful for conversational interfaces or when you need control
over when transcription happens.
"""

import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Calibrating for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    audio_source = SpeechRecognitionSource(mic)
    transcriber = SRTranscriber(recognizer)

    listener = Listener(
        source=audio_source,
        transcriber=transcriber,
        vad=EnergyVAD(),
    )

    listener.start()

    print("Active listening mode demo")
    print("Say something, then I'll transcribe it.\n")

    try:
        while True:
            print("Waiting for speech...")
            segment = listener.wait_for_speech()

            if segment:
                print(f"Got {segment.duration:.1f}s of audio, transcribing...")
                try:
                    text = transcriber.transcribe(segment)
                    print(f"You said: {text}\n")
                except sr.UnknownValueError:
                    print("Could not understand audio\n")
                except sr.RequestError as e:
                    print(f"API error: {e}\n")

    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
```

**Step 3: Test examples can be imported**

Run: `python -c "import examples.basic_usage; import examples.active_mode; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add examples/
git commit -m "docs: add example scripts

- basic_usage.py: passive mode with auto-transcription
- active_mode.py: active mode with manual control"
```

---

## Part 9: Final Integration & Polish

### Task 12: Final Testing & Verification

**Files:**
- Run all tests
- Verify package structure

**Step 1: Run complete test suite**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Run with coverage**

Run: `pytest --cov=hearken --cov-report=term-missing`
Expected: Coverage > 80%

**Step 3: Type check with mypy**

Run: `mypy hearken/ --strict`
Expected: No errors (or fix any type issues)

**Step 4: Format code**

Run: `black hearken/ tests/ examples/`
Expected: All files formatted

**Step 5: Lint code**

Run: `ruff check hearken/ tests/ examples/`
Expected: No issues (or fix any linting issues)

**Step 6: Test package can be built**

Run: `uv build`
Expected: Package built successfully

**Step 7: Commit any fixes**

```bash
git add -A
git commit -m "chore: apply formatting, linting, and type fixes"
```

### Task 13: Final Documentation Update

**Files:**
- Update: `CLAUDE.md`

**Step 1: Update CLAUDE.md to reflect hearken**

Update the project overview section in `CLAUDE.md` to reflect the new package name and structure:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hearken` is a robust speech recognition pipeline for Python that prevents audio drops during transcription. The core innovation is decoupling audio capture, voice activity detection (VAD), and transcription into independent threads with queue-based communication.

## Core Architecture

The pipeline consists of three independent threads:

1. **Capture Thread**: Continuously reads 30ms audio chunks from microphone, never blocks on downstream processing
2. **Detect Thread**: Runs voice activity detection and finite state machine (FSM) to segment continuous audio into discrete utterances
3. **Transcribe Thread**: Consumes utterances and calls recognition backends (optional, passive mode only)

**Critical Design Principle**: The capture thread must NEVER block. Queues use explicit backpressure - when full, drop data with logging rather than blocking upstream.

[... rest of CLAUDE.md updated similarly ...]
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for hearken package"
```

### Task 14: Create Final Release Commit

**Step 1: Verify all tests pass**

Run: `pytest -v`
Expected: All tests PASS

**Step 2: Create final commit**

```bash
git add -A
git commit -m "feat: hearken v0.1.0 MVP release

Complete implementation of hearken speech recognition pipeline:

Core Features:
- 3-thread architecture (capture, detect, transcribe)
- Clean abstractions (AudioSource, Transcriber, VAD)
- EnergyVAD implementation with dynamic calibration
- 4-state FSM detector with false start handling
- Active (wait_for_speech) and passive (callbacks) modes
- Async callback execution to prevent blocking
- Logging-based observability

Components:
- hearken.Listener: Main pipeline class
- hearken.EnergyVAD: Energy-based voice activity detection
- hearken.types: Core data types
- hearken.interfaces: Abstract base classes
- hearken.adapters.sr: speech_recognition adapters

Testing:
- Unit tests with >80% coverage
- Mock audio generation for deterministic tests
- Integration ready (SR adapters tested if installed)

Documentation:
- Complete README with examples
- basic_usage.py and active_mode.py examples
- API documentation in docstrings

Dependencies:
- Core: numpy only
- Optional: hearken[sr] for speech_recognition

Next: v0.2 will add WebRTC VAD support

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Step 3: Tag release**

```bash
git tag -a v0.1.0 -m "Hearken v0.1.0 MVP Release"
```

---

## Completion

The hearken v0.1.0 MVP implementation is complete!

**What was built:**
- ✅ Complete 3-thread pipeline architecture
- ✅ Core abstractions (AudioSource, Transcriber, VAD)
- ✅ EnergyVAD with dynamic calibration
- ✅ 4-state FSM speech detector
- ✅ Listener with active and passive modes
- ✅ speech_recognition adapters
- ✅ Unit tests with >80% coverage
- ✅ Examples and documentation
- ✅ Package configuration with uv

**Ready for:**
- Publishing to PyPI
- User testing and feedback
- v0.2 development (WebRTC VAD)
