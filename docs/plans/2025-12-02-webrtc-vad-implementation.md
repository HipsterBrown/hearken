# WebRTC VAD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add WebRTC VAD as a drop-in replacement for EnergyVAD with improved accuracy in noisy environments.

**Architecture:** Implement `WebRTCVAD` class in `hearken/vad/webrtc.py` that implements the `VAD` interface. Uses `webrtcvad-wheels` library with runtime validation for sample rate (8/16/32/48 kHz) and frame duration (10/20/30 ms) constraints. Binary confidence scores (0.0 or 1.0) based on WebRTC VAD's boolean output.

**Tech Stack:** Python 3.11+, webrtcvad-wheels >=2.0.10, pytest, numpy

---

## Task 1: Add webrtcvad-wheels Dependency

**Files:**
- Modify: `pyproject.toml:31-45`

**Step 1: Add webrtc optional dependency group**

In `pyproject.toml`, add new `webrtc` optional dependency group after the `sr` group:

```toml
[project.optional-dependencies]
sr = [
    "SpeechRecognition>=3.8",
    "PyAudio>=0.2.11",
]
webrtc = [
    "webrtcvad-wheels>=2.0.10",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "mypy>=1.0",
    "ruff>=0.1.0",
]
all = [
    "hearken[sr,webrtc]",
]
```

**Step 2: Install the new dependency**

Run: `uv sync --extra webrtc`
Expected: Successfully installs webrtcvad-wheels

**Step 3: Verify installation**

Run: `uv pip list | grep webrtcvad`
Expected: Shows webrtcvad-wheels package installed

**Step 4: Commit dependency addition**

```bash
git add pyproject.toml
git commit -m "feat(deps): add webrtcvad-wheels optional dependency

Add [webrtc] optional dependency group for WebRTC VAD support.
Users can install with: pip install hearken[webrtc]"
```

---

## Task 2: Create WebRTCVAD Class Structure (TDD - Red Phase)

**Files:**
- Create: `tests/test_vad_webrtc.py`
- Create: `hearken/vad/webrtc.py`

**Step 1: Write failing test for WebRTCVAD initialization**

Create `tests/test_vad_webrtc.py`:

```python
"""Tests for WebRTC VAD implementation."""
import pytest
from hearken.vad.webrtc import WebRTCVAD


def test_webrtc_vad_creation_default():
    """Test WebRTCVAD initialization with default aggressiveness."""
    vad = WebRTCVAD()

    # Should create successfully with default aggressiveness=1
    assert vad is not None


def test_webrtc_vad_creation_with_aggressiveness():
    """Test WebRTCVAD initialization with custom aggressiveness."""
    vad = WebRTCVAD(aggressiveness=2)

    assert vad is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_creation_default -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'hearken.vad.webrtc'"

**Step 3: Create minimal WebRTCVAD class**

Create `hearken/vad/webrtc.py`:

```python
"""WebRTC-based voice activity detection."""
import logging
from typing import Optional

try:
    import webrtcvad
except ImportError:
    raise ImportError(
        "webrtcvad-wheels is required for WebRTCVAD. "
        "Install with: pip install hearken[webrtc]"
    )

from ..interfaces import VAD
from ..types import AudioChunk, VADResult

logger = logging.getLogger("hearken")


class WebRTCVAD(VAD):
    """
    WebRTC-based voice activity detection.

    Uses Google's WebRTC VAD for robust speech detection.
    More accurate than EnergyVAD in noisy environments.

    Constraints:
    - Sample rate must be 8000, 16000, 32000, or 48000 Hz
    - Frame duration must be 10, 20, or 30 ms
    """

    SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
    SUPPORTED_FRAME_DURATIONS_MS = [10, 20, 30]

    def __init__(self, aggressiveness: int = 1):
        """
        Initialize WebRTC VAD.

        Args:
            aggressiveness: Aggressiveness mode (0-3)
                0: Least aggressive (more speech detected)
                1: Quality mode (default)
                2: Low bitrate mode
                3: Very aggressive (less speech detected)

        Raises:
            ValueError: If aggressiveness is not in range 0-3
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError(
                f"Aggressiveness must be 0-3, got {aggressiveness}"
            )

        self._aggressiveness = aggressiveness
        self._vad = webrtcvad.Vad(aggressiveness)
        self._validated = False
        self._sample_rate: Optional[int] = None

    def process(self, chunk: AudioChunk) -> VADResult:
        """Process audio chunk and return speech detection result."""
        # TODO: Implement validation and processing
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset internal state between utterances."""
        # Recreate VAD instance for clean state
        self._vad = webrtcvad.Vad(self._aggressiveness)

    @property
    def required_sample_rate(self) -> Optional[int]:
        """Required sample rate, or None if flexible."""
        return None  # Validated at runtime

    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        """Required frame duration in ms, or None if flexible."""
        return None  # Validated at runtime
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_creation_default -v`
Expected: PASS

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_creation_with_aggressiveness -v`
Expected: PASS

**Step 5: Commit basic structure**

```bash
git add tests/test_vad_webrtc.py hearken/vad/webrtc.py
git commit -m "feat(vad): add WebRTCVAD class skeleton

- Add WebRTCVAD class with aggressiveness parameter
- Implement __init__, reset, and property methods
- Add tests for initialization
- process() method stubbed for next task"
```

---

## Task 3: Add Aggressiveness Validation Tests (TDD - Red Phase)

**Files:**
- Modify: `tests/test_vad_webrtc.py`

**Step 1: Write failing tests for invalid aggressiveness**

Add to `tests/test_vad_webrtc.py`:

```python
def test_webrtc_vad_invalid_aggressiveness_negative():
    """Test WebRTCVAD rejects negative aggressiveness."""
    with pytest.raises(ValueError, match="Aggressiveness must be 0-3"):
        WebRTCVAD(aggressiveness=-1)


def test_webrtc_vad_invalid_aggressiveness_too_high():
    """Test WebRTCVAD rejects aggressiveness > 3."""
    with pytest.raises(ValueError, match="Aggressiveness must be 0-3"):
        WebRTCVAD(aggressiveness=4)


def test_webrtc_vad_all_valid_aggressiveness_modes():
    """Test all valid aggressiveness modes (0-3)."""
    for mode in [0, 1, 2, 3]:
        vad = WebRTCVAD(aggressiveness=mode)
        assert vad is not None
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py -v -k "aggressiveness"`
Expected: All aggressiveness tests PASS (validation already implemented)

**Step 3: Commit aggressiveness validation tests**

```bash
git add tests/test_vad_webrtc.py
git commit -m "test(vad): add aggressiveness validation tests

- Test invalid negative aggressiveness
- Test invalid aggressiveness > 3
- Test all valid modes 0-3"
```

---

## Task 4: Implement Sample Rate Validation (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_webrtc.py`
- Modify: `hearken/vad/webrtc.py`

**Step 1: Write failing test for sample rate validation**

Add to `tests/test_vad_webrtc.py`:

```python
import numpy as np
import time
from hearken.types import AudioChunk


def create_test_chunk(
    sample_rate: int = 16000,
    duration_ms: int = 30,
    amplitude: int = 1000
) -> AudioChunk:
    """Create a test audio chunk with synthetic audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = np.random.randint(-amplitude, amplitude, size=num_samples, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )


def test_webrtc_vad_supported_sample_rates():
    """Test WebRTC VAD accepts all supported sample rates."""
    vad = WebRTCVAD()

    for sample_rate in [8000, 16000, 32000, 48000]:
        chunk = create_test_chunk(sample_rate=sample_rate)
        result = vad.process(chunk)  # Should not raise
        assert result is not None


def test_webrtc_vad_unsupported_sample_rate():
    """Test WebRTC VAD rejects unsupported sample rate."""
    vad = WebRTCVAD()
    chunk = create_test_chunk(sample_rate=44100)  # CD quality, not supported

    with pytest.raises(ValueError, match="WebRTC VAD requires sample rate"):
        vad.process(chunk)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_supported_sample_rates -v`
Expected: FAIL with "NotImplementedError"

**Step 3: Implement sample rate validation in process()**

Update `hearken/vad/webrtc.py` `process()` method:

```python
def process(self, chunk: AudioChunk) -> VADResult:
    """Process audio chunk and return speech detection result."""
    # Validate sample rate on first call
    if not self._validated:
        if chunk.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"WebRTC VAD requires sample rate of {self.SUPPORTED_SAMPLE_RATES} Hz. "
                f"Got {chunk.sample_rate} Hz. "
                f"Configure your AudioSource with a supported sample rate."
            )
        self._sample_rate = chunk.sample_rate
        self._validated = True

    # Call WebRTC VAD
    is_speech = self._vad.is_speech(chunk.data, self._sample_rate)

    # Map boolean to confidence
    confidence = 1.0 if is_speech else 0.0

    return VADResult(is_speech=is_speech, confidence=confidence)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py -v -k "sample_rate"`
Expected: All sample rate tests PASS

**Step 5: Commit sample rate validation**

```bash
git add tests/test_vad_webrtc.py hearken/vad/webrtc.py
git commit -m "feat(vad): implement sample rate validation for WebRTCVAD

- Validate sample rate on first process() call
- Support 8000, 16000, 32000, 48000 Hz
- Reject unsupported rates with clear error message
- Add tests for supported and unsupported sample rates"
```

---

## Task 5: Implement Frame Duration Validation (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_webrtc.py`
- Modify: `hearken/vad/webrtc.py`

**Step 1: Write failing test for frame duration validation**

Add to `tests/test_vad_webrtc.py`:

```python
def test_webrtc_vad_supported_frame_durations():
    """Test WebRTC VAD accepts all supported frame durations."""
    vad = WebRTCVAD()

    for duration_ms in [10, 20, 30]:
        chunk = create_test_chunk(sample_rate=16000, duration_ms=duration_ms)
        result = vad.process(chunk)  # Should not raise
        assert result is not None


def test_webrtc_vad_unsupported_frame_duration():
    """Test WebRTC VAD rejects unsupported frame duration."""
    vad = WebRTCVAD()
    chunk = create_test_chunk(sample_rate=16000, duration_ms=25)  # Not 10/20/30

    with pytest.raises(ValueError, match="WebRTC VAD requires frame duration"):
        vad.process(chunk)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_unsupported_frame_duration -v`
Expected: FAIL (no validation yet)

**Step 3: Implement frame duration validation**

Update `hearken/vad/webrtc.py` `process()` method to add frame duration validation:

```python
def process(self, chunk: AudioChunk) -> VADResult:
    """Process audio chunk and return speech detection result."""
    # Validate sample rate and frame duration on first call
    if not self._validated:
        if chunk.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"WebRTC VAD requires sample rate of {self.SUPPORTED_SAMPLE_RATES} Hz. "
                f"Got {chunk.sample_rate} Hz. "
                f"Configure your AudioSource with a supported sample rate."
            )

        # Calculate frame duration from chunk
        num_samples = len(chunk.data) // chunk.sample_width
        duration_ms = int((num_samples / chunk.sample_rate) * 1000)

        if duration_ms not in self.SUPPORTED_FRAME_DURATIONS_MS:
            raise ValueError(
                f"WebRTC VAD requires frame duration of {self.SUPPORTED_FRAME_DURATIONS_MS} ms. "
                f"Got {duration_ms} ms. "
                f"Configure your AudioSource with a supported frame duration."
            )

        self._sample_rate = chunk.sample_rate
        self._validated = True

    # Call WebRTC VAD
    is_speech = self._vad.is_speech(chunk.data, self._sample_rate)

    # Map boolean to confidence
    confidence = 1.0 if is_speech else 0.0

    return VADResult(is_speech=is_speech, confidence=confidence)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py -v -k "frame_duration"`
Expected: All frame duration tests PASS

**Step 5: Commit frame duration validation**

```bash
git add tests/test_vad_webrtc.py hearken/vad/webrtc.py
git commit -m "feat(vad): implement frame duration validation for WebRTCVAD

- Validate frame duration on first process() call
- Support 10, 20, 30 ms frames
- Calculate duration from chunk size
- Reject unsupported durations with clear error message
- Add tests for supported and unsupported frame durations"
```

---

## Task 6: Test Speech Detection Functionality (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_webrtc.py`

**Step 1: Write tests for speech vs silence detection**

Add to `tests/test_vad_webrtc.py`:

```python
def create_silence_chunk(sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Create a chunk of silence (zeros)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    samples = np.zeros(num_samples, dtype=np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )


def create_speech_chunk(sample_rate: int = 16000, duration_ms: int = 30) -> AudioChunk:
    """Create a chunk with high-energy audio (simulated speech)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    # Generate sine wave with high amplitude
    t = np.linspace(0, duration_ms / 1000, num_samples)
    frequency = 300  # Human voice range
    samples = (5000 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

    return AudioChunk(
        data=samples.tobytes(),
        timestamp=time.monotonic(),
        sample_rate=sample_rate,
        sample_width=2,
    )


def test_webrtc_vad_detects_silence():
    """Test WebRTC VAD detects silence correctly."""
    vad = WebRTCVAD(aggressiveness=1)
    chunk = create_silence_chunk()

    result = vad.process(chunk)

    assert result.is_speech is False
    assert result.confidence == 0.0


def test_webrtc_vad_detects_speech():
    """Test WebRTC VAD detects speech-like audio."""
    vad = WebRTCVAD(aggressiveness=1)
    chunk = create_speech_chunk()

    result = vad.process(chunk)

    assert result.is_speech is True
    assert result.confidence == 1.0


def test_webrtc_vad_confidence_binary():
    """Test WebRTC VAD returns binary confidence (0.0 or 1.0)."""
    vad = WebRTCVAD()

    # Test multiple chunks
    for _ in range(5):
        chunk = create_silence_chunk()
        result = vad.process(chunk)
        assert result.confidence in [0.0, 1.0]

    for _ in range(5):
        chunk = create_speech_chunk()
        result = vad.process(chunk)
        assert result.confidence in [0.0, 1.0]
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py -v -k "detects"`
Expected: All detection tests PASS (implementation already complete)

**Step 3: Commit speech detection tests**

```bash
git add tests/test_vad_webrtc.py
git commit -m "test(vad): add speech detection tests for WebRTCVAD

- Test silence detection (zeros)
- Test speech detection (sine wave)
- Test binary confidence values (0.0 or 1.0)
- Add helper functions for creating test chunks"
```

---

## Task 7: Test Reset Functionality (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_webrtc.py`

**Step 1: Write test for reset() method**

Add to `tests/test_vad_webrtc.py`:

```python
def test_webrtc_vad_reset():
    """Test WebRTC VAD reset recreates internal VAD instance."""
    vad = WebRTCVAD(aggressiveness=2)

    # Process a chunk
    chunk = create_speech_chunk()
    result1 = vad.process(chunk)

    # Reset
    vad.reset()

    # Process another chunk - should work normally
    result2 = vad.process(chunk)

    assert result1.is_speech == result2.is_speech
    assert result1.confidence == result2.confidence


def test_webrtc_vad_reset_clears_validation_state():
    """Test reset allows revalidation with different sample rates."""
    vad = WebRTCVAD()

    # Process with 16kHz
    chunk1 = create_speech_chunk(sample_rate=16000)
    vad.process(chunk1)

    # Reset should clear validation state
    vad.reset()

    # Should be able to process with different sample rate
    # Note: In practice, same VAD instance should use same sample rate,
    # but reset should clear the validation flag
    chunk2 = create_speech_chunk(sample_rate=8000)
    result = vad.process(chunk2)
    assert result is not None
```

**Step 2: Run tests to verify current behavior**

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_reset -v`
Expected: PASS (reset already implemented)

Run: `uv run pytest tests/test_vad_webrtc.py::test_webrtc_vad_reset_clears_validation_state -v`
Expected: FAIL (validation state not cleared)

**Step 3: Update reset() to clear validation state**

Update `hearken/vad/webrtc.py` `reset()` method:

```python
def reset(self) -> None:
    """Reset internal state between utterances."""
    # Recreate VAD instance for clean state
    self._vad = webrtcvad.Vad(self._aggressiveness)
    # Clear validation state to allow revalidation
    self._validated = False
    self._sample_rate = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vad_webrtc.py -v -k "reset"`
Expected: All reset tests PASS

**Step 5: Commit reset functionality**

```bash
git add tests/test_vad_webrtc.py hearken/vad/webrtc.py
git commit -m "feat(vad): implement reset() for WebRTCVAD

- Reset clears validation state
- Recreates WebRTC VAD instance
- Allows revalidation with different audio parameters
- Add tests for reset functionality"
```

---

## Task 8: Export WebRTCVAD from Module

**Files:**
- Modify: `hearken/vad/__init__.py`
- Modify: `hearken/__init__.py`

**Step 1: Update vad module __init__.py**

Update `hearken/vad/__init__.py`:

```python
"""Voice activity detection implementations."""

from .energy import EnergyVAD

try:
    from .webrtc import WebRTCVAD
    __all__ = ['EnergyVAD', 'WebRTCVAD']
except ImportError:
    # webrtcvad-wheels not installed
    __all__ = ['EnergyVAD']
```

**Step 2: Update main module __init__.py**

Update `hearken/__init__.py` to conditionally export WebRTCVAD:

```python
"""
Hearken - Robust speech recognition pipeline for Python.

Decouples audio capture, voice activity detection, and transcription
into independent threads to prevent audio drops during processing.
"""

__version__ = "0.1.3"

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

try:
    from .vad.webrtc import WebRTCVAD
    _webrtc_available = True
except ImportError:
    _webrtc_available = False

# Build __all__ dynamically
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

if _webrtc_available:
    __all__.append("WebRTCVAD")
```

**Step 3: Test import with webrtc installed**

Run: `python -c "from hearken import WebRTCVAD; print('Success')"`
Expected: "Success" (no ImportError)

**Step 4: Test import from vad module**

Run: `python -c "from hearken.vad import WebRTCVAD; print('Success')"`
Expected: "Success"

**Step 5: Commit module exports**

```bash
git add hearken/vad/__init__.py hearken/__init__.py
git commit -m "feat(vad): export WebRTCVAD from public API

- Conditionally export WebRTCVAD if webrtcvad-wheels installed
- Update hearken.vad.__all__ with graceful ImportError handling
- Update hearken.__all__ to include WebRTCVAD when available
- Allow imports: from hearken import WebRTCVAD"
```

---

## Task 9: Create WebRTC VAD Example

**Files:**
- Create: `examples/webrtc_vad.py`

**Step 1: Create example file**

Create `examples/webrtc_vad.py`:

```python
"""
Example: Using WebRTC VAD for improved accuracy.

Demonstrates using WebRTC VAD instead of EnergyVAD for more robust
speech detection in noisy environments.

Requirements:
    pip install hearken[webrtc,sr]
"""

import speech_recognition as sr
from hearken import Listener
from hearken.vad import WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    # Setup speech_recognition components
    recognizer = sr.Recognizer()
    # WebRTC VAD requires supported sample rate (8/16/32/48 kHz)
    mic = sr.Microphone(sample_rate=16000)

    print("Calibrating for ambient noise... (1 second)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold set to: {recognizer.energy_threshold}")

    # Create hearken components with WebRTC VAD
    audio_source = SpeechRecognitionSource(mic)
    vad = WebRTCVAD(aggressiveness=2)  # Low bitrate mode
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    # Callback for transcripts
    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    # Create and start listener
    listener = Listener(
        source=audio_source,
        vad=vad,
        transcriber=transcriber,
        on_transcript=on_transcript,
    )

    print("\nListening with WebRTC VAD... (Ctrl+C to stop)")
    print("Aggressiveness mode: 2 (low bitrate)")
    print("Sample rate: 16000 Hz")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
```

**Step 2: Test example runs (dry run without microphone)**

Run: `python examples/webrtc_vad.py --help 2>&1 || echo "Example file exists"`
Expected: File exists and imports successfully

**Step 3: Commit example**

```bash
git add examples/webrtc_vad.py
git commit -m "docs: add WebRTC VAD usage example

- Demonstrate WebRTC VAD with speech_recognition
- Show 16kHz sample rate configuration
- Use aggressiveness mode 2 (low bitrate)
- Include clear installation requirements"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md features section**

Add WebRTC VAD to the features list in `README.md`. Find the features section and add:

```markdown
### Voice Activity Detection (VAD)

- **EnergyVAD**: Simple energy-based detection with dynamic threshold calibration
- **WebRTCVAD**: Google WebRTC VAD for improved accuracy in noisy environments
  - Requires sample rates: 8000, 16000, 32000, or 48000 Hz
  - Configurable aggressiveness (0-3)
  - Install with: `pip install hearken[webrtc]`
```

**Step 2: Update installation section in README.md**

Update the installation section to include optional dependencies:

```markdown
## Installation

```bash
# Basic installation (includes EnergyVAD)
pip install hearken

# With speech_recognition support
pip install hearken[sr]

# With WebRTC VAD support
pip install hearken[webrtc]

# All optional dependencies
pip install hearken[all]
```
```

**Step 3: Update CLAUDE.md module structure**

Update the module structure section in `CLAUDE.md`:

```markdown
### Module Structure

```
hearken/
├── __init__.py           # Public API exports
├── listener.py           # Listener class (main pipeline)
├── types.py              # Core data types
├── interfaces.py         # Abstract interfaces
├── detector.py           # SpeechDetector FSM
├── vad/
│   ├── __init__.py
│   ├── energy.py         # EnergyVAD implementation
│   └── webrtc.py         # WebRTCVAD implementation
└── adapters/
    ├── __init__.py
    └── sr.py             # speech_recognition adapters
```
```

**Step 4: Update CLAUDE.md version and roadmap**

Update version and roadmap in `CLAUDE.md`:

```markdown
## Version

Current version: **0.1.3** (WebRTC VAD Release)

- 3-thread architecture with clean abstractions
- EnergyVAD and WebRTCVAD implementations
- 4-state FSM detector
- Active and passive modes
- speech_recognition adapters

Roadmap:
- ✅ v0.1: MVP with EnergyVAD
- ✅ v0.2: WebRTC VAD support
- v0.3: Async transcriber support
- v0.4: Silero VAD (neural network)
```

**Step 5: Commit documentation updates**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update documentation for WebRTC VAD

- Add WebRTC VAD to features list
- Document installation options for [webrtc] extra
- Update module structure with webrtc.py
- Mark v0.2 roadmap item as completed
- Document sample rate constraints"
```

---

## Task 11: Run Full Test Suite

**Files:**
- N/A (verification only)

**Step 1: Run all WebRTC VAD tests**

Run: `uv run pytest tests/test_vad_webrtc.py -v`
Expected: All tests PASS

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (including existing tests)

**Step 3: Run with coverage**

Run: `uv run pytest --cov=hearken --cov-report=term-missing tests/test_vad_webrtc.py`
Expected: High coverage (>90%) for webrtc.py

**Step 4: Verify type checking**

Run: `uv run mypy hearken/vad/webrtc.py`
Expected: No type errors

**Step 5: Verify linting**

Run: `uv run ruff check hearken/vad/webrtc.py tests/test_vad_webrtc.py`
Expected: No linting errors

---

## Task 12: Manual Testing Checklist

**Files:**
- N/A (manual verification)

**Step 1: Test basic import**

```python
python -c "from hearken import WebRTCVAD; vad = WebRTCVAD(); print('OK')"
```

Expected: "OK"

**Step 2: Test with real microphone (if available)**

Run example: `python examples/webrtc_vad.py`

Verify:
- [ ] Microphone initializes correctly
- [ ] Speech is detected when speaking
- [ ] Silence is detected when quiet
- [ ] Transcription works (if internet available)
- [ ] Clean shutdown on Ctrl+C

**Step 3: Test aggressiveness modes**

Modify example to test different aggressiveness modes (0, 1, 2, 3).
Verify:
- [ ] Mode 0 is very sensitive (picks up more sounds)
- [ ] Mode 3 is very conservative (requires clear speech)

**Step 4: Test error messages**

Try with unsupported sample rate:

```python
from hearken import Listener, WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource
import speech_recognition as sr

mic = sr.Microphone(sample_rate=44100)  # Unsupported
source = SpeechRecognitionSource(mic)
vad = WebRTCVAD()
listener = Listener(source=source, vad=vad)
listener.start()
```

Expected: Clear error message about supported sample rates

**Step 5: Document manual test results**

Create a simple test report (not committed):
```
Manual Testing Results - WebRTC VAD
===================================
Date: [DATE]
Tester: [NAME]

✓ Basic import works
✓ Example runs without errors
✓ Speech detection works
✓ Silence detection works
✓ Error messages are clear
✓ All aggressiveness modes work

Notes:
- Tested on [OS/Platform]
- Microphone: [Device]
- Any issues: [None/List issues]
```

---

## Task 13: Final Integration and Version Bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `hearken/__init__.py`

**Step 1: Update version number**

Update version in `pyproject.toml`:

```toml
[project]
name = "hearken"
version = "0.2.0"
```

Update version in `hearken/__init__.py`:

```python
__version__ = "0.2.0"
```

**Step 2: Run full test suite one final time**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 3: Commit version bump**

```bash
git add pyproject.toml hearken/__init__.py
git commit -m "chore: bump version to v0.2.0

Release WebRTC VAD support (v0.2 roadmap milestone):
- Add WebRTCVAD class with aggressiveness modes
- Runtime validation for sample rate and frame duration
- Optional [webrtc] dependency group
- Comprehensive test coverage
- Example and documentation"
```

**Step 4: Create git tag**

```bash
git tag -a v0.2.0 -m "Release v0.2.0: WebRTC VAD Support"
```

**Step 5: Verify tag**

Run: `git tag -l -n1 v0.2.0`
Expected: Shows tag with message

---

## Completion Checklist

After all tasks are complete, verify:

- [ ] All tests pass (`pytest -v`)
- [ ] Type checking passes (`mypy hearken/`)
- [ ] Linting passes (`ruff check hearken/ tests/`)
- [ ] WebRTCVAD is importable: `from hearken import WebRTCVAD`
- [ ] WebRTCVAD is importable from submodule: `from hearken.vad import WebRTCVAD`
- [ ] Example runs without errors: `python examples/webrtc_vad.py`
- [ ] Documentation is updated (README.md, CLAUDE.md)
- [ ] Version bumped to 0.2.0
- [ ] All commits follow conventional commit format
- [ ] Git tag created for v0.2.0

## Post-Implementation

After implementation is complete and verified:

1. **Merge to main**: If using a feature branch, merge to main
2. **Push tags**: `git push --tags`
3. **Optional: Publish to PyPI**: `python -m build && twine upload dist/*`
4. **Update roadmap**: Mark v0.2 as completed, plan v0.3

---

## Notes for Implementer

- **TDD throughout**: Write tests first, see them fail, implement, see them pass
- **Commit frequently**: Each task should have 1-2 commits minimum
- **Use existing patterns**: Follow EnergyVAD structure closely
- **Test with real audio**: Manual testing with microphone is important
- **Clear error messages**: Users should know exactly how to fix issues
- **Import safety**: Handle missing webrtcvad-wheels gracefully

## Skills Referenced

- @superpowers:test-driven-development - Write tests first, watch fail, minimal code
- @superpowers:verification-before-completion - Run tests before claiming done
- @superpowers:systematic-debugging - If tests fail unexpectedly, investigate root cause
