# WebRTC VAD Design

**Date:** 2025-12-02
**Feature:** WebRTC VAD Support (v0.2 Roadmap Item)

## Overview

Add WebRTC VAD as a more accurate alternative to EnergyVAD. WebRTC VAD uses signal processing techniques (spectral analysis, noise estimation) from Google's WebRTC project, providing better speech detection in noisy environments compared to simple energy thresholding.

### Key Constraints

- WebRTC VAD requires specific sample rates: 8000, 16000, 32000, or 48000 Hz
- Frame durations must be 10, 20, or 30 ms
- The current pipeline's 30ms frame size is compatible
- Sample rate must be validated at initialization

### Integration with Existing Architecture

WebRTC VAD implements the existing `VAD` interface from `interfaces.py`, making it a drop-in replacement for EnergyVAD:

```python
from hearken import Listener
from hearken.vad import WebRTCVAD

# Replace EnergyVAD with WebRTCVAD
vad = WebRTCVAD(aggressiveness=1)
listener = Listener(vad=vad, ...)
```

The three-thread pipeline (capture → detect → transcribe) remains unchanged. WebRTCVAD simply provides more accurate `VADResult` outputs to the SpeechDetector FSM.

## Implementation Details

### Class Structure (`hearken/vad/webrtc.py`)

```python
class WebRTCVAD(VAD):
    def __init__(self, aggressiveness: int = 1):
        """
        Args:
            aggressiveness: 0-3, higher = more aggressive filtering
                0: Least aggressive (more speech detected)
                1: Quality mode (default)
                2: Low bitrate mode
                3: Very aggressive (less speech detected)
        """
        # Validate aggressiveness
        # Create webrtcvad.Vad instance
        # Store sample_rate/frame_duration from first chunk

    def process(self, chunk: AudioChunk) -> VADResult:
        # First call: validate sample_rate and frame_duration
        # Call vad.is_speech(chunk.data, sample_rate)
        # Return VADResult(is_speech, confidence)

    def reset(self) -> None:
        # WebRTC VAD is stateless per-frame, but recreate instance
        # to ensure clean state between utterances

    @property
    def required_sample_rate(self) -> int | None:
        return None  # Validated at runtime, not enforced upfront

    @property
    def required_frame_duration_ms(self) -> int | None:
        return None  # Validated at runtime
```

### Validation Strategy

On the first `process()` call, validate that `chunk.sample_rate` is in `[8000, 16000, 32000, 48000]` and frame duration is in `[10, 20, 30]` ms. Raise `ValueError` with clear guidance if invalid:

```
ValueError: WebRTC VAD requires sample rate of 8000, 16000, 32000, or 48000 Hz.
Got 44100 Hz. Configure your AudioSource with a supported sample rate.
```

### Confidence Calculation

Unlike EnergyVAD which computes a gradient, WebRTC VAD returns binary true/false. Map to confidence:
- `is_speech=True` → confidence=1.0
- `is_speech=False` → confidence=0.0

Simple and accurate to the underlying library's behavior.

## Testing Strategy

### Test Coverage (`tests/test_vad_webrtc.py`)

Using synthetic audio generated with numpy:

1. **Initialization tests:**
   - Valid aggressiveness values (0-3) succeed
   - Invalid aggressiveness (<0, >3) raises ValueError

2. **Validation tests:**
   - Supported sample rates (8000, 16000, 32000, 48000 Hz) work
   - Unsupported sample rates (e.g., 44100 Hz) raise ValueError with clear message
   - Supported frame durations (10, 20, 30 ms) work
   - Unsupported frame durations raise ValueError

3. **Functionality tests:**
   - Silence (zeros) returns `VADResult(is_speech=False, confidence=0.0)`
   - High-energy synthetic audio (sine wave) triggers speech detection
   - `reset()` recreates the VAD instance cleanly

4. **Integration test:**
   - Create mock AudioChunk with known sample rate/duration
   - Verify WebRTCVAD processes it and returns valid VADResult

## Dependency Management

### pyproject.toml Updates

```toml
[project.optional-dependencies]
sr = ["SpeechRecognition>=3.8", "PyAudio>=0.2.11"]
webrtc = ["webrtcvad-wheels>=2.0.10"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "mypy>=1.0", "ruff>=0.1.0"]
all = ["hearken[sr,webrtc]"]
```

Users install with:
- `pip install hearken[webrtc]` - Just WebRTC VAD support
- `pip install hearken[all]` - Everything including WebRTC

### Dependency Choice

Using `webrtcvad-wheels` instead of the original `webrtcvad`:
- Actively maintained fork
- Pre-built wheels for easier installation across platforms
- Same API as original webrtcvad
- Avoids C compiler requirements on user machines

## Documentation & Examples

### Public API Export (`hearken/vad/__init__.py`)

```python
from .energy import EnergyVAD

try:
    from .webrtc import WebRTCVAD
except ImportError:
    # webrtcvad-wheels not installed
    pass

__all__ = ["EnergyVAD", "WebRTCVAD"]
```

This allows `from hearken.vad import WebRTCVAD` when the optional dependency is installed, gracefully handles missing dependency.

### Documentation Updates

1. **README.md** - Add WebRTC VAD to features list:
   - Mention improved accuracy over EnergyVAD
   - Note installation: `pip install hearken[webrtc]`
   - List sample rate constraints (8/16/32/48 kHz)

2. **CLAUDE.md** - Update:
   - Add WebRTC VAD to module structure
   - Document constraints and validation behavior
   - Note v0.2 milestone completed

### Example Usage (`examples/webrtc_vad.py`)

```python
"""Example: Using WebRTC VAD for improved accuracy."""
import speech_recognition as sr
from hearken import Listener
from hearken.vad import WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=16000)  # WebRTC VAD needs 16kHz

    print("Calibrating for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    audio_source = SpeechRecognitionSource(mic)
    vad = WebRTCVAD(aggressiveness=2)  # Low bitrate mode
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    listener = Listener(
        source=audio_source,
        vad=vad,
        transcriber=transcriber,
        on_transcript=on_transcript,
    )

    print("\nListening with WebRTC VAD... (Ctrl+C to stop)")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
```

## Module Structure

After implementation, the module structure will be:

```
hearken/
├── __init__.py           # Public API exports
├── listener.py           # Listener class (main pipeline)
├── types.py              # Core data types
├── interfaces.py         # Abstract interfaces
├── detector.py           # SpeechDetector FSM
├── vad/
│   ├── __init__.py       # Export EnergyVAD and WebRTCVAD
│   ├── energy.py         # EnergyVAD implementation
│   └── webrtc.py         # WebRTCVAD implementation (NEW)
└── adapters/
    ├── __init__.py
    └── sr.py             # speech_recognition adapters
```

## Roadmap Impact

This feature completes the v0.2 milestone. Update roadmap:

**Completed:**
- ✅ v0.1: MVP with EnergyVAD, 3-thread architecture, active/passive modes
- ✅ v0.2: WebRTC VAD support

**Remaining:**
- v0.3: Async transcriber support
- v0.4: Silero VAD (neural network)

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python library | webrtcvad-wheels | Active maintenance, pre-built wheels |
| Sample rate handling | Strict validation with clear errors | Keep library simple, users configure AudioSource |
| Aggressiveness API | Constructor parameter (default=1) | Simple, matches underlying library |
| Testing approach | Synthetic audio only | Clean, no binary files, manual testing before release |
| Dependency packaging | New `[webrtc]` optional group | Modular, users install only what they need |
| Module organization | `hearken/vad/webrtc.py` | Consistent with EnergyVAD pattern |
