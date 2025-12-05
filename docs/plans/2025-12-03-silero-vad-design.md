# Silero VAD Design Document

**Date:** 2025-12-03
**Feature:** Silero VAD Support (v0.3.0)
**Status:** Design Complete

## Overview

Add neural network-based voice activity detection using Silero VAD v5 with ONNX Runtime. This provides superior accuracy compared to rule-based approaches (EnergyVAD, WebRTCVAD) while maintaining the same `VAD` interface for drop-in compatibility.

## Design Decisions

### 1. Model Loading Strategy: Eager Initialization

**Decision:** Load ONNX model during `__init__` (Option A)

**Rationale:**
- Simple, predictable API
- Follows WebRTCVAD pattern
- Immediate error feedback if model unavailable
- Initialization cost paid once at startup

**Alternatives Considered:**
- Lazy loading on first `process()` - defers errors, unpredictable latency
- External model management - complex API, pushes responsibility to user

### 2. Model Format: ONNX Only

**Decision:** Use ONNX Runtime exclusively (Option B)

**Rationale:**
- Smaller dependency (~50-100MB vs ~100-200MB for PyTorch)
- Optimized specifically for inference workloads
- Silero VAD v5 ONNX performance is excellent (800-1240x real-time)
- Simpler implementation - single code path

**Alternatives Considered:**
- PyTorch only - larger dependency, now comparable performance in v5
- Support both - added complexity, two code paths to maintain

### 3. Sample Rate: Strict 16kHz Requirement

**Decision:** Only accept 16kHz audio (Option A)

**Rationale:**
- Silero VAD trained primarily on 16kHz
- Best model performance at native sample rate
- Clear, predictable behavior
- Matches WebRTC VAD constraint pattern

**Alternatives Considered:**
- Auto-resample internally - adds heavy dependency (librosa), hidden cost
- Accept 8kHz/16kHz - may degrade accuracy
- Provide separate resampling utility - can add later if needed

### 4. Chunk Size: Flexible Processing

**Decision:** Accept any chunk size, process as-is (Option C)

**Rationale:**
- Silero VAD v5 handles variable chunk sizes well
- Current 30ms chunks (480 samples at 16kHz) are near-optimal
- Simplest implementation, no buffering needed
- Compatible with existing pipeline architecture

**Alternatives Considered:**
- Internal buffering - adds complexity and latency
- Require specific sizes - breaks compatibility with 30ms default

### 5. Model Distribution: Download on First Use

**Decision:** Download model on first initialization with configurable cache (Option B)

**Rationale:**
- Zero size impact for users who don't use Silero
- Python package data can't be conditionally installed by extras
- Configurable for offline environments
- Provides model path override flexibility

**Configuration:**
- Constructor parameter: `model_path=None` (auto-download)
- Environment variable: `HEARKEN_SILERO_MODEL_PATH`
- Default cache: `~/.cache/hearken/silero_vad_v5.onnx`

**Alternatives Considered:**
- Bundle in package - 2MB overhead for all installations
- User provides path - poor developer experience

### 6. Confidence Threshold: Configurable with Default

**Decision:** Constructor parameter `threshold=0.5` (Option B)

**Rationale:**
- Silero outputs probability scores (0.0-1.0)
- 0.5 is recommended default
- Users can tune for different environments (noisy vs clean)
- Confidence always available in `VADResult` for inspection

**Alternatives Considered:**
- Fixed 0.5 threshold - no tuning flexibility
- Sensitivity presets - less precise than numeric threshold
- No threshold, return raw confidence - breaks VAD interface contract

### 7. State Management: Reinitialize on Reset

**Decision:** Create new ONNX session on `reset()` (Option A)

**Rationale:**
- Guaranteed clean state
- Simple, robust implementation
- Reset is infrequent operation (cost acceptable)
- Less fragile to model architecture changes

**Alternatives Considered:**
- Reset state tensors - faster but requires understanding internal architecture
- External state management - more complex inference code

### 8. Testing Strategy: Mock-Based

**Decision:** Mock ONNX runtime for all tests (Option A)

**Rationale:**
- Fast test execution
- Deterministic, easy to test edge cases
- No model download in CI
- Can add integration tests later if maintenance issues arise

**Alternatives Considered:**
- Real model tests - slower, requires model download
- Hybrid approach - added complexity
- Manual only - no CI validation

### 9. Error Handling: Fail-Fast on Download Failure

**Decision:** Raise exception immediately during `__init__` if model unavailable (Option A)

**Rationale:**
- Consistent with eager loading decision
- Clear, actionable error messages
- User knows immediately there's a problem
- Fail-fast principle

**Error Message Template:**
```
Failed to download Silero VAD model from GitHub.
Please check your internet connection or download manually:
  URL: https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
  Save to: ~/.cache/hearken/silero_vad_v5.onnx
Or set environment variable: HEARKEN_SILERO_MODEL_PATH=/path/to/model.onnx
```

**Alternatives Considered:**
- Lazy failure - error happens later, harder to debug
- Retry with backoff - slow startup on persistent failures
- Background download - very complex, unpredictable

### 10. Dependency Group: `[silero]` Extra

**Decision:** New optional extra `pip install hearken[silero]` (Option A)

**Rationale:**
- Parallel to `[webrtc]` and `[sr]` pattern
- Clear, descriptive naming
- Granular control over dependencies
- Update `[all]` to include silero

**Alternatives Considered:**
- `[neural]` or `[ml]` - less specific, unclear what's included
- Combine with `[all]` only - forces installing everything

## Architecture

### Class Structure

```python
from hearken.interfaces import VAD
from hearken.types import AudioChunk, VADResult
import onnxruntime as ort
import numpy as np
from typing import Optional

class SileroVAD(VAD):
    """Neural network-based VAD using Silero VAD v5 with ONNX Runtime.

    Requires 16kHz audio. Provides superior accuracy compared to rule-based
    approaches, especially in noisy environments.

    Args:
        threshold: Confidence threshold for speech detection (0.0-1.0).
                   Default 0.5. Lower = more sensitive, higher = more conservative.
        model_path: Path to ONNX model file. If None, downloads from GitHub
                    and caches in ~/.cache/hearken/silero_vad_v5.onnx.
                    Can also be set via HEARKEN_SILERO_MODEL_PATH env var.

    Raises:
        ValueError: If threshold not in [0.0, 1.0]
        RuntimeError: If model download fails
    """

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[str] = None
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be 0.0-1.0, got {threshold}")

        self._threshold = threshold
        self._model_path = self._resolve_model_path(model_path)
        self._ensure_model_downloaded()
        self._session = ort.InferenceSession(self._model_path)
        self._validated = False
        self._sample_rate: Optional[int] = None

    def process(self, chunk: AudioChunk) -> VADResult:
        # Validate 16kHz on first call
        # Convert to float32, normalize to [-1, 1]
        # Run inference
        # Apply threshold
        # Return VADResult(is_speech, confidence)
        ...

    def reset(self) -> None:
        # Reinitialize ONNX session
        ...
```

### Model Download Flow

1. Check `model_path` parameter (highest priority)
2. Check `HEARKEN_SILERO_MODEL_PATH` environment variable
3. Default to `~/.cache/hearken/silero_vad_v5.onnx`
4. If file doesn't exist at resolved path:
   - Create cache directory if needed
   - Download from `https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx`
   - Verify file size (~2MB)
   - Save to cache location
5. If download fails, raise `RuntimeError` with clear instructions

### Inference Flow

```
AudioChunk (bytes, 16kHz)
    ↓
Convert to float32 numpy array
    ↓
Normalize to [-1.0, 1.0] range
    ↓
ONNX Inference: output, h, c = session.run(inputs={audio, h, c})
    ↓
confidence = output[0][0]  # Extract probability
    ↓
is_speech = (confidence >= threshold)
    ↓
VADResult(is_speech, confidence)
```

### State Management

Silero VAD uses LSTM-like hidden states (h, c tensors) that carry information across frames. On `reset()`, we reinitialize the entire ONNX session, which automatically resets all internal state to zeros.

## Implementation Details

### Dependencies

**New Required:**
- `onnxruntime>=1.16.0` - ONNX inference engine

**Packaging:**
```toml
[project.optional-dependencies]
silero = ["onnxruntime>=1.16.0"]
all = ["hearken[sr,webrtc,silero]"]
```

### Module Structure

```
hearken/vad/
├── __init__.py      # Conditional export with ImportError handling
├── energy.py        # EnergyVAD
├── webrtc.py        # WebRTCVAD
└── silero.py        # SileroVAD (new)
```

**Conditional Export Pattern:**
```python
# hearken/vad/__init__.py
from .energy import EnergyVAD

try:
    from .webrtc import WebRTCVAD
    _webrtc_available = True
except ImportError:
    _webrtc_available = False

try:
    from .silero import SileroVAD
    _silero_available = True
except ImportError:
    _silero_available = False

__all__ = ['EnergyVAD']
if _webrtc_available:
    __all__.append('WebRTCVAD')
if _silero_available:
    __all__.append('SileroVAD')
```

### Error Messages

**Sample Rate Validation:**
```
Invalid sample rate for Silero VAD: {chunk.sample_rate} Hz
Silero VAD requires 16000 Hz audio.
Please configure your AudioSource to use 16kHz sample rate.
```

**Model Download Failure:**
```
Failed to download Silero VAD model from GitHub: {error}
Please check your internet connection or download manually:
  URL: https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
  Save to: {cache_path}
Or set environment variable: HEARKEN_SILERO_MODEL_PATH=/path/to/model.onnx
```

**Threshold Validation:**
```
Threshold must be between 0.0 and 1.0, got {threshold}
Lower values (e.g., 0.3) are more sensitive.
Higher values (e.g., 0.7) are more conservative.
```

## Testing Strategy

### Unit Tests (Mock-Based)

**Test File:** `tests/test_vad_silero.py`

1. **Initialization Tests**
   - Valid threshold values (0.0, 0.5, 1.0)
   - Invalid threshold values (-0.1, 1.5)
   - Model path resolution (parameter > env var > default)
   - Mock model download success/failure

2. **Sample Rate Validation Tests**
   - Accept 16kHz audio
   - Reject non-16kHz audio with clear error

3. **Inference Tests**
   - Mock ONNX session to return controlled confidence scores
   - Test threshold application:
     - confidence=0.3, threshold=0.5 → is_speech=False
     - confidence=0.7, threshold=0.5 → is_speech=True
     - confidence=0.5, threshold=0.5 → is_speech=True (boundary)
   - Verify confidence passthrough in VADResult

4. **Reset Tests**
   - Verify ONNX session is reinitialized
   - Check sample rate validation is cleared

5. **Error Handling Tests**
   - Download failure with clear error message
   - Invalid sample rate with helpful message
   - Invalid threshold with helpful message

**Target Coverage:** ~90% of silero.py

### Manual Testing Checklist

**Document:** `docs/manual-testing-silero-vad.md`

1. Test with real microphone at 16kHz
2. Compare accuracy vs EnergyVAD in clean environment
3. Compare accuracy vs WebRTCVAD in noisy environment
4. Test different threshold values (0.3, 0.5, 0.7)
5. Verify model caching (no re-download on second run)
6. Test offline mode with pre-downloaded model
7. Test with `model_path` parameter override
8. Test with `HEARKEN_SILERO_MODEL_PATH` environment variable

## Documentation Updates

### README.md

**Features Section:**
```markdown
### Voice Activity Detection (VAD)

- **EnergyVAD**: Simple energy-based detection with dynamic threshold calibration
- **WebRTCVAD**: Google WebRTC VAD for improved accuracy in noisy environments
- **SileroVAD**: Neural network-based VAD for superior accuracy
  - Requires 16kHz audio
  - Configurable sensitivity threshold
  - Automatic model download and caching
  - Install with: `pip install hearken[silero]`
```

**Installation Section:**
```markdown
```bash
# With Silero VAD support (neural network)
pip install hearken[silero]

# All optional dependencies
pip install hearken[all]
```
```

### CLAUDE.md

**Module Structure:**
```
hearken/vad/
├── __init__.py
├── energy.py         # EnergyVAD implementation
├── webrtc.py         # WebRTCVAD implementation
└── silero.py         # SileroVAD implementation (new)
```

**Version:**
```
Current version: 0.3.0 (Silero VAD Release)

Roadmap:
- ✅ v0.1: MVP with EnergyVAD
- ✅ v0.2: WebRTC VAD support
- ✅ v0.3: Silero VAD (neural network)
- v0.4: Async transcriber support
```

**Common Pitfalls:**
```
6. **Silero VAD sample rate**: Silero VAD requires exactly 16kHz audio.
   Unlike WebRTC VAD which supports multiple rates, Silero is strict about 16kHz.
```

### Example File

**File:** `examples/silero_vad.py`

```python
"""Example using Silero VAD for neural network-based voice activity detection.

Silero VAD provides superior accuracy compared to rule-based approaches,
especially in noisy environments. Requires 16kHz audio.
"""

import speech_recognition as sr
from hearken import Listener, SileroVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

# Setup recognizer with 16kHz sample rate
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

with mic as source:
    print("Calibrating for ambient noise...")
    recognizer.adjust_for_ambient_noise(source)

# Create listener with Silero VAD
# threshold=0.5 is default (lower=more sensitive, higher=more conservative)
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    vad=SileroVAD(threshold=0.5),
    on_transcript=lambda text, seg: print(f"You said: {text}")
)

print("Listening with Silero VAD (neural network)...")
print("Speak naturally. Press Ctrl+C to stop.")

listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    print("\nStopping...")
    listener.stop()
```

## Version

**Target Version:** v0.3.0

**Version Bump Locations:**
- `pyproject.toml`: `version = "0.3.0"`
- `hearken/__init__.py`: `__version__ = "0.3.0"`

## Summary

Silero VAD adds state-of-the-art neural network-based voice activity detection to hearken while maintaining the same clean `VAD` interface. The ONNX Runtime backend provides excellent performance (~800x real-time) with a reasonable dependency size (~50-100MB). Automatic model downloading with configurable caching makes it easy to use while supporting offline environments. The design prioritizes simplicity, robustness, and consistency with existing VAD implementations.
