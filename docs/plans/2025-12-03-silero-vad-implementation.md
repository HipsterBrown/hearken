# Silero VAD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add neural network-based voice activity detection using Silero VAD v5 with ONNX Runtime

**Architecture:** Implement `SileroVAD` class following the same `VAD` interface pattern as EnergyVAD and WebRTCVAD. Use ONNX Runtime for efficient inference, download model on first use with configurable caching, strict 16kHz sample rate requirement.

**Tech Stack:** Python 3.11+, onnxruntime>=1.16.0, numpy, pytest with mocking

---

## Task 1: Add onnxruntime dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add silero optional dependency**

In `pyproject.toml`, update the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
sr = ["SpeechRecognition>=3.8.1"]
webrtc = ["webrtcvad-wheels>=2.0.10"]
silero = ["onnxruntime>=1.16.0"]
all = ["hearken[sr,webrtc,silero]"]
```

**Step 2: Verify dependency added**

Run: `grep -A 5 "\[project.optional-dependencies\]" pyproject.toml`
Expected: Should show `silero = ["onnxruntime>=1.16.0"]` and updated `all` line

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add onnxruntime dependency for Silero VAD"
```

---

## Task 2: Create SileroVAD class structure (TDD - Red)

**Files:**
- Create: `hearken/vad/silero.py`
- Create: `tests/test_vad_silero.py`

**Step 1: Write failing test for basic initialization**

Create `tests/test_vad_silero.py`:

```python
"""Tests for Silero VAD implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from hearken.vad.silero import SileroVAD


def test_silero_vad_creation_default():
    """Test SileroVAD creation with default parameters."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad = SileroVAD()
        assert vad._threshold == 0.5
        assert vad._validated is False
        assert vad._sample_rate is None
```

**Step 2: Run test to verify it fails**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py::test_silero_vad_creation_default -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'hearken.vad.silero'"

**Step 3: Create minimal SileroVAD class**

Create `hearken/vad/silero.py`:

```python
"""Silero VAD implementation using ONNX Runtime."""

import os
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for SileroVAD. "
        "Install with: pip install hearken[silero]"
    )

from hearken.interfaces import VAD
from hearken.types import AudioChunk, VADResult


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

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hearken"
    DEFAULT_MODEL_NAME = "silero_vad_v5.onnx"

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[str] = None
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {threshold}\n"
                "Lower values (e.g., 0.3) are more sensitive.\n"
                "Higher values (e.g., 0.7) are more conservative."
            )

        self._threshold = threshold
        self._model_path = self._resolve_model_path(model_path)
        self._ensure_model_downloaded()
        self._session = ort.InferenceSession(self._model_path)
        self._validated = False
        self._sample_rate: Optional[int] = None

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve model path from parameter, env var, or default."""
        if model_path:
            return model_path

        env_path = os.environ.get("HEARKEN_SILERO_MODEL_PATH")
        if env_path:
            return env_path

        return str(self.DEFAULT_CACHE_DIR / self.DEFAULT_MODEL_NAME)

    def _ensure_model_downloaded(self) -> None:
        """Download model if it doesn't exist at resolved path."""
        # Placeholder - will implement in next task
        pass

    def process(self, chunk: AudioChunk) -> VADResult:
        """Process audio chunk and return VAD result."""
        # Placeholder - will implement in later task
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset VAD state."""
        # Placeholder - will implement in later task
        raise NotImplementedError()
```

**Step 4: Run test to verify it passes**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py::test_silero_vad_creation_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add hearken/vad/silero.py tests/test_vad_silero.py
git commit -m "test: add SileroVAD initialization test (TDD - Red)"
```

---

## Task 3: Add threshold validation tests (TDD - Red)

**Files:**
- Modify: `tests/test_vad_silero.py`

**Step 1: Write failing tests for threshold validation**

Add to `tests/test_vad_silero.py`:

```python
def test_silero_vad_creation_with_threshold():
    """Test SileroVAD creation with custom threshold."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad = SileroVAD(threshold=0.7)
        assert vad._threshold == 0.7


def test_silero_vad_invalid_threshold_negative():
    """Test SileroVAD rejects negative threshold."""
    with pytest.raises(ValueError) as exc_info:
        SileroVAD(threshold=-0.1)

    assert "Threshold must be between 0.0 and 1.0" in str(exc_info.value)
    assert "got -0.1" in str(exc_info.value)


def test_silero_vad_invalid_threshold_too_high():
    """Test SileroVAD rejects threshold > 1.0."""
    with pytest.raises(ValueError) as exc_info:
        SileroVAD(threshold=1.5)

    assert "Threshold must be between 0.0 and 1.0" in str(exc_info.value)
    assert "got 1.5" in str(exc_info.value)


def test_silero_vad_boundary_thresholds():
    """Test SileroVAD accepts boundary values 0.0 and 1.0."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad_min = SileroVAD(threshold=0.0)
        assert vad_min._threshold == 0.0

        vad_max = SileroVAD(threshold=1.0)
        assert vad_max._threshold == 1.0
```

**Step 2: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k threshold -v`
Expected: All 4 threshold tests PASS (validation already implemented in Task 2)

**Step 3: Commit**

```bash
git add tests/test_vad_silero.py
git commit -m "test: add threshold validation tests for SileroVAD"
```

---

## Task 4: Implement model path resolution tests (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_silero.py`

**Step 1: Write failing tests for model path resolution**

Add to `tests/test_vad_silero.py`:

```python
def test_silero_vad_model_path_parameter():
    """Test model path from constructor parameter (highest priority)."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad = SileroVAD(model_path="/custom/path/model.onnx")
        assert vad._model_path == "/custom/path/model.onnx"


def test_silero_vad_model_path_env_var():
    """Test model path from environment variable."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}):
        vad = SileroVAD()
        assert vad._model_path == "/env/path/model.onnx"


def test_silero_vad_model_path_default():
    """Test default model path."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {}, clear=True):
        vad = SileroVAD()
        expected = str(Path.home() / ".cache" / "hearken" / "silero_vad_v5.onnx")
        assert vad._model_path == expected


def test_silero_vad_model_path_parameter_overrides_env():
    """Test parameter takes precedence over environment variable."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}):
        vad = SileroVAD(model_path="/param/path/model.onnx")
        assert vad._model_path == "/param/path/model.onnx"
```

**Step 2: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k model_path -v`
Expected: All 4 model_path tests PASS (resolution already implemented in Task 2)

**Step 3: Commit**

```bash
git add tests/test_vad_silero.py
git commit -m "test: add model path resolution tests for SileroVAD"
```

---

## Task 5: Implement model download functionality (TDD - Red/Green)

**Files:**
- Modify: `hearken/vad/silero.py`
- Modify: `tests/test_vad_silero.py`

**Step 1: Write failing test for model download**

Add to `tests/test_vad_silero.py`:

```python
from unittest.mock import mock_open


def test_silero_vad_downloads_model_if_missing():
    """Test model is downloaded if not present."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.Path.exists', return_value=False), \
         patch('hearken.vad.silero.Path.mkdir'), \
         patch('hearken.vad.silero.urllib.request.urlopen') as mock_urlopen, \
         patch('builtins.open', mock_open()) as mock_file:

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.read.return_value = b'fake model data'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        vad = SileroVAD()

        # Verify download was attempted
        mock_urlopen.assert_called_once()
        assert SileroVAD.MODEL_URL in str(mock_urlopen.call_args)


def test_silero_vad_skips_download_if_exists():
    """Test model download is skipped if file exists."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.Path.exists', return_value=True), \
         patch('hearken.vad.silero.urllib.request.urlopen') as mock_urlopen:

        vad = SileroVAD()

        # Verify download was NOT attempted
        mock_urlopen.assert_not_called()


def test_silero_vad_download_failure_raises_error():
    """Test clear error when model download fails."""
    with patch('hearken.vad.silero.Path.exists', return_value=False), \
         patch('hearken.vad.silero.Path.mkdir'), \
         patch('hearken.vad.silero.urllib.request.urlopen', side_effect=Exception("Network error")):

        with pytest.raises(RuntimeError) as exc_info:
            SileroVAD()

        error_msg = str(exc_info.value)
        assert "Failed to download Silero VAD model" in error_msg
        assert "Network error" in error_msg
        assert SileroVAD.MODEL_URL in error_msg
        assert "HEARKEN_SILERO_MODEL_PATH" in error_msg
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k download -v`
Expected: FAIL (download logic not implemented)

**Step 3: Implement model download**

Update `hearken/vad/silero.py` - add import at top:

```python
import urllib.request
```

Update `_ensure_model_downloaded` method:

```python
def _ensure_model_downloaded(self) -> None:
    """Download model if it doesn't exist at resolved path."""
    model_file = Path(self._model_path)

    # Skip download if file exists
    if model_file.exists():
        return

    # Create cache directory if needed
    model_file.parent.mkdir(parents=True, exist_ok=True)

    # Download model
    try:
        with urllib.request.urlopen(self.MODEL_URL) as response:
            model_data = response.read()

        # Write to file
        with open(model_file, 'wb') as f:
            f.write(model_data)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download Silero VAD model from GitHub: {e}\n"
            f"Please check your internet connection or download manually:\n"
            f"  URL: {self.MODEL_URL}\n"
            f"  Save to: {self._model_path}\n"
            f"Or set environment variable: HEARKEN_SILERO_MODEL_PATH=/path/to/model.onnx"
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k download -v`
Expected: All 3 download tests PASS

**Step 5: Commit**

```bash
git add hearken/vad/silero.py tests/test_vad_silero.py
git commit -m "feat: implement model download with caching for SileroVAD"
```

---

## Task 6: Implement sample rate validation (TDD - Red/Green)

**Files:**
- Modify: `hearken/vad/silero.py`
- Modify: `tests/test_vad_silero.py`

**Step 1: Write failing tests for sample rate validation**

Add to `tests/test_vad_silero.py`:

```python
from hearken.types import AudioChunk


def test_silero_vad_accepts_16khz():
    """Test SileroVAD accepts 16kHz audio."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        # Mock ONNX session to return controlled output
        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.8]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD()
        chunk = AudioChunk(
            data=b'\x00' * 960,  # 480 samples * 2 bytes = 960 bytes
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )

        result = vad.process(chunk)
        assert result is not None  # Should not raise


def test_silero_vad_rejects_non_16khz():
    """Test SileroVAD rejects non-16kHz audio."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        vad = SileroVAD()
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=8000,  # Wrong sample rate
            sample_width=2,
            timestamp=0.0
        )

        with pytest.raises(ValueError) as exc_info:
            vad.process(chunk)

        error_msg = str(exc_info.value)
        assert "Invalid sample rate for Silero VAD" in error_msg
        assert "8000 Hz" in error_msg
        assert "16000 Hz" in error_msg
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k "16khz or non_16khz" -v`
Expected: FAIL (process method not implemented)

**Step 3: Implement sample rate validation in process method**

Update `hearken/vad/silero.py` `process` method:

```python
def process(self, chunk: AudioChunk) -> VADResult:
    """Process audio chunk and return VAD result.

    Args:
        chunk: Audio chunk to process (must be 16kHz)

    Returns:
        VADResult with is_speech decision and confidence score

    Raises:
        ValueError: If sample rate is not 16kHz
    """
    # Validate 16kHz on first call
    if not self._validated:
        if chunk.sample_rate != 16000:
            raise ValueError(
                f"Invalid sample rate for Silero VAD: {chunk.sample_rate} Hz\n"
                f"Silero VAD requires 16000 Hz audio.\n"
                f"Please configure your AudioSource to use 16kHz sample rate."
            )
        self._sample_rate = chunk.sample_rate
        self._validated = True

    # Convert bytes to float32 numpy array
    audio_int16 = np.frombuffer(chunk.data, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    # Prepare input for ONNX model
    # Silero VAD expects shape: (batch_size, samples)
    audio_input = audio_float32.reshape(1, -1)

    # Run inference
    # Model outputs: (speech_prob, h, c)
    ort_inputs = {'input': audio_input}
    ort_outputs = self._session.run(None, ort_inputs)
    confidence = float(ort_outputs[0][0][0])

    # Apply threshold
    is_speech = confidence >= self._threshold

    return VADResult(is_speech=is_speech, confidence=confidence)
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k "16khz or non_16khz" -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
git add hearken/vad/silero.py tests/test_vad_silero.py
git commit -m "feat: implement sample rate validation and inference for SileroVAD"
```

---

## Task 7: Test threshold application (TDD - Red/Green)

**Files:**
- Modify: `tests/test_vad_silero.py`

**Step 1: Write tests for threshold application**

Add to `tests/test_vad_silero.py`:

```python
def test_silero_vad_threshold_below():
    """Test confidence below threshold returns is_speech=False."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        # Mock returns confidence=0.3
        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.3]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )

        result = vad.process(chunk)
        assert result.is_speech is False
        assert result.confidence == 0.3


def test_silero_vad_threshold_above():
    """Test confidence above threshold returns is_speech=True."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        # Mock returns confidence=0.7
        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.7]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )

        result = vad.process(chunk)
        assert result.is_speech is True
        assert result.confidence == 0.7


def test_silero_vad_threshold_boundary():
    """Test confidence exactly at threshold returns is_speech=True."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        # Mock returns confidence=0.5
        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.5]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )

        result = vad.process(chunk)
        assert result.is_speech is True  # >= threshold
        assert result.confidence == 0.5


def test_silero_vad_custom_threshold():
    """Test custom threshold value works correctly."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        # Mock returns confidence=0.6
        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.6]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD(threshold=0.7)
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )

        result = vad.process(chunk)
        assert result.is_speech is False  # 0.6 < 0.7
        assert result.confidence == 0.6
```

**Step 2: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k "threshold_below or threshold_above or threshold_boundary or custom_threshold" -v`
Expected: All 4 tests PASS (threshold logic already implemented in Task 6)

**Step 3: Commit**

```bash
git add tests/test_vad_silero.py
git commit -m "test: add threshold application tests for SileroVAD"
```

---

## Task 8: Implement reset functionality (TDD - Red/Green)

**Files:**
- Modify: `hearken/vad/silero.py`
- Modify: `tests/test_vad_silero.py`

**Step 1: Write failing tests for reset**

Add to `tests/test_vad_silero.py`:

```python
def test_silero_vad_reset():
    """Test reset reinitializes ONNX session."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session_class, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        mock_session_instance = MagicMock()
        mock_session_class.return_value = mock_session_instance

        vad = SileroVAD()

        # Verify session created once on init
        assert mock_session_class.call_count == 1

        # Reset
        vad.reset()

        # Verify session created again
        assert mock_session_class.call_count == 2


def test_silero_vad_reset_clears_validation_state():
    """Test reset clears validation state."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):

        mock_output = MagicMock()
        mock_output.run.return_value = ([[[0.7]]], None, None)
        mock_session.return_value = mock_output

        vad = SileroVAD()

        # Process chunk to trigger validation
        chunk = AudioChunk(
            data=b'\x00' * 960,
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0
        )
        vad.process(chunk)

        assert vad._validated is True
        assert vad._sample_rate == 16000

        # Reset
        vad.reset()

        # Verify validation state cleared
        assert vad._validated is False
        assert vad._sample_rate is None
```

**Step 2: Run tests to verify they fail**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k reset -v`
Expected: FAIL (reset not implemented)

**Step 3: Implement reset method**

Update `hearken/vad/silero.py` `reset` method:

```python
def reset(self) -> None:
    """Reset VAD state by reinitializing ONNX session."""
    self._session = ort.InferenceSession(self._model_path)
    self._validated = False
    self._sample_rate = None
```

**Step 4: Run tests to verify they pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest tests/test_vad_silero.py -k reset -v`
Expected: Both reset tests PASS

**Step 5: Commit**

```bash
git add hearken/vad/silero.py tests/test_vad_silero.py
git commit -m "feat: implement reset functionality for SileroVAD"
```

---

## Task 9: Export SileroVAD from vad module

**Files:**
- Modify: `hearken/vad/__init__.py`
- Modify: `hearken/__init__.py`

**Step 1: Add conditional export in vad/__init__.py**

Update `hearken/vad/__init__.py`:

```python
"""Voice Activity Detection implementations."""

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

**Step 2: Add conditional export in hearken/__init__.py**

Update `hearken/__init__.py` - add import after WebRTCVAD section:

```python
try:
    from .vad.silero import SileroVAD
    _silero_available = True
except ImportError:
    _silero_available = False
```

Update `__all__` list building at the end:

```python
__all__ = [
    "Listener",
    "AudioChunk",
    "SpeechSegment",
    "VADResult",
    "DetectorConfig",
    "DetectorState",
    "AudioSource",
    "Transcriber",
    "VAD",
    "EnergyVAD",
]

if _sr_available:
    __all__.extend(["SpeechRecognitionSource", "SRTranscriber"])

if _webrtc_available:
    __all__.append("WebRTCVAD")

if _silero_available:
    __all__.append("SileroVAD")
```

**Step 3: Test imports work**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/python -c "from hearken import SileroVAD; print('SileroVAD imported successfully')"`
Expected: Should fail with ImportError about onnxruntime (expected - not installed in venv yet)

**Step 4: Install silero extras and test again**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && uv pip install onnxruntime>=1.16.0`
Expected: onnxruntime installed successfully

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/python -c "from hearken import SileroVAD; print('SileroVAD imported successfully')"`
Expected: "SileroVAD imported successfully"

**Step 5: Commit**

```bash
git add hearken/vad/__init__.py hearken/__init__.py
git commit -m "feat: export SileroVAD from hearken module"
```

---

## Task 10: Create Silero VAD example

**Files:**
- Create: `examples/silero_vad.py`

**Step 1: Create example file**

Create `examples/silero_vad.py`:

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

**Step 2: Verify example syntax**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/python -m py_compile examples/silero_vad.py`
Expected: No output (successful compilation)

**Step 3: Commit**

```bash
git add examples/silero_vad.py
git commit -m "docs: add Silero VAD usage example"
```

---

## Task 11: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README.md features section**

In `README.md`, update the "Voice Activity Detection (VAD)" section under Features:

```markdown
### Voice Activity Detection (VAD)

- **EnergyVAD**: Simple energy-based detection with dynamic threshold calibration
- **WebRTCVAD**: Google WebRTC VAD for improved accuracy in noisy environments
  - Requires sample rates: 8000, 16000, 32000, or 48000 Hz
  - Configurable aggressiveness (0-3)
  - Install with: `pip install hearken[webrtc]`
- **SileroVAD**: Neural network-based VAD for superior accuracy
  - Requires 16kHz audio
  - Configurable sensitivity threshold
  - Automatic model download and caching
  - Install with: `pip install hearken[silero]`
```

**Step 2: Update README.md installation section**

In `README.md`, update the installation section:

```markdown
## Installation

```bash
# Basic installation (includes EnergyVAD)
pip install hearken

# With speech_recognition support
pip install hearken[sr]

# With WebRTC VAD support
pip install hearken[webrtc]

# With Silero VAD support (neural network)
pip install hearken[silero]

# All optional dependencies
pip install hearken[all]
```
```

**Step 3: Update CLAUDE.md module structure**

In `CLAUDE.md`, update the module structure section:

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
│   ├── webrtc.py         # WebRTCVAD implementation
│   └── silero.py         # SileroVAD implementation
└── adapters/
    ├── __init__.py
    └── sr.py             # speech_recognition adapters
```
```

**Step 4: Update CLAUDE.md version and roadmap**

In `CLAUDE.md`, update version section:

```markdown
## Version

Current version: **0.3.0** (Silero VAD Release)

- 3-thread architecture with clean abstractions
- EnergyVAD, WebRTCVAD, and SileroVAD implementations
- 4-state FSM detector
- Active and passive modes
- speech_recognition adapters

Roadmap:
- ✅ v0.1: MVP with EnergyVAD
- ✅ v0.2: WebRTC VAD support
- ✅ v0.3: Silero VAD (neural network)
- v0.4: Async transcriber support
```

**Step 5: Update CLAUDE.md Common Pitfalls**

In `CLAUDE.md`, add to Common Pitfalls section:

```markdown
6. **Silero VAD sample rate**: Silero VAD requires exactly 16kHz audio.
   Unlike WebRTC VAD which supports multiple rates, Silero is strict about 16kHz.
```

**Step 6: Update CLAUDE.md Dependencies**

In `CLAUDE.md`, update Optional dependencies section:

```markdown
### Optional
- `SpeechRecognition` >= 3.8 (recognition backends via adapters)
- `PyAudio` >= 0.2.11 (audio capture, installed via speech_recognition)
- `webrtcvad-wheels` >= 2.0.10 (WebRTC VAD support)
- `onnxruntime` >= 1.16.0 (Silero VAD support)

Install with: `pip install hearken[sr]` for speech_recognition support, `pip install hearken[webrtc]` for WebRTC VAD support, or `pip install hearken[silero]` for Silero VAD support
```

**Step 7: Verify changes**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && grep -A 10 "SileroVAD" README.md`
Expected: Should show SileroVAD documentation

**Step 8: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE.md for Silero VAD"
```

---

## Task 12: Run full test suite

**Files:**
- N/A (verification step)

**Step 1: Run all tests**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest -v`
Expected: All tests pass (including new Silero VAD tests)

**Step 2: Check test coverage**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest --cov=hearken --cov-report=term-missing`
Expected: Coverage report shows good coverage on silero.py (~90%)

**Step 3: Run type checking**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/mypy hearken/`
Expected: No type errors (may have warning about missing onnxruntime stubs - acceptable)

**Step 4: Run linting**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/ruff check hearken/ tests/ examples/`
Expected: No linting errors

**Step 5: Document results**

Note: No commit needed - this is verification only

---

## Task 13: Create manual testing checklist

**Files:**
- Create: `docs/manual-testing-silero-vad.md`

**Step 1: Create manual testing document**

Create `docs/manual-testing-silero-vad.md`:

```markdown
# Silero VAD Manual Testing Checklist

## Prerequisites

- Microphone configured for 16kHz
- Internet connection (for first run, model download)
- Install: `pip install hearken[all]`

## Test Scenarios

### 1. Basic Functionality (Clean Environment)

**Setup:**
```bash
python examples/silero_vad.py
```

**Test:**
- [ ] Speak clearly in quiet room
- [ ] Verify speech is detected and transcribed
- [ ] Verify silence periods don't trigger detection
- [ ] Stop and restart, verify model not re-downloaded

**Expected:**
- Accurate speech detection
- Clear transcriptions
- No false positives during silence
- Fast startup on second run (cached model)

---

### 2. Noisy Environment Comparison

**Setup:**
Run with background noise (music, fan, etc.)

**Test:**
- [ ] Run with EnergyVAD: `python examples/basic_usage.py`
- [ ] Run with WebRTCVAD: `python examples/webrtc_vad.py`
- [ ] Run with SileroVAD: `python examples/silero_vad.py`
- [ ] Compare false positive rates

**Expected:**
- SileroVAD should have fewer false positives than EnergyVAD
- SileroVAD comparable or better than WebRTCVAD

---

### 3. Threshold Sensitivity

**Test different threshold values:**

**Low threshold (0.3) - More Sensitive:**
```python
vad=SileroVAD(threshold=0.3)
```
- [ ] Verify catches soft speech
- [ ] May have more false positives

**Default threshold (0.5) - Balanced:**
```python
vad=SileroVAD(threshold=0.5)
```
- [ ] Verify good balance
- [ ] Normal speech detected reliably

**High threshold (0.7) - Conservative:**
```python
vad=SileroVAD(threshold=0.7)
```
- [ ] Verify fewer false positives
- [ ] May miss very soft speech

---

### 4. Model Caching

**Test:**
- [ ] Delete cache: `rm ~/.cache/hearken/silero_vad_v5.onnx`
- [ ] Run example, observe model download
- [ ] Check file exists: `ls -lh ~/.cache/hearken/silero_vad_v5.onnx`
- [ ] Run again, verify no re-download (fast startup)

**Expected:**
- First run downloads ~2MB model
- Subsequent runs use cached model
- File size ~2MB

---

### 5. Custom Model Path

**Test:**
```python
# Download model manually
mkdir -p /tmp/custom_models
# Copy model to /tmp/custom_models/silero.onnx

vad=SileroVAD(model_path="/tmp/custom_models/silero.onnx")
```

- [ ] Verify uses custom path
- [ ] Verify works correctly

---

### 6. Environment Variable

**Test:**
```bash
export HEARKEN_SILERO_MODEL_PATH="/tmp/custom_models/silero.onnx"
python examples/silero_vad.py
```

- [ ] Verify uses env var path
- [ ] Verify works correctly

---

### 7. Error Messages

**Test wrong sample rate:**
```python
mic = sr.Microphone(sample_rate=8000)  # Wrong rate
vad=SileroVAD()
```

- [ ] Verify clear error message
- [ ] Message mentions 16kHz requirement

**Test offline mode (no model, no internet):**
```bash
rm ~/.cache/hearken/silero_vad_v5.onnx
# Disconnect internet
python examples/silero_vad.py
```

- [ ] Verify clear error message
- [ ] Message includes manual download instructions
- [ ] Message includes URL and paths

---

### 8. Performance

**Test:**
- [ ] Run for 5+ minutes continuously
- [ ] Monitor CPU usage
- [ ] Monitor memory usage
- [ ] Verify no memory leaks
- [ ] Verify real-time performance (no lag)

**Expected:**
- CPU usage reasonable (<50% on modern CPU)
- Memory stable (no growth over time)
- No audio drops or lag

---

## Success Criteria

- [ ] All 8 test scenarios pass
- [ ] No crashes or unexpected errors
- [ ] Performance acceptable for real-time use
- [ ] Better accuracy than EnergyVAD in noisy conditions
- [ ] Model caching works reliably
- [ ] Error messages are clear and actionable
```

**Step 2: Commit**

```bash
git add docs/manual-testing-silero-vad.md
git commit -m "docs: add manual testing checklist for Silero VAD"
```

---

## Task 14: Final integration and version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `hearken/__init__.py`

**Step 1: Update version in pyproject.toml**

In `pyproject.toml`, update version:

```toml
[project]
name = "hearken"
version = "0.3.0"
```

**Step 2: Update version in hearken/__init__.py**

In `hearken/__init__.py`, update version near the top:

```python
__version__ = "0.3.0"
```

**Step 3: Verify all tests still pass**

Run: `cd ~/.config/superpowers/worktrees/hearken/feature-silero-vad && .venv/bin/pytest -v`
Expected: All tests pass

**Step 4: Commit version bump**

```bash
git add pyproject.toml hearken/__init__.py
git commit -m "chore: bump version to v0.3.0"
```

**Step 5: Summary**

The Silero VAD implementation is complete:
- ✅ ONNX Runtime integration with automatic model download
- ✅ 16kHz sample rate requirement with clear validation
- ✅ Configurable confidence threshold (default 0.5)
- ✅ Model caching in ~/.cache/hearken/
- ✅ Support for custom model paths via parameter or env var
- ✅ Comprehensive test coverage (~90%) with mocked ONNX runtime
- ✅ Documentation and examples
- ✅ Version bumped to 0.3.0

---

## Execution Notes

- All tasks follow strict TDD: test first, see it fail, implement, see it pass
- Each task is bite-sized (2-5 minutes)
- Frequent commits with clear messages
- Mock-based testing for fast CI without model downloads
- Manual testing checklist for real-world validation
