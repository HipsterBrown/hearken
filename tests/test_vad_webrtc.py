"""Tests for WebRTC VAD implementation."""
import numpy as np
import time
import pytest
from hearken.vad.webrtc import WebRTCVAD
from hearken.types import AudioChunk


def test_webrtc_vad_creation_default():
    """Test WebRTCVAD initialization with default aggressiveness."""
    vad = WebRTCVAD()

    # Should create successfully with default aggressiveness=1
    assert vad is not None


def test_webrtc_vad_creation_with_aggressiveness():
    """Test WebRTCVAD initialization with custom aggressiveness."""
    vad = WebRTCVAD(aggressiveness=2)

    assert vad is not None


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
    for sample_rate in [8000, 16000, 32000, 48000]:
        vad = WebRTCVAD()  # Create fresh VAD for each sample rate
        chunk = create_test_chunk(sample_rate=sample_rate)
        result = vad.process(chunk)  # Should not raise
        assert result is not None


def test_webrtc_vad_unsupported_sample_rate():
    """Test WebRTC VAD rejects unsupported sample rate."""
    vad = WebRTCVAD()
    chunk = create_test_chunk(sample_rate=44100)  # CD quality, not supported

    with pytest.raises(ValueError, match="WebRTC VAD requires sample rate"):
        vad.process(chunk)
