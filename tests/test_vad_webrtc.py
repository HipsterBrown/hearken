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
