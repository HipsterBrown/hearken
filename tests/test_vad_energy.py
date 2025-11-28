import numpy as np
import time
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
