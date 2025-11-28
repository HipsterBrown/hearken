from hearken.detector import SpeechDetector
from hearken.vad.energy import EnergyVAD
from hearken.types import DetectorState, DetectorConfig, AudioChunk
import numpy as np
import time


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


def test_detector_initialization():
    """Test SpeechDetector initialization."""
    vad = EnergyVAD()
    config = DetectorConfig()

    detector = SpeechDetector(vad=vad, config=config)

    assert detector.state == DetectorState.IDLE
    assert detector.vad is vad
    assert detector.config is config


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
