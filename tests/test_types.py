import time
from hearken.types import AudioChunk, SpeechSegment, VADResult, DetectorState, DetectorConfig


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
