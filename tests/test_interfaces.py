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
