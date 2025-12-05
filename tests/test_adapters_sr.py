import pytest
from unittest.mock import Mock, MagicMock

# Skip all tests if SpeechRecognition not installed
sr = pytest.importorskip("speech_recognition")

from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber
from hearken.types import SpeechSegment
import numpy as np


@pytest.fixture
def mock_microphone():
    """Create a mock Microphone that doesn't require actual audio device."""
    mic = Mock(spec=sr.Microphone)
    mic.SAMPLE_RATE = 16000
    mic.SAMPLE_WIDTH = 2

    # Mock the context manager protocol
    mic.__enter__ = Mock(return_value=mic)
    mic.__exit__ = Mock(return_value=None)

    # Mock the stream for read operations
    mic.stream = Mock()
    mic.stream.read = Mock(return_value=b"\x00" * 1024)

    return mic


def test_speech_recognition_source_lifecycle(mock_microphone):
    """Test SpeechRecognitionSource open/close."""
    source = SpeechRecognitionSource(mock_microphone)

    source.open()
    assert source.sample_rate > 0
    assert source.sample_width > 0
    assert mock_microphone.__enter__.called

    source.close()
    assert mock_microphone.__exit__.called


def test_speech_recognition_source_context_manager(mock_microphone):
    """Test SpeechRecognitionSource context manager."""
    source = SpeechRecognitionSource(mock_microphone)

    with source as s:
        assert s is source
        assert source.sample_rate > 0


def test_sr_transcriber_initialization():
    """Test SRTranscriber initialization."""
    recognizer = sr.Recognizer()
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    assert transcriber.recognizer is recognizer
    assert transcriber.method_name == 'recognize_google'


def test_sr_transcriber_invalid_method():
    """Test SRTranscriber with invalid method name."""
    recognizer = sr.Recognizer()

    try:
        transcriber = SRTranscriber(recognizer, method='invalid_method')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no method" in str(e).lower()


def test_sr_transcriber_creates_audio_data():
    """Test SRTranscriber converts SpeechSegment to AudioData."""
    recognizer = sr.Recognizer()
    transcriber = SRTranscriber(recognizer, method='recognize_sphinx')  # Offline

    # Create dummy segment
    samples = np.random.randint(-100, 100, size=16000, dtype=np.int16)
    segment = SpeechSegment(
        audio_data=samples.tobytes(),
        sample_rate=16000,
        sample_width=2,
        start_time=0.0,
        end_time=1.0,
    )

    # This will likely fail to recognize, but we're testing the conversion
    try:
        transcriber.transcribe(segment)
    except sr.UnknownValueError:
        pass  # Expected for random noise
    except sr.RequestError as e:
        # PocketSphinx may not be installed - that's OK
        if "PocketSphinx" in str(e):
            pass
        else:
            raise
