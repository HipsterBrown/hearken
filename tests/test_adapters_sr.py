import pytest

# Skip all tests if SpeechRecognition not installed
sr = pytest.importorskip("speech_recognition")

from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber
from hearken.types import SpeechSegment
import numpy as np


def test_speech_recognition_source_lifecycle():
    """Test SpeechRecognitionSource open/close."""
    mic = sr.Microphone()
    source = SpeechRecognitionSource(mic)

    source.open()
    assert source.sample_rate > 0
    assert source.sample_width > 0
    source.close()


def test_speech_recognition_source_context_manager():
    """Test SpeechRecognitionSource context manager."""
    mic = sr.Microphone()
    source = SpeechRecognitionSource(mic)

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
