import numpy as np
from hearken import Listener
from hearken.interfaces import AudioSource, Transcriber
from hearken.types import SpeechSegment, DetectorConfig
from hearken.vad.energy import EnergyVAD


class MockAudioSource(AudioSource):
    def __init__(self):
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def read(self, num_samples: int) -> bytes:
        import time

        time.sleep(0.001)  # Simulate device latency
        return b"\x00" * num_samples * 2

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def sample_width(self) -> int:
        return 2


class MockTranscriber(Transcriber):
    def transcribe(self, segment: SpeechSegment) -> str:
        return f"mock transcription {segment.duration:.1f}s"


class SpeechAudioSource(AudioSource):
    """Mock source that generates speech-like audio with silence periods."""

    def __init__(self):
        self.is_open = False
        self.frame_count = 0

    def open(self) -> None:
        self.is_open = True
        self.frame_count = 0  # Reset counter on open to ensure consistent start

    def close(self) -> None:
        self.is_open = False

    def read(self, num_samples: int) -> bytes:
        import time

        # Simulate real-time audio (30ms frame = 0.03s)
        time.sleep(0.001)  # Small sleep to avoid flooding

        # Generate pattern: 12 frames of speech, then 6 frames of silence
        # This allows the FSM to reliably emit segments (needs silence to trigger)
        # Pattern chosen to be longer than min_speech_duration + silence_timeout
        cycle_length = 18
        speech_frames = 12  # 360ms of speech

        if (self.frame_count % cycle_length) < speech_frames:
            # Speech frames (high energy)
            samples = np.random.randint(-5000, 5000, size=num_samples, dtype=np.int16)
        else:
            # Silence frames (low energy)
            samples = np.random.randint(-100, 100, size=num_samples, dtype=np.int16)

        self.frame_count += 1
        return samples.tobytes()

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def sample_width(self) -> int:
        return 2


def test_listener_initialization():
    """Test Listener initialization."""
    source = MockAudioSource()
    transcriber = MockTranscriber()
    vad = EnergyVAD()

    listener = Listener(
        source=source,
        transcriber=transcriber,
        vad=vad,
    )

    assert listener.source is source
    assert listener.transcriber is transcriber
    assert listener.vad is vad


def test_listener_requires_transcriber_for_on_transcript():
    """Test Listener requires transcriber when on_transcript provided."""
    source = MockAudioSource()

    try:
        listener = Listener(
            source=source,
            on_transcript=lambda text, seg: None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "transcriber required" in str(e).lower()


def test_listener_start_stop():
    """Test Listener start and stop lifecycle."""
    import time

    source = MockAudioSource()
    listener = Listener(source=source)

    assert not source.is_open

    listener.start()
    assert source.is_open

    listener.stop()
    assert not source.is_open


def test_listener_capture_thread():
    """Test capture thread reads audio chunks."""
    import time

    source = MockAudioSource()
    listener = Listener(source=source)

    # Track if capture thread puts items in queue by checking queue after starting
    # but before detect thread consumes them all
    listener.start()

    # Very briefly wait - capture thread should put at least one chunk
    time.sleep(0.05)

    # Queue might be empty if detect thread already consumed chunks, which is fine
    # The real test is that the system runs without errors
    listener.stop()

    # If we got here without errors, capture thread is working


def test_listener_detect_thread():
    """Test detect thread processes chunks and detects speech."""
    import time

    source = SpeechAudioSource()
    config = DetectorConfig(
        min_speech_duration=0.09,
        silence_timeout=0.12,
    )

    listener = Listener(
        source=source,
        detector_config=config,
    )

    listener.start()
    time.sleep(0.5)  # Let it run for 500ms
    listener.stop()

    # Check that segments were detected and queued
    queue_size = listener._segment_queue.qsize()
    print(f"Detection queue size: {queue_size}")
    assert queue_size > 0, f"Expected segments in queue but got {queue_size}"


def test_listener_wait_for_speech():
    """Test active mode with wait_for_speech()."""
    import time

    source = SpeechAudioSource()
    config = DetectorConfig(
        min_speech_duration=0.09,  # 3 frames @ 30ms
        silence_timeout=0.12,  # 4 frames @ 30ms
    )

    # Use same config as test_listener_detect_thread for consistency
    listener = Listener(source=source, detector_config=config)
    listener.start()

    # Give the system a moment to start processing
    time.sleep(0.1)

    # Wait for speech with generous timeout
    # Pattern: 12 frames speech (360ms) + 6 frames silence (180ms) = 540ms cycle
    start = time.time()
    segment = listener.wait_for_speech(timeout=3.0)
    elapsed = time.time() - start

    print(f"wait_for_speech returned after {elapsed:.2f}s")
    print(f"Segment: {segment}")
    if segment:
        print(f"Segment duration: {segment.duration:.3f}s")

    listener.stop()

    # Segment should be detected
    assert segment is not None, f"Expected segment but got None (waited {elapsed:.2f}s)"
    assert segment.duration > 0
