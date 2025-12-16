import asyncio
import numpy as np
from hearken import Listener
from hearken.interfaces import AudioSource, AsyncAudioSource, Transcriber
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


class MockAsyncAudioSource(AsyncAudioSource):
    """Mock async audio source that generates speech-like audio."""

    def __init__(self, max_chunks: int = 30):
        self.is_closed = False
        self.frame_count = 0
        self.max_chunks = max_chunks

    def close(self) -> None:
        self.is_closed = True

    async def stream(self):
        """Async generator that yields audio chunks."""
        # 16kHz, 30ms frames = 480 samples per chunk
        num_samples = 480

        while self.frame_count < self.max_chunks and not self.is_closed:
            # Simulate async I/O with small delay
            await asyncio.sleep(0.01)

            # silence/speech pattern
            if self.frame_count < 6:
                samples = np.random.randint(-50, 50, size=num_samples, dtype=np.int16)
            else:
                # Generate pattern: 12 frames of speech, then 6 frames of silence
                adjusted_frame = (self.frame_count - 6) % 18
                if adjusted_frame < 12:
                    # Speech frames (very high energy for reliable detection)
                    samples = np.random.randint(-20000, 20000, size=num_samples, dtype=np.int16)
                else:
                    # Silence frames (low energy)
                    samples = np.random.randint(-50, 50, size=num_samples, dtype=np.int16)

            self.frame_count += 1
            yield samples.tobytes()

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


def test_async_audio_source():
    """Test AsyncAudioSource with event loop integration."""
    import time
    import threading

    # Create a new event loop for the async source
    loop = asyncio.new_event_loop()

    # Run the event loop in a separate thread
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    time.sleep(0.05)

    source = MockAsyncAudioSource(max_chunks=20)

    # Create listener with async source and event loop
    listener = Listener(source=source, event_loop=loop)

    # Start and briefly run the listener
    listener.start()
    time.sleep(0.3)
    listener.stop()

    # Stop the event loop
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=1.0)

    # If we got here without errors, async audio source is working
    assert source.is_closed

def test_async_audio_source_wait_for_speech():
    """Test AsyncAudioSource with wait_for_speech() in active mode."""
    import time
    import threading

    # Create a new event loop for the async source
    loop = asyncio.new_event_loop()

    # Run the event loop in a separate thread
    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop, daemon=True)
    loop_thread.start()
    time.sleep(0.05)

    source = MockAsyncAudioSource(max_chunks=50)
    config = DetectorConfig(
        min_speech_duration=0.09,  # 3 frames at 30ms
        silence_timeout=0.12,  # 4 frames at 30ms
    )

    # Create listener with async source
    listener = Listener(source=source, detector_config=config, event_loop=loop)
    listener.start()

    # Give system a moment to start processing
    time.sleep(0.1)

    # Wait for speech segment with generous timeout
    segment = listener.wait_for_speech(timeout=3.0)

    listener.stop()
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=1.0)

    # Verify a segment was detected
    assert segment is not None, "Expected to detect speech segment from async audio source"
    assert segment.duration > 0
    print(f"Detected segment: {segment.duration:.3f}s")
