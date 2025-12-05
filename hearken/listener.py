"""Main Listener class for hearken pipeline."""

import logging
import threading
import queue
import time
from typing import Optional, Callable

from .interfaces import AudioSource, Transcriber, VAD
from .types import AudioChunk, SpeechSegment, DetectorConfig
from .detector import SpeechDetector
from .vad.energy import EnergyVAD

logger = logging.getLogger("hearken")


class Listener:
    """
    Multi-threaded speech recognition pipeline.

    Decouples audio capture, voice activity detection, and transcription
    into independent threads to prevent audio drops during processing.
    """

    def __init__(
        self,
        source: AudioSource,
        transcriber: Optional[Transcriber] = None,
        vad: Optional[VAD] = None,
        detector_config: Optional[DetectorConfig] = None,
        on_speech: Optional[Callable[[SpeechSegment], None]] = None,
        on_transcript: Optional[Callable[[str, SpeechSegment], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        capture_queue_size: int = 100,
        segment_queue_size: int = 10,
    ):
        """
        Args:
            source: Audio input source
            transcriber: Transcription engine (required if on_transcript provided)
            vad: Voice activity detector (defaults to EnergyVAD)
            detector_config: Detection parameters (uses defaults if None)
            on_speech: Callback for raw speech segments (passive mode)
            on_transcript: Callback for transcribed segments (passive mode)
            on_error: Error callback (defaults to logging.error)
            capture_queue_size: Max chunks in capture queue
            segment_queue_size: Max segments in segment queue
        """
        self.source = source
        self.transcriber = transcriber
        self.vad = vad or EnergyVAD()
        self.detector_config = detector_config or DetectorConfig()
        self.on_speech = on_speech
        self.on_transcript = on_transcript
        self.on_error = on_error or self._default_error_handler

        # Validate configuration
        if on_transcript and not transcriber:
            raise ValueError("transcriber required when on_transcript is provided")

        # Queues
        self._capture_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue(
            maxsize=capture_queue_size
        )
        self._segment_queue: queue.Queue[Optional[SpeechSegment]] = queue.Queue(
            maxsize=segment_queue_size
        )

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start all pipeline threads."""
        if self._running:
            raise RuntimeError("Listener already running")

        logger.info("Starting listener")
        self._running = True
        self._stop_event.clear()

        # Open audio source
        try:
            self.source.open()
        except Exception as e:
            self._running = False
            logger.error(f"Failed to open audio source: {e}")
            raise

        # Start threads
        self._threads = [
            threading.Thread(target=self._capture_loop, name="hearken-capture", daemon=True),
            threading.Thread(target=self._detect_loop, name="hearken-detect", daemon=True),
        ]

        # Only start transcribe thread if needed for passive mode
        if self.on_transcript:
            self._threads.append(
                threading.Thread(
                    target=self._transcribe_loop, name="hearken-transcribe", daemon=True
                )
            )

        for t in self._threads:
            t.start()

        logger.info("Listener started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop all pipeline threads gracefully."""
        if not self._running:
            return

        logger.info("Stopping listener")
        self._running = False
        self._stop_event.set()

        # Send poison pills
        try:
            self._capture_queue.put_nowait(None)
        except queue.Full:
            pass

        try:
            self._segment_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for threads
        for t in self._threads:
            t.join(timeout=timeout)

        self._threads.clear()

        # Close audio source
        try:
            self.source.close()
        except Exception as e:
            logger.error(f"Error closing audio source: {e}")

        logger.info("Listener stopped")

    def wait(self) -> None:
        """Block until stop() is called or threads exit."""
        while self._running and any(t.is_alive() for t in self._threads):
            time.sleep(0.1)

    def wait_for_speech(self, timeout: Optional[float] = None) -> Optional[SpeechSegment]:
        """
        Block until a speech segment is detected (active mode).

        Args:
            timeout: Optional timeout in seconds (None = wait indefinitely)

        Returns:
            SpeechSegment if detected, None if timeout

        Raises:
            RuntimeError: If listener not running
        """
        if not self._running:
            raise RuntimeError("Listener not running")

        try:
            segment = self._segment_queue.get(timeout=timeout)
            return segment if segment is not None else None
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        """Capture thread: reads audio chunks at fixed intervals."""
        frame_duration_ms = (
            self.vad.required_frame_duration_ms or self.detector_config.frame_duration_ms
        )
        chunk_samples = int(self.source.sample_rate * frame_duration_ms / 1000)

        logger.debug(
            f"Capture thread started (frame_duration={frame_duration_ms}ms, samples={chunk_samples})"
        )

        chunks_captured = 0
        chunks_dropped = 0

        while self._running:
            try:
                # Read audio - releases GIL during device read
                data = self.source.read(chunk_samples)

                chunk = AudioChunk(
                    data=data,
                    timestamp=time.monotonic(),
                    sample_rate=self.source.sample_rate,
                    sample_width=self.source.sample_width,
                )

                # Non-blocking put
                try:
                    self._capture_queue.put_nowait(chunk)
                    chunks_captured += 1
                except queue.Full:
                    chunks_dropped += 1
                    if chunks_dropped % 100 == 0:
                        drop_rate = chunks_dropped / (chunks_captured + chunks_dropped) * 100
                        logger.warning(
                            f"Capture queue full, dropped {chunks_dropped} chunks ({drop_rate:.1f}%)"
                        )

            except Exception as e:
                if self._running:
                    logger.error(f"Capture error: {e}")
                    self.on_error(e)
                break

        logger.debug(
            f"Capture thread stopped (captured={chunks_captured}, dropped={chunks_dropped})"
        )

    def _detect_loop(self) -> None:
        """Detection thread: runs VAD and FSM to segment audio."""
        detector = SpeechDetector(
            vad=self.vad,
            config=self.detector_config,
            on_segment=self._handle_segment,
        )

        logger.debug("Detection thread started")

        while self._running:
            try:
                chunk = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # Poison pill
                break

            detector.process(chunk)

        logger.debug("Detection thread stopped")

    def _handle_segment(self, segment: SpeechSegment) -> None:
        """Handle detected speech segment."""
        # Call on_speech callback asynchronously (don't block detect thread)
        if self.on_speech:
            t = threading.Thread(
                target=self._safe_callback,
                args=(self.on_speech, segment),
                daemon=False,  # Not daemon so it completes
                name="hearken-callback",
            )
            t.start()

        # Queue for active mode or transcription
        try:
            self._segment_queue.put_nowait(segment)
        except queue.Full:
            logger.warning(f"Segment queue full, dropping {segment.duration:.1f}s segment")

    def _safe_callback(self, callback: Callable, *args) -> None:
        """Execute callback with error handling."""
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Callback failed: {e}", exc_info=True)
            self.on_error(e)

    def _transcribe_loop(self) -> None:
        """Transcription thread: transcribes segments and invokes callback."""
        logger.debug("Transcription thread started")

        while self._running:
            try:
                segment = self._segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if segment is None:  # Poison pill
                break

            try:
                # Transcribe - may release GIL during network I/O
                text = self.transcriber.transcribe(segment)

                # Fire callback asynchronously (don't block transcription)
                if self.on_transcript:
                    threading.Thread(
                        target=self._safe_callback,
                        args=(self.on_transcript, text, segment),
                        daemon=False,
                        name="hearken-callback",
                    ).start()

            except Exception as e:
                logger.error(f"Transcription failed: {e}")
                self.on_error(e)

        logger.debug("Transcription thread stopped")

    def _default_error_handler(self, error: Exception) -> None:
        """Default error handler - just logs."""
        logger.error(f"Pipeline error: {error}", exc_info=True)
