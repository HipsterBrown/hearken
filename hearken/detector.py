"""Speech detection finite state machine."""

import logging
import collections
from typing import Optional, Callable

from .types import AudioChunk, SpeechSegment, DetectorState, DetectorConfig, VADResult
from .interfaces import VAD

logger = logging.getLogger('hearken.detector')


class SpeechDetector:
    """
    Finite state machine for segmenting continuous audio into speech segments.

    Implements 4-state FSM to robustly detect speech boundaries:
    - IDLE: Waiting for speech
    - SPEECH_STARTING: Speech detected, confirming it's not transient noise
    - SPEAKING: Confirmed speech, accumulating audio
    - TRAILING_SILENCE: Speech may have ended, waiting to confirm
    """

    def __init__(
        self,
        vad: VAD,
        config: Optional[DetectorConfig] = None,
        on_segment: Optional[Callable[[SpeechSegment], None]] = None,
    ):
        """
        Args:
            vad: Voice activity detector instance
            config: Detection configuration (uses defaults if None)
            on_segment: Callback when complete segment detected
        """
        self.vad = vad
        self.config = config or DetectorConfig()
        self.on_segment = on_segment

        # FSM state
        self.state = DetectorState.IDLE

        # Ring buffer for speech padding (pre-roll)
        padding_frames = int(
            self.config.speech_padding * 1000 / self.config.frame_duration_ms
        )
        self.padding_buffer: collections.deque[AudioChunk] = collections.deque(
            maxlen=max(1, padding_frames)
        )

        # Current segment accumulator
        self.segment_chunks: list[AudioChunk] = []
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None

    def process(self, chunk: AudioChunk) -> None:
        """
        Process an audio chunk through the FSM.

        Args:
            chunk: Audio chunk to process
        """
        # Run VAD
        try:
            vad_result = self.vad.process(chunk)
        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            return

        is_speech = vad_result.is_speech
        now = chunk.timestamp

        # FSM transitions
        if self.state == DetectorState.IDLE:
            self._handle_idle(chunk, is_speech, now)
        elif self.state == DetectorState.SPEECH_STARTING:
            self._handle_speech_starting(chunk, is_speech, now)
        elif self.state == DetectorState.SPEAKING:
            self._handle_speaking(chunk, is_speech, now)
        elif self.state == DetectorState.TRAILING_SILENCE:
            self._handle_trailing_silence(chunk, is_speech, now)

    def _handle_idle(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
        """IDLE state: waiting for speech."""
        self.padding_buffer.append(chunk)

        if is_speech:
            logger.debug("Speech detected, transitioning to SPEECH_STARTING")
            self.state = DetectorState.SPEECH_STARTING
            self.speech_start_time = now
            self.last_speech_time = now
            # Include padding buffer
            self.segment_chunks = list(self.padding_buffer)

    def _handle_speech_starting(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
        """SPEECH_STARTING: confirming speech isn't transient noise."""
        self.segment_chunks.append(chunk)

        if is_speech:
            self.last_speech_time = now

            # Check if we've exceeded minimum speech duration
            speech_duration = now - self.speech_start_time
            if speech_duration >= self.config.min_speech_duration:
                logger.debug(f"Speech confirmed after {speech_duration:.2f}s, transitioning to SPEAKING")
                self.state = DetectorState.SPEAKING
        else:
            # Check if silence has exceeded timeout (false start)
            silence_duration = now - self.last_speech_time
            if silence_duration >= self.config.silence_timeout:
                logger.debug(f"False start detected, returning to IDLE")
                self.state = DetectorState.IDLE
                self.segment_chunks = []
                self.padding_buffer.clear()
                self.vad.reset()

    def _handle_speaking(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
        """SPEAKING: confirmed speech, accumulating audio."""
        self.segment_chunks.append(chunk)

        if is_speech:
            self.last_speech_time = now

        # Check max duration (force split)
        speech_duration = now - self.speech_start_time
        if speech_duration >= self.config.max_speech_duration:
            logger.debug(f"Max duration ({self.config.max_speech_duration}s) reached, emitting segment")
            self._emit_segment(now)
            self.state = DetectorState.IDLE
        elif not is_speech:
            logger.debug("Silence detected, transitioning to TRAILING_SILENCE")
            self.state = DetectorState.TRAILING_SILENCE

    def _handle_trailing_silence(self, chunk: AudioChunk, is_speech: bool, now: float) -> None:
        """TRAILING_SILENCE: speech may have ended, waiting to confirm."""
        self.segment_chunks.append(chunk)

        if is_speech:
            logger.debug("Speech resumed, returning to SPEAKING")
            self.last_speech_time = now
            self.state = DetectorState.SPEAKING
        else:
            # Check if silence timeout exceeded
            silence_duration = now - self.last_speech_time
            if silence_duration >= self.config.silence_timeout:
                logger.debug(f"Silence confirmed after {silence_duration:.2f}s, emitting segment")
                self._emit_segment(now)
                self.state = DetectorState.IDLE

    def _emit_segment(self, end_time: float) -> None:
        """Emit a complete speech segment."""
        if not self.segment_chunks:
            return

        # Combine chunks into single audio blob
        audio_data = b''.join(c.data for c in self.segment_chunks)

        segment = SpeechSegment(
            audio_data=audio_data,
            sample_rate=self.segment_chunks[0].sample_rate,
            sample_width=self.segment_chunks[0].sample_width,
            start_time=self.speech_start_time,
            end_time=end_time,
        )

        logger.info(f"Speech segment detected: {segment.duration:.2f}s")

        # Reset for next segment
        self.segment_chunks = []
        self.padding_buffer.clear()
        self.vad.reset()

        # Invoke callback
        if self.on_segment:
            try:
                self.on_segment(segment)
            except Exception as e:
                logger.error(f"Segment callback failed: {e}")

    def reset(self) -> None:
        """Reset detector to initial state."""
        self.state = DetectorState.IDLE
        self.segment_chunks = []
        self.padding_buffer.clear()
        self.speech_start_time = None
        self.last_speech_time = None
        self.vad.reset()
