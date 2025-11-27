"""Core data types for hearken pipeline."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    data: bytes
    timestamp: float          # time.monotonic() when captured
    sample_rate: int
    sample_width: int         # bytes per sample (2 for 16-bit)


@dataclass
class SpeechSegment:
    """A complete speech segment ready for transcription."""
    audio_data: bytes         # Raw PCM audio
    sample_rate: int
    sample_width: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class VADResult:
    """Result from voice activity detection."""
    is_speech: bool
    confidence: float = 1.0   # 0.0 to 1.0


class DetectorState(Enum):
    """FSM states for utterance segmentation."""
    IDLE = auto()             # Waiting for speech
    SPEECH_STARTING = auto()  # Speech detected, confirming it's not noise
    SPEAKING = auto()         # Confirmed speech, accumulating audio
    TRAILING_SILENCE = auto() # Speech may have ended, waiting to confirm


@dataclass
class DetectorConfig:
    """Configuration for utterance detection FSM."""

    # Minimum speech duration to consider valid (filters transients)
    min_speech_duration: float = 0.25  # seconds

    # Maximum speech duration before forced segmentation
    max_speech_duration: float = 30.0  # seconds

    # Silence duration to end an utterance
    silence_timeout: float = 0.8  # seconds

    # Audio to prepend before detected speech start
    speech_padding: float = 0.3  # seconds

    # Frame duration for audio chunks
    frame_duration_ms: int = 30  # milliseconds
