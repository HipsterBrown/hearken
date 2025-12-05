"""Abstract interfaces for hearken components."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import AudioChunk, SpeechSegment, VADResult


class AudioSource(ABC):
    """Abstract interface for audio input devices."""

    @abstractmethod
    def open(self) -> None:
        """Open the audio source for reading."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the audio source and release resources."""
        ...

    @abstractmethod
    def read(self, num_samples: int) -> bytes:
        """Read audio samples from the source."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate in Hz (e.g., 16000)."""
        ...

    @property
    @abstractmethod
    def sample_width(self) -> int:
        """Bytes per sample (e.g., 2 for 16-bit)."""
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class Transcriber(ABC):
    """Abstract interface for speech-to-text transcription."""

    @abstractmethod
    def transcribe(self, segment: "SpeechSegment") -> str:
        """Transcribe audio to text. May raise exceptions for API errors."""
        ...


class VAD(ABC):
    """Voice Activity Detection interface."""

    @abstractmethod
    def process(self, chunk: "AudioChunk") -> "VADResult":
        """Process audio chunk and return speech detection result."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state between utterances."""
        ...

    @property
    def required_sample_rate(self) -> int | None:
        """Required sample rate, or None if flexible."""
        return None

    @property
    def required_frame_duration_ms(self) -> int | float | None:
        """Required frame duration in ms, or None if flexible."""
        return None
