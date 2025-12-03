"""WebRTC-based voice activity detection."""
import logging
from typing import Optional

try:
    import webrtcvad
except ImportError:
    raise ImportError(
        "webrtcvad-wheels is required for WebRTCVAD. "
        "Install with: pip install hearken[webrtc]"
    )

from ..interfaces import VAD
from ..types import AudioChunk, VADResult

logger = logging.getLogger("hearken")


class WebRTCVAD(VAD):
    """
    WebRTC-based voice activity detection.

    Uses Google's WebRTC VAD for robust speech detection.
    More accurate than EnergyVAD in noisy environments.

    Constraints:
    - Sample rate must be 8000, 16000, 32000, or 48000 Hz
    - Frame duration must be 10, 20, or 30 ms
    """

    SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
    SUPPORTED_FRAME_DURATIONS_MS = [10, 20, 30]

    def __init__(self, aggressiveness: int = 1):
        """
        Initialize WebRTC VAD.

        Args:
            aggressiveness: Aggressiveness mode (0-3)
                0: Least aggressive (more speech detected)
                1: Quality mode (default)
                2: Low bitrate mode
                3: Very aggressive (less speech detected)

        Raises:
            ValueError: If aggressiveness is not in range 0-3
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError(
                f"Aggressiveness must be 0-3, got {aggressiveness}"
            )

        self._aggressiveness = aggressiveness
        self._vad = webrtcvad.Vad(aggressiveness)
        self._validated = False
        self._sample_rate: Optional[int] = None

    def process(self, chunk: AudioChunk) -> VADResult:
        """Process audio chunk and return speech detection result."""
        # Validate sample rate and frame duration on first call
        if not self._validated:
            if chunk.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
                raise ValueError(
                    f"WebRTC VAD requires sample rate of {self.SUPPORTED_SAMPLE_RATES} Hz. "
                    f"Got {chunk.sample_rate} Hz. "
                    f"Configure your AudioSource with a supported sample rate."
                )

            # Calculate frame duration from chunk
            num_samples = len(chunk.data) // chunk.sample_width
            duration_ms = int((num_samples / chunk.sample_rate) * 1000)

            if duration_ms not in self.SUPPORTED_FRAME_DURATIONS_MS:
                raise ValueError(
                    f"WebRTC VAD requires frame duration of {self.SUPPORTED_FRAME_DURATIONS_MS} ms. "
                    f"Got {duration_ms} ms. "
                    f"Configure your AudioSource with a supported frame duration."
                )

            self._sample_rate = chunk.sample_rate
            self._validated = True

        # Call WebRTC VAD
        is_speech = self._vad.is_speech(chunk.data, self._sample_rate)

        # Map boolean to confidence
        confidence = 1.0 if is_speech else 0.0

        return VADResult(is_speech=is_speech, confidence=confidence)

    def reset(self) -> None:
        """Reset internal state between utterances."""
        # Recreate VAD instance for clean state
        self._vad = webrtcvad.Vad(self._aggressiveness)
        # Clear validation state to allow revalidation
        self._validated = False
        self._sample_rate = None

    @property
    def required_sample_rate(self) -> Optional[int]:
        """Required sample rate, or None if flexible."""
        return None  # Validated at runtime

    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        """Required frame duration in ms, or None if flexible."""
        return None  # Validated at runtime
