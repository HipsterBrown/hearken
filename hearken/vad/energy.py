"""Energy-based voice activity detection."""

import numpy as np
from typing import Optional

from ..interfaces import VAD
from ..types import AudioChunk, VADResult


class EnergyVAD(VAD):
    """
    Simple energy-based voice activity detection.

    Uses RMS (root mean square) energy threshold to detect speech.
    Optionally adapts threshold based on ambient noise during calibration.
    """

    def __init__(
        self,
        threshold: float = 300.0,
        dynamic: bool = True,
        calibration_samples: int = 50,  # ~1.5s at 30ms frames
    ):
        """
        Args:
            threshold: RMS energy threshold for speech detection
            dynamic: If True, adapt threshold based on ambient noise
            calibration_samples: Number of initial samples for calibration
        """
        self.base_threshold = threshold
        self.dynamic = dynamic
        self.calibration_samples = calibration_samples

        self._ambient_energy: Optional[float] = None
        self._samples_seen = 0

    def process(self, chunk: AudioChunk) -> VADResult:
        # Convert bytes to int16 samples
        samples = np.frombuffer(chunk.data, dtype=np.int16)

        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        # Dynamic threshold adjustment during calibration
        if self.dynamic and self._samples_seen < self.calibration_samples:
            if self._ambient_energy is None:
                self._ambient_energy = energy
            else:
                # Exponential moving average
                self._ambient_energy = 0.9 * self._ambient_energy + 0.1 * energy
            self._samples_seen += 1

            # Use 1.5x ambient energy as threshold during calibration
            effective_threshold = max(self.base_threshold, self._ambient_energy * 1.5)
        else:
            effective_threshold = self.base_threshold

        is_speech = bool(energy > effective_threshold)

        # Confidence: how far above/below threshold
        confidence = min(1.0, float(energy / effective_threshold)) if is_speech else 0.0

        return VADResult(is_speech=is_speech, confidence=confidence)

    def reset(self) -> None:
        """Reset between utterances. Don't reset ambient calibration."""
        pass

    @property
    def required_sample_rate(self) -> Optional[int]:
        return None  # Works with any sample rate

    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        return None  # Works with any frame size
