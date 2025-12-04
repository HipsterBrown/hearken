"""Silero VAD implementation using ONNX Runtime."""

import os
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for SileroVAD. "
        "Install with: pip install hearken[silero]"
    )

from hearken.interfaces import VAD
from hearken.types import AudioChunk, VADResult


class SileroVAD(VAD):
    """Neural network-based VAD using Silero VAD v5 with ONNX Runtime.

    Requires 16kHz audio. Provides superior accuracy compared to rule-based
    approaches, especially in noisy environments.

    Args:
        threshold: Confidence threshold for speech detection (0.0-1.0).
                   Default 0.5. Lower = more sensitive, higher = more conservative.
        model_path: Path to ONNX model file. If None, downloads from GitHub
                    and caches in ~/.cache/hearken/silero_vad_v5.onnx.
                    Can also be set via HEARKEN_SILERO_MODEL_PATH env var.

    Raises:
        ValueError: If threshold not in [0.0, 1.0]
        RuntimeError: If model download fails
    """

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hearken"
    DEFAULT_MODEL_NAME = "silero_vad_v5.onnx"

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[str] = None
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {threshold}\n"
                "Lower values (e.g., 0.3) are more sensitive.\n"
                "Higher values (e.g., 0.7) are more conservative."
            )

        self._threshold = threshold
        self._model_path = self._resolve_model_path(model_path)
        self._ensure_model_downloaded()
        self._session = ort.InferenceSession(self._model_path)
        self._validated = False
        self._sample_rate: Optional[int] = None

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve model path from parameter, env var, or default."""
        if model_path:
            return model_path

        env_path = os.environ.get("HEARKEN_SILERO_MODEL_PATH")
        if env_path:
            return env_path

        return str(self.DEFAULT_CACHE_DIR / self.DEFAULT_MODEL_NAME)

    def _ensure_model_downloaded(self) -> None:
        """Download model if it doesn't exist at resolved path."""
        model_file = Path(self._model_path)

        # Skip download if file exists
        if model_file.exists():
            return

        # Create cache directory if needed
        model_file.parent.mkdir(parents=True, exist_ok=True)

        # Download model
        try:
            with urllib.request.urlopen(self.MODEL_URL) as response:
                model_data = response.read()

            # Write to file
            with open(model_file, 'wb') as f:
                f.write(model_data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to download Silero VAD model from GitHub: {e}\n"
                f"Please check your internet connection or download manually:\n"
                f"  URL: {self.MODEL_URL}\n"
                f"  Save to: {self._model_path}\n"
                f"Or set environment variable: HEARKEN_SILERO_MODEL_PATH=/path/to/model.onnx"
            )

    def process(self, chunk: AudioChunk) -> VADResult:
        """Process audio chunk and return VAD result.

        Args:
            chunk: Audio chunk to process. Must be 16kHz.

        Returns:
            VADResult with is_speech boolean and confidence score.

        Raises:
            ValueError: If sample rate is not 16kHz.
        """
        # Validate 16kHz on first call
        if not self._validated:
            if chunk.sample_rate != 16000:
                raise ValueError(
                    f"Invalid sample rate for Silero VAD: {chunk.sample_rate} Hz\n"
                    f"Silero VAD requires 16000 Hz audio.\n"
                    f"Please configure your AudioSource to use 16kHz sample rate."
                )

            self._sample_rate = chunk.sample_rate
            self._validated = True

        # Convert bytes to numpy float32 array
        audio_int16 = np.frombuffer(chunk.data, dtype=np.int16)

        # Normalize to [-1.0, 1.0] range
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Run ONNX inference
        # Note: For now, we'll use simple inference without state management
        # Full state management (h, c tensors) will be implemented in later task
        ort_inputs = {'input': audio_float32[np.newaxis, :]}
        ort_outputs = self._session.run(None, ort_inputs)

        # Extract confidence from output
        # ort_outputs is a list of arrays, first one contains the speech probability
        confidence_array = ort_outputs[0]
        # Handle nested structure - confidence_array is [[value]] or [[[value]]]
        while hasattr(confidence_array, '__getitem__') and len(confidence_array) > 0:
            if isinstance(confidence_array[0], (int, float, np.number)):
                confidence = float(confidence_array[0])
                break
            confidence_array = confidence_array[0]
        else:
            confidence = float(confidence_array)

        # Apply threshold
        is_speech = confidence >= self._threshold

        return VADResult(is_speech=is_speech, confidence=confidence)

    def reset(self) -> None:
        """Reset VAD state by reinitializing ONNX session.

        Creates a new inference session to ensure completely clean state.
        Also clears validation state to allow sample rate revalidation.
        """
        # Reinitialize ONNX session for clean state
        self._session = ort.InferenceSession(self._model_path)

        # Clear validation state
        self._validated = False
        self._sample_rate = None
