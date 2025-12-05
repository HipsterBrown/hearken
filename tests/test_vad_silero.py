"""Tests for Silero VAD implementation."""

import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from hearken.vad.silero import SileroVAD


def test_silero_vad_creation_default():
    """Test SileroVAD creation with default parameters."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):
        vad = SileroVAD()
        assert vad._threshold == 0.5
        assert vad._validated is False
        assert vad._sample_rate is None


def test_silero_vad_creation_with_threshold():
    """Test SileroVAD creation with custom threshold."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):
        vad = SileroVAD(threshold=0.7)
        assert vad._threshold == 0.7


def test_silero_vad_invalid_threshold_negative():
    """Test SileroVAD rejects negative threshold."""
    with pytest.raises(ValueError) as exc_info:
        SileroVAD(threshold=-0.1)

    assert "Threshold must be between 0.0 and 1.0" in str(exc_info.value)
    assert "got -0.1" in str(exc_info.value)


def test_silero_vad_invalid_threshold_too_high():
    """Test SileroVAD rejects threshold > 1.0."""
    with pytest.raises(ValueError) as exc_info:
        SileroVAD(threshold=1.5)

    assert "Threshold must be between 0.0 and 1.0" in str(exc_info.value)
    assert "got 1.5" in str(exc_info.value)


def test_silero_vad_boundary_thresholds():
    """Test SileroVAD accepts boundary values 0.0 and 1.0."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):
        vad_min = SileroVAD(threshold=0.0)
        assert vad_min._threshold == 0.0

        vad_max = SileroVAD(threshold=1.0)
        assert vad_max._threshold == 1.0


def test_silero_vad_model_path_parameter():
    """Test model path from constructor parameter (highest priority)."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):
        vad = SileroVAD(model_path="/custom/path/model.onnx")
        assert vad._model_path == "/custom/path/model.onnx"


def test_silero_vad_model_path_env_var():
    """Test model path from environment variable."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
        patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}),
    ):
        vad = SileroVAD()
        assert vad._model_path == "/env/path/model.onnx"


def test_silero_vad_model_path_default():
    """Test default model path."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
        patch.dict(os.environ, {}, clear=True),
    ):
        vad = SileroVAD()
        expected = str(Path.home() / ".cache" / "hearken" / "silero_vad_v5.onnx")
        assert vad._model_path == expected


def test_silero_vad_model_path_parameter_overrides_env():
    """Test parameter takes precedence over environment variable."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
        patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}),
    ):
        vad = SileroVAD(model_path="/param/path/model.onnx")
        assert vad._model_path == "/param/path/model.onnx"


def test_silero_vad_downloads_model_if_missing():
    """Test model is downloaded if not present."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.Path.exists", return_value=False),
        patch("hearken.vad.silero.Path.mkdir"),
        patch("hearken.vad.silero.urllib.request.urlopen") as mock_urlopen,
        patch("builtins.open", mock_open()) as mock_file,
    ):

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.read.return_value = b"fake model data"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        vad = SileroVAD()

        # Verify download was attempted
        mock_urlopen.assert_called_once()
        assert SileroVAD.MODEL_URL in str(mock_urlopen.call_args)


def test_silero_vad_skips_download_if_exists():
    """Test model download is skipped if file exists."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.Path.exists", return_value=True),
        patch("hearken.vad.silero.urllib.request.urlopen") as mock_urlopen,
    ):

        vad = SileroVAD()

        # Verify download was NOT attempted
        mock_urlopen.assert_not_called()


def test_silero_vad_download_failure_raises_error():
    """Test clear error when model download fails."""
    with (
        patch("hearken.vad.silero.Path.exists", return_value=False),
        patch("hearken.vad.silero.Path.mkdir"),
        patch("hearken.vad.silero.urllib.request.urlopen", side_effect=Exception("Network error")),
    ):

        with pytest.raises(RuntimeError) as exc_info:
            SileroVAD()

        error_msg = str(exc_info.value)
        assert "Failed to download Silero VAD model" in error_msg
        assert "Network error" in error_msg
        assert SileroVAD.MODEL_URL in error_msg
        assert "HEARKEN_SILERO_MODEL_PATH" in error_msg


# Sample Rate Validation Tests


def test_silero_vad_accepts_16khz():
    """Test SileroVAD accepts 16kHz audio."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock ONNX session inference
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.7), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.5)

        # Create 16kHz audio chunk
        chunk = AudioChunk(
            data=b"\x00" * 960,  # 30ms at 16kHz = 480 samples * 2 bytes
            sample_rate=16000,
            sample_width=2,
            timestamp=0.0,
        )

        # Should not raise
        result = vad.process(chunk)
        assert vad._validated is True
        assert vad._sample_rate == 16000


def test_silero_vad_rejects_non_16khz():
    """Test SileroVAD rejects non-16kHz (or multiples) audio with clear error."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession"),
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        vad = SileroVAD(threshold=0.5)

        # Test various invalid sample rates
        invalid_rates = [8000, 44100]

        for rate in invalid_rates:
            chunk = AudioChunk(data=b"\x00" * 480, sample_rate=rate, sample_width=2, timestamp=0.0)

            with pytest.raises(ValueError) as exc_info:
                vad.process(chunk)

            error_msg = str(exc_info.value)
            assert "Invalid sample rate for Silero VAD" in error_msg
            assert str(rate) in error_msg
            assert "16000 Hz" in error_msg
            assert "Please configure your AudioSource" in error_msg


def test_silero_vad_validates_only_on_first_call():
    """Test sample rate validation only happens on first process() call."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock ONNX session inference
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.7), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.5)

        # Create 16kHz audio chunk
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)

        # First call should validate
        assert vad._validated is False
        vad.process(chunk)
        assert vad._validated is True

        # Second call should not re-validate
        vad.process(chunk)
        assert vad._validated is True


# Threshold Application Tests


def test_silero_vad_threshold_below():
    """Test confidence below threshold returns is_speech=False."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock returns confidence=0.3
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.3), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)

        result = vad.process(chunk)
        assert result.is_speech is False
        assert result.confidence == 0.3


def test_silero_vad_threshold_above():
    """Test confidence above threshold returns is_speech=True."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock returns confidence=0.7
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.7), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)

        result = vad.process(chunk)
        assert result.is_speech is True
        assert result.confidence == 0.7


def test_silero_vad_threshold_boundary():
    """Test confidence exactly at threshold returns is_speech=True."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock returns confidence=0.5
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.5), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.5)
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)

        result = vad.process(chunk)
        assert result.is_speech is True  # >= threshold
        assert result.confidence == 0.5


def test_silero_vad_custom_threshold():
    """Test custom threshold value works correctly."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        # Mock returns confidence=0.6
        mock_instance = Mock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_instance.run.return_value = (np.array(0.6), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_instance

        vad = SileroVAD(threshold=0.7)
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)

        result = vad.process(chunk)
        assert result.is_speech is False  # 0.6 < 0.7
        assert result.confidence == 0.6


# Reset Functionality Tests


def test_silero_vad_reset():
    """Test reset reinitializes ONNX session."""
    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session_class,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        mock_session_instance = MagicMock()
        mock_session_class.return_value = mock_session_instance

        vad = SileroVAD()

        # Verify session created once on init
        assert mock_session_class.call_count == 1

        # Reset
        vad.reset()

        # Verify session created again
        assert mock_session_class.call_count == 2


def test_silero_vad_reset_clears_validation_state():
    """Test reset clears validation state."""
    from hearken.types import AudioChunk

    with (
        patch("hearken.vad.silero.ort.InferenceSession") as mock_session,
        patch("hearken.vad.silero.SileroVAD._ensure_model_downloaded"),
    ):

        mock_output = MagicMock()
        # Return (confidence_tensor, state) - confidence needs .item() method
        mock_output.run.return_value = (np.array(0.7), np.zeros((2, 1, 128), dtype=np.float32))
        mock_session.return_value = mock_output

        vad = SileroVAD()

        # Process chunk to trigger validation
        chunk = AudioChunk(data=b"\x00" * 960, sample_rate=16000, sample_width=2, timestamp=0.0)
        vad.process(chunk)

        assert vad._validated is True
        assert vad._sample_rate == 16000

        # Reset
        vad.reset()

        # Verify validation state cleared
        assert vad._validated is False
        assert vad._sample_rate is None
