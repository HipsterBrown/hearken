"""Tests for Silero VAD implementation."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from hearken.vad.silero import SileroVAD


def test_silero_vad_creation_default():
    """Test SileroVAD creation with default parameters."""
    with patch('hearken.vad.silero.ort.InferenceSession') as mock_session, \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad = SileroVAD()
        assert vad._threshold == 0.5
        assert vad._validated is False
        assert vad._sample_rate is None


def test_silero_vad_creation_with_threshold():
    """Test SileroVAD creation with custom threshold."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
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
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad_min = SileroVAD(threshold=0.0)
        assert vad_min._threshold == 0.0

        vad_max = SileroVAD(threshold=1.0)
        assert vad_max._threshold == 1.0


def test_silero_vad_model_path_parameter():
    """Test model path from constructor parameter (highest priority)."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'):
        vad = SileroVAD(model_path="/custom/path/model.onnx")
        assert vad._model_path == "/custom/path/model.onnx"


def test_silero_vad_model_path_env_var():
    """Test model path from environment variable."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}):
        vad = SileroVAD()
        assert vad._model_path == "/env/path/model.onnx"


def test_silero_vad_model_path_default():
    """Test default model path."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {}, clear=True):
        vad = SileroVAD()
        expected = str(Path.home() / ".cache" / "hearken" / "silero_vad_v5.onnx")
        assert vad._model_path == expected


def test_silero_vad_model_path_parameter_overrides_env():
    """Test parameter takes precedence over environment variable."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.SileroVAD._ensure_model_downloaded'), \
         patch.dict(os.environ, {"HEARKEN_SILERO_MODEL_PATH": "/env/path/model.onnx"}):
        vad = SileroVAD(model_path="/param/path/model.onnx")
        assert vad._model_path == "/param/path/model.onnx"


def test_silero_vad_downloads_model_if_missing():
    """Test model is downloaded if not present."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.Path.exists', return_value=False), \
         patch('hearken.vad.silero.Path.mkdir'), \
         patch('hearken.vad.silero.urllib.request.urlopen') as mock_urlopen, \
         patch('builtins.open', mock_open()) as mock_file:

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.read.return_value = b'fake model data'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        vad = SileroVAD()

        # Verify download was attempted
        mock_urlopen.assert_called_once()
        assert SileroVAD.MODEL_URL in str(mock_urlopen.call_args)


def test_silero_vad_skips_download_if_exists():
    """Test model download is skipped if file exists."""
    with patch('hearken.vad.silero.ort.InferenceSession'), \
         patch('hearken.vad.silero.Path.exists', return_value=True), \
         patch('hearken.vad.silero.urllib.request.urlopen') as mock_urlopen:

        vad = SileroVAD()

        # Verify download was NOT attempted
        mock_urlopen.assert_not_called()


def test_silero_vad_download_failure_raises_error():
    """Test clear error when model download fails."""
    with patch('hearken.vad.silero.Path.exists', return_value=False), \
         patch('hearken.vad.silero.Path.mkdir'), \
         patch('hearken.vad.silero.urllib.request.urlopen', side_effect=Exception("Network error")):

        with pytest.raises(RuntimeError) as exc_info:
            SileroVAD()

        error_msg = str(exc_info.value)
        assert "Failed to download Silero VAD model" in error_msg
        assert "Network error" in error_msg
        assert SileroVAD.MODEL_URL in error_msg
        assert "HEARKEN_SILERO_MODEL_PATH" in error_msg
