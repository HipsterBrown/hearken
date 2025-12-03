"""Tests for Silero VAD implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
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
