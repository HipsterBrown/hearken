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
