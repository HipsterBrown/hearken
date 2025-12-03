"""Tests for WebRTC VAD implementation."""
import pytest
from hearken.vad.webrtc import WebRTCVAD


def test_webrtc_vad_creation_default():
    """Test WebRTCVAD initialization with default aggressiveness."""
    vad = WebRTCVAD()

    # Should create successfully with default aggressiveness=1
    assert vad is not None


def test_webrtc_vad_creation_with_aggressiveness():
    """Test WebRTCVAD initialization with custom aggressiveness."""
    vad = WebRTCVAD(aggressiveness=2)

    assert vad is not None


def test_webrtc_vad_invalid_aggressiveness_negative():
    """Test WebRTCVAD rejects negative aggressiveness."""
    with pytest.raises(ValueError, match="Aggressiveness must be 0-3"):
        WebRTCVAD(aggressiveness=-1)


def test_webrtc_vad_invalid_aggressiveness_too_high():
    """Test WebRTCVAD rejects aggressiveness > 3."""
    with pytest.raises(ValueError, match="Aggressiveness must be 0-3"):
        WebRTCVAD(aggressiveness=4)


def test_webrtc_vad_all_valid_aggressiveness_modes():
    """Test all valid aggressiveness modes (0-3)."""
    for mode in [0, 1, 2, 3]:
        vad = WebRTCVAD(aggressiveness=mode)
        assert vad is not None
