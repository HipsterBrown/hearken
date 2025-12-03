"""Tests for WebRTC VAD implementation."""
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
