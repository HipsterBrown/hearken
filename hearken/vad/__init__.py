"""Voice activity detection implementations."""

from .energy import EnergyVAD

try:
    from .webrtc import WebRTCVAD
    __all__ = ['EnergyVAD', 'WebRTCVAD']
except ImportError:
    # webrtcvad-wheels not installed
    __all__ = ['EnergyVAD']
