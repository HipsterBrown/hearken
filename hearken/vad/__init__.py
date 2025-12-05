"""Voice Activity Detection implementations."""

from .energy import EnergyVAD

try:
    from .webrtc import WebRTCVAD
    _webrtc_available = True
except ImportError:
    _webrtc_available = False

try:
    from .silero import SileroVAD
    _silero_available = True
except ImportError:
    _silero_available = False

__all__ = ['EnergyVAD']
if _webrtc_available:
    __all__.append('WebRTCVAD')
if _silero_available:
    __all__.append('SileroVAD')
