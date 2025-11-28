"""
Hearken - Robust speech recognition pipeline for Python.

Decouples audio capture, voice activity detection, and transcription
into independent threads to prevent audio drops during processing.
"""

__version__ = "0.1.0"

# Core components
from .listener import Listener
from .types import (
    AudioChunk,
    SpeechSegment,
    VADResult,
    DetectorConfig,
    DetectorState,
)

# Interfaces
from .interfaces import AudioSource, Transcriber, VAD

# VAD implementations
from .vad.energy import EnergyVAD

__all__ = [
    # Main class
    "Listener",
    # Data types
    "AudioChunk",
    "SpeechSegment",
    "VADResult",
    "DetectorConfig",
    "DetectorState",
    # Interfaces
    "AudioSource",
    "Transcriber",
    "VAD",
    # VAD implementations
    "EnergyVAD",
]
