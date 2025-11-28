"""Adapters for third-party libraries."""

# Speech recognition adapters (optional dependency)
try:
    from .sr import SpeechRecognitionSource, SRTranscriber
    __all__ = ['SpeechRecognitionSource', 'SRTranscriber']
except ImportError:
    __all__ = []
