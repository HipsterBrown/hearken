"""
Adapters for speech_recognition library compatibility.

These adapters allow using speech_recognition's Microphone and Recognizer
with hearken's abstract interfaces.

Only available when SpeechRecognition is installed (hearken[sr]).
"""

try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

    # Define stubs so module can still be imported
    class sr:
        class Microphone:
            pass

        class Recognizer:
            pass

        class AudioData:
            pass


from ..interfaces import AudioSource, Transcriber
from ..types import SpeechSegment


class SpeechRecognitionSource(AudioSource):
    """
    Adapter for speech_recognition.Microphone.

    Example:
        import speech_recognition as sr
        from hearken.adapters.sr import SpeechRecognitionSource

        mic = sr.Microphone()
        source = SpeechRecognitionSource(mic)

        listener = Listener(source=source, ...)
    """

    def __init__(self, microphone: "sr.Microphone"):
        """
        Args:
            microphone: speech_recognition Microphone instance
        """
        if not SR_AVAILABLE:
            raise ImportError(
                "SpeechRecognition not installed. " "Install with: pip install hearken[sr]"
            )

        self.microphone = microphone
        self._context_manager = None

    def open(self) -> None:
        """Open the microphone."""
        self._context_manager = self.microphone.__enter__()

    def close(self) -> None:
        """Close the microphone."""
        if self._context_manager is not None:
            self.microphone.__exit__(None, None, None)
            self._context_manager = None

    def read(self, num_samples: int) -> bytes:
        """Read audio samples from microphone."""
        return self.microphone.stream.read(num_samples)

    @property
    def sample_rate(self) -> int:
        return self.microphone.SAMPLE_RATE

    @property
    def sample_width(self) -> int:
        return self.microphone.SAMPLE_WIDTH


class SRTranscriber(Transcriber):
    """
    Adapter for speech_recognition.Recognizer.

    Wraps any recognition method from speech_recognition (Google, Sphinx, etc.)

    Example:
        import speech_recognition as sr
        from hearken.adapters.sr import SRTranscriber

        recognizer = sr.Recognizer()

        # Adjust for ambient noise
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)

        # Use Google Speech API
        transcriber = SRTranscriber(
            recognizer,
            method='recognize_google',
            language='en-US'
        )

        listener = Listener(source=..., transcriber=transcriber)
    """

    def __init__(self, recognizer: "sr.Recognizer", method: str = "recognize_google", **kwargs):
        """
        Args:
            recognizer: speech_recognition Recognizer instance
            method: Recognition method name (e.g., 'recognize_google')
            **kwargs: Additional arguments passed to recognition method
        """
        if not SR_AVAILABLE:
            raise ImportError(
                "SpeechRecognition not installed. " "Install with: pip install hearken[sr]"
            )

        self.recognizer = recognizer
        self.method_name = method
        self.kwargs = kwargs

        # Get the recognition method
        if not hasattr(recognizer, method):
            raise ValueError(f"Recognizer has no method '{method}'")

        self._recognize_func = getattr(recognizer, method)

    def transcribe(self, segment: SpeechSegment) -> str:
        """
        Transcribe a speech segment using speech_recognition.

        Args:
            segment: Speech segment to transcribe

        Returns:
            Transcribed text

        Raises:
            Exception: If transcription fails (network error, etc.)
        """
        # Convert SpeechSegment to sr.AudioData
        audio_data = sr.AudioData(segment.audio_data, segment.sample_rate, segment.sample_width)

        # Call recognition method
        # Note: recognize_* methods raise sr.UnknownValueError if no speech detected
        # and sr.RequestError for API failures. Let these propagate.
        return self._recognize_func(audio_data, **self.kwargs)
