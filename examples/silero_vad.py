"""Example using Silero VAD for neural network-based voice activity detection.

Silero VAD provides superior accuracy compared to rule-based approaches,
especially in noisy environments. Requires 16kHz audio.
"""

import speech_recognition as sr
from hearken import Listener, SileroVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

# Setup recognizer with 16kHz sample rate
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=16000)

# Create listener with Silero VAD
# threshold=0.5 is default (lower=more sensitive, higher=more conservative)
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    vad=SileroVAD(threshold=0.5),
    on_transcript=lambda text, seg: print(f"You said: {text}")
)

print("Listening with Silero VAD (neural network)...")
print("Speak naturally. Press Ctrl+C to stop.")

listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    print("\nStopping...")
    listener.stop()
