"""Example using Silero VAD for neural network-based voice activity detection.

Silero VAD provides superior accuracy compared to rule-based approaches,
especially in noisy environments. Requires 16kHz audio.
"""

import logging

import speech_recognition as sr
from hearken import Listener, SileroVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("hearken").setLevel(logging.DEBUG)

# Setup recognizer with default sample rate
recognizer = sr.Recognizer()
mics = sr.Microphone.list_microphone_names()
mic = sr.Microphone(device_index=mics.index("MacBook Pro Microphone"))

# Create listener with Silero VAD
# threshold=0.5 is default (lower=more sensitive, higher=more conservative)
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    vad=SileroVAD(threshold=0.5),
    on_transcript=lambda text, seg: print(f"You said: {text}"),
)

print("Listening with Silero VAD (neural network)...")
print("Speak naturally. Press Ctrl+C to stop.")

listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    print("\nStopping...")
    listener.stop()
