"""
Example: Using WebRTC VAD for improved accuracy.

Demonstrates using WebRTC VAD instead of EnergyVAD for more robust
speech detection in noisy environments.

Requirements:
    pip install hearken[webrtc,sr]
"""

import logging
import speech_recognition as sr
from hearken import Listener, DetectorConfig
from hearken.vad import WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("hearken").setLevel(logging.DEBUG)


def main():
    # Setup speech_recognition components
    recognizer = sr.Recognizer()
    # WebRTC VAD requires supported sample rate (8/16/32/48 kHz)
    mics = sr.Microphone.list_microphone_names()
    mic = sr.Microphone(device_index=mics.index("MacBook Pro Microphone"))

    # Create hearken components with WebRTC VAD
    audio_source = SpeechRecognitionSource(mic)
    vad = WebRTCVAD(aggressiveness=1)  # Quality mode
    transcriber = SRTranscriber(recognizer, method="recognize_google")

    # Callback for transcripts
    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    # Create and start listener
    listener = Listener(
        source=audio_source,
        vad=vad,
        detector_config=DetectorConfig(frame_duration_ms=10),
        transcriber=transcriber,
        on_transcript=on_transcript,
    )

    print("\nListening with WebRTC VAD... (Ctrl+C to stop)")
    print("Aggressiveness mode: 1 (Quality)")
    print(f"Sample rate: {mic.SAMPLE_RATE} Hz")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
