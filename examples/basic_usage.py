"""
Basic hearken usage example with speech_recognition adapters.

Demonstrates passive mode with automatic transcription.
"""

import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    # Setup speech_recognition components
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Adjust for ambient noise
    print("Calibrating for ambient noise... (1 second)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold set to: {recognizer.energy_threshold}")

    # Create hearken components
    audio_source = SpeechRecognitionSource(mic)
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    # Callback for transcripts
    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    # Create and start listener
    listener = Listener(
        source=audio_source,
        transcriber=transcriber,
        vad=EnergyVAD(dynamic=True),
        on_transcript=on_transcript,
    )

    print("\nListening... (Ctrl+C to stop)")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
