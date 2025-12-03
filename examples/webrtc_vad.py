"""
Example: Using WebRTC VAD for improved accuracy.

Demonstrates using WebRTC VAD instead of EnergyVAD for more robust
speech detection in noisy environments.

Requirements:
    pip install hearken[webrtc,sr]
"""

import speech_recognition as sr
from hearken import Listener
from hearken.vad import WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    # Setup speech_recognition components
    recognizer = sr.Recognizer()
    # WebRTC VAD requires supported sample rate (8/16/32/48 kHz)
    mic = sr.Microphone(sample_rate=16000)

    print("Calibrating for ambient noise... (1 second)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold set to: {recognizer.energy_threshold}")

    # Create hearken components with WebRTC VAD
    audio_source = SpeechRecognitionSource(mic)
    vad = WebRTCVAD(aggressiveness=2)  # Low bitrate mode
    transcriber = SRTranscriber(recognizer, method='recognize_google')

    # Callback for transcripts
    def on_transcript(text: str, segment):
        print(f"[{segment.duration:.1f}s] You said: {text}")

    # Create and start listener
    listener = Listener(
        source=audio_source,
        vad=vad,
        transcriber=transcriber,
        on_transcript=on_transcript,
    )

    print("\nListening with WebRTC VAD... (Ctrl+C to stop)")
    print("Aggressiveness mode: 2 (low bitrate)")
    print("Sample rate: 16000 Hz")
    listener.start()

    try:
        listener.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
