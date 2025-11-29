"""
Active mode example - explicitly request speech segments.

Useful for conversational interfaces or when you need control
over when transcription happens.
"""

import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber


def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Calibrating for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    audio_source = SpeechRecognitionSource(mic)
    transcriber = SRTranscriber(recognizer)

    listener = Listener(
        source=audio_source,
        transcriber=transcriber,
        vad=EnergyVAD(),
    )

    listener.start()

    print("Active listening mode demo")
    print("Say something, then I'll transcribe it.\n")

    try:
        while True:
            print("Waiting for speech...")
            segment = listener.wait_for_speech()

            if segment:
                print(f"Got {segment.duration:.1f}s of audio, transcribing...")
                try:
                    text = transcriber.transcribe(segment)
                    print(f"You said: {text}\n")
                except sr.UnknownValueError:
                    print("Could not understand audio\n")
                except sr.RequestError as e:
                    print(f"API error: {e}\n")

    except KeyboardInterrupt:
        print("\nStopping...")
        listener.stop()


if __name__ == "__main__":
    main()
