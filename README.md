# Hearken

Robust speech recognition pipeline for Python that prevents audio drops during transcription.

## The Problem

In typical speech detection programs, audio capture is blocked during transcription. This causes dropped frames when network I/O is slow, resulting in missed speech.

## The Solution

Hearken decouples capture, voice activity detection (VAD), and transcription into independent threads with queue-based communication. The capture thread never blocks, preventing audio loss even during slow transcription.

## Installation

```bash
# Core package
pip install hearken

# With speech_recognition adapters (recommended for getting started)
pip install hearken[sr]
```

## Quick Start

```python
import speech_recognition as sr
from hearken import Listener, EnergyVAD
from hearken.adapters.sr import SpeechRecognitionSource, SRTranscriber

# Setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Create listener
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
    on_transcript=lambda text, seg: print(f"You said: {text}")
)

# Run
listener.start()
try:
    listener.wait()
except KeyboardInterrupt:
    listener.stop()
```

## Features

- **No dropped frames**: Capture thread never blocks on downstream processing
- **Flexible VAD**: Pluggable voice activity detection (Energy-based included, WebRTC coming soon)
- **Two modes**: Passive (callbacks) and active (`wait_for_speech()`)
- **Clean abstractions**: Bring your own audio source and transcriber
- **Production-ready FSM**: Robust 4-state detector filters false starts and handles pauses

## Architecture

```
Microphone → [Capture Thread] → Queue → [Detect Thread] → Queue → [Transcribe Thread] → Callback
                   ↓                          ↓                         ↓
            AudioChunk (30ms)      SpeechSegment (complete)    Text transcription
```

## Active Mode

```python
listener = Listener(
    source=SpeechRecognitionSource(mic),
    transcriber=SRTranscriber(recognizer),
)

listener.start()

while True:
    print("Waiting for speech...")
    segment = listener.wait_for_speech()

    if segment:
        try:
            text = listener.transcriber.transcribe(segment)
            print(f"You said: {text}")
        except sr.UnknownValueError:
            print("Could not understand")
```

## Documentation

See [examples/](examples/) for more usage patterns.

## Development

```bash
# Clone repository
git clone https://github.com/hipsterbrown/hearken.git
cd hearken

# Install with dev dependencies
uv sync --all-extras

# Run tests
pytest

# Run tests with coverage
pytest --cov=hearken --cov-report=term-missing

# Format code
black hearken/ tests/

# Type checking
mypy hearken/

# Linting
ruff check hearken/ tests/
```

## Roadmap

- v0.1: EnergyVAD, core pipeline ✓
- v0.2: WebRTC VAD support
- v0.3: Async transcriber support
- v0.4: Silero VAD (neural network)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Credits

Created by Nick Hehr (@hipsterbrown)
