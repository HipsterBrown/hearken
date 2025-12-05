# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`hearken` is a robust speech recognition pipeline for Python that prevents audio drops during transcription. The core innovation is decoupling audio capture, voice activity detection (VAD), and transcription into independent threads with queue-based communication.

## Core Architecture

The pipeline consists of three independent threads:

1. **Capture Thread**: Continuously reads 30ms audio chunks from microphone, never blocks on downstream processing
2. **Detect Thread**: Runs voice activity detection and finite state machine (FSM) to segment continuous audio into discrete speech segments
3. **Transcribe Thread**: Consumes segments and calls recognition backends (optional, passive mode only)

**Critical Design Principle**: The capture thread must NEVER block. Queues use explicit backpressure - when full, drop data with logging rather than blocking upstream.

### Key Components

- **Listener** (`listener.py`): Main pipeline class orchestrating the three-thread architecture
- **Data Types** (`types.py`): `AudioChunk`, `SpeechSegment`, `VADResult`, `DetectorConfig`, `DetectorState`
- **Interfaces** (`interfaces.py`): `AudioSource`, `Transcriber`, `VAD` abstract base classes
- **SpeechDetector** (`detector.py`): 4-state FSM for utterance segmentation
- **VAD Implementations**: `EnergyVAD` (baseline), `WebRTCVAD` (production-ready), `SileroVAD` (neural network)
- **Adapters** (`adapters/sr.py`): speech_recognition library integration

### Module Structure

```
hearken/
├── __init__.py           # Public API exports
├── listener.py           # Listener class (main pipeline)
├── types.py              # Core data types
├── interfaces.py         # Abstract interfaces
├── detector.py           # SpeechDetector FSM
├── vad/
│   ├── __init__.py
│   ├── energy.py         # EnergyVAD implementation
│   ├── webrtc.py         # WebRTCVAD implementation
│   └── silero.py         # SileroVAD implementation
└── adapters/
    ├── __init__.py
    └── sr.py             # speech_recognition adapters
```

## Development Commands

This project uses uv for dependency management (Python 3.11+).

### Environment Setup
```bash
# Install dependencies
uv sync

# Install with all extras (including speech_recognition)
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hearken --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_listener.py

# Verbose output
uv run pytest -v
```

### Code Quality
```bash
# Format code
uv run black hearken/ tests/ examples/

# Type checking
uv run mypy hearken/

# Linting
uv run ruff check hearken/ tests/ examples/
```

### Running Examples
```bash
# Basic passive mode example
uv run python examples/basic_usage.py

# Active mode example
uv run python examples/active_mode.py
```

## Architecture Details

### Three-Thread Pipeline

```
Microphone → [Capture Thread] → Queue → [Detect Thread] → Queue → [Transcribe Thread] → Callback
                   ↓                          ↓                         ↓
            AudioChunk (30ms)      SpeechSegment (complete)    Text transcription
```

### Two Modes of Operation

1. **Passive Mode** (callbacks):
   - Provide `on_speech` and/or `on_transcript` callbacks
   - Listener automatically processes audio and fires callbacks
   - Non-blocking, event-driven

2. **Active Mode** (polling):
   - Call `wait_for_speech()` to block until speech detected
   - Manually transcribe returned segments
   - Fine-grained control over processing

### Queue Backpressure Strategy

| Queue | Full Behavior | Rationale |
|-------|---------------|-----------|
| `capture_queue` | Drop newest chunk | Capture must never block; old audio more valuable during overflow |
| `segment_queue` | Drop with warning | Transcription backlog indicates systemic issue |

### FSM State Transitions

The detector FSM prevents false triggers from transient noise:

- **IDLE → SPEECH_STARTING**: Speech detected, start accumulating
- **SPEECH_STARTING → SPEAKING**: Duration exceeds `min_speech_duration` (default 250ms)
- **SPEECH_STARTING → IDLE**: Silence exceeds `silence_timeout` before min duration (false start)
- **SPEAKING → TRAILING_SILENCE**: Speech stops
- **TRAILING_SILENCE → SPEAKING**: Speech resumes (handles pauses)
- **TRAILING_SILENCE → IDLE**: Silence exceeds `silence_timeout` (emit segment)

## Testing Strategy

### Unit Tests
- Mock audio sources with controlled energy profiles
- Test VAD implementations with synthetic audio (silence, speech, noise)
- Test FSM transitions with controlled input sequences
- Verify queue backpressure behavior

### Integration Tests
- End-to-end pipeline tests with mock audio
- Verify segment detection and emission
- Test both active and passive modes
- Thread lifecycle and cleanup

### Test Execution
```bash
# All tests
uv run pytest

# Specific module
uv run pytest tests/test_listener.py

# With coverage
uv run pytest --cov=hearken --cov-report=html
```

## Common Pitfalls

1. **Blocking the capture thread**: Never call slow operations in capture loop. Queue puts must be non-blocking.

2. **Ignoring VAD constraints**: WebRTC VAD requires specific sample rates (8/16/32/48 kHz) and frame durations (10/20/30 ms). Validate early.

3. **Forgetting padding buffer**: Speech detection is reactive - without pre-roll padding, you'll miss the first 300ms of utterances.

4. **Not resetting VAD state**: Call `vad.reset()` when starting a new utterance (done automatically in detector).

5. **Assuming numpy dtypes**: Convert `numpy.bool_` and `numpy.float_` to Python types when needed.

6. **Silero VAD sample rate**: Silero VAD requires exactly 16kHz audio. Unlike WebRTC VAD which supports multiple rates, Silero is strict about 16kHz.

## Dependencies

### Required
- `numpy` >= 1.20 (audio processing, energy calculation)

### Optional
- `SpeechRecognition` >= 3.8 (recognition backends via adapters)
- `PyAudio` >= 0.2.11 (audio capture, installed via speech_recognition)
- `webrtcvad-wheels` >= 2.0.10 (WebRTC VAD support)
- `onnxruntime` >= 1.16.0 (Silero VAD support)

Install with: `pip install hearken[sr]` for speech_recognition support, `pip install hearken[webrtc]` for WebRTC VAD support, or `pip install hearken[silero]` for Silero VAD support

## Version

Current version: **0.3.0** (Silero VAD Release)

- 3-thread architecture with clean abstractions
- EnergyVAD, WebRTCVAD, and SileroVAD implementations
- 4-state FSM detector
- Active and passive modes
- speech_recognition adapters

Roadmap:
- ✅ v0.1: MVP with EnergyVAD
- ✅ v0.2: WebRTC VAD support
- ✅ v0.3: Silero VAD (neural network)
- v0.4: Async transcriber support
