# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sr_pipeline` is a userland wrapper for Python's `speech_recognition` library that fixes architectural limitations causing dropped audio frames during transcription. The core innovation is decoupling audio capture, voice activity detection (VAD), and transcription into independent threads with queue-based communication.

## Core Architecture

The pipeline consists of three independent threads:

1. **Capture Thread**: Continuously reads 30ms audio chunks from microphone, never blocks on downstream processing
2. **VAD Thread**: Runs voice activity detection and finite state machine (FSM) to segment continuous audio into discrete utterances
3. **Transcribe Thread**: Consumes utterances and calls recognition backends (Google, etc.)

**Critical Design Principle**: The capture thread must NEVER block. Queues use explicit backpressure - when full, drop data with metrics tracking rather than blocking upstream.

### Key Components

- **Data Classes** (`types.py`): `AudioChunk`, `Utterance`, `TranscriptResult`, `PipelineMetrics`
- **VAD Interface** (`vad/base.py`): Abstract base class for pluggable voice activity detectors
- **VAD Implementations**: `EnergyVAD` (baseline), `WebRTCVAD` (production quality)
- **Pipeline** (`pipeline.py`): `AudioPipeline` orchestrates the three-thread architecture
- **Detector FSM** (`detector.py`): States: IDLE → SPEECH_STARTING → SPEAKING → TRAILING_SILENCE

### Module Structure

```
sr_pipeline/
├── __init__.py           # Public API exports
├── pipeline.py           # AudioPipeline class
├── types.py              # AudioChunk, Utterance, TranscriptResult, etc.
├── vad/
│   ├── __init__.py       # VAD interface, factory function
│   ├── base.py           # VAD abstract base class
│   ├── energy.py         # EnergyVAD implementation
│   └── webrtc.py         # WebRTCVAD implementation
├── detector.py           # DetectorState FSM, DetectorConfig
├── audio/
│   ├── __init__.py
│   └── resample.py       # Sample rate conversion utilities
└── compat.py             # listen_in_background_improved() wrapper
```

## Development Commands

This project uses uv for dependency management (Python 3.11+).

### Environment Setup
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=sr_pipeline --cov-report=term-missing

# Run single test function
pytest tests/test_vad.py::test_webrtc_vad_basic
```

### Code Quality
```bash
# Format code
black sr_pipeline/ tests/

# Type checking
mypy sr_pipeline/

# Linting
ruff check sr_pipeline/ tests/
```

### Running Examples
```bash
# Basic pipeline example
python examples/basic_usage.py

# Google Cloud Speech-to-Text
python examples/google_cloud.py

# Drop-in replacement demo
python examples/compat_demo.py
```

## Technical Constraints

### WebRTC VAD Requirements

The `WebRTCVAD` implementation has strict requirements inherited from the underlying C library:

- **Sample rates**: Must be exactly 8000, 16000, 32000, or 48000 Hz
- **Frame durations**: Must be exactly 10, 20, or 30 ms
- **Audio format**: 16-bit mono PCM only

When implementing audio capture or resampling, validate these constraints early. The capture thread should check VAD requirements via `vad.required_sample_rate` and `vad.required_frame_duration_ms`.

### GIL Considerations

- **PyAudio releases the GIL** during blocking device reads - capture thread won't block other threads
- **Network I/O releases the GIL** during recognition API calls - transcription thread won't block capture
- **CPU-bound VAD** (like future Silero implementation) would require `multiprocessing` to avoid GIL contention

### Queue Backpressure Strategy

| Queue | Full Behavior | Rationale |
|-------|---------------|-----------|
| `capture_queue` | Drop newest chunk | Capture must never block; old audio more valuable during overflow |
| `utterance_queue` | Drop with warning | Transcription backlog indicates systemic issue |

## Implementation Patterns

### Adding a New VAD Implementation

1. Subclass `VAD` from `vad/base.py`
2. Implement `process(chunk: AudioChunk) -> VADResult`
3. Implement `reset()` for clearing state between utterances
4. Define `required_sample_rate` and `required_frame_duration_ms` properties
5. Add to factory function in `vad/__init__.py`
6. Add unit tests with mock audio data

### FSM State Transitions

The detector FSM prevents false triggers from transient noise:

- **IDLE → SPEECH_STARTING**: Speech detected, start accumulating
- **SPEECH_STARTING → SPEAKING**: Duration exceeds `min_speech_duration` (default 250ms)
- **SPEECH_STARTING → IDLE**: Silence exceeds `silence_timeout` before min duration (false start)
- **SPEAKING → TRAILING_SILENCE**: Speech stops
- **TRAILING_SILENCE → SPEAKING**: Speech resumes
- **TRAILING_SILENCE → IDLE**: Silence exceeds `silence_timeout` (emit utterance)

### Metrics and Observability

Always expose metrics through `PipelineMetrics`:
- `chunks_captured` / `chunks_dropped`: Monitor capture health
- `drop_rate` property: Percentage of dropped chunks (should be <1%)
- `utterances_detected` / `utterances_transcribed`: Monitor pipeline flow
- `transcription_errors`: Track API failures

## Testing Strategy

### Unit Tests
- Mock `AudioChunk` data with known energy profiles
- Test VAD implementations with synthetic audio (silence, speech, noise)
- Test FSM transitions with controlled input sequences
- Verify queue backpressure behavior

### Integration Tests
- Use real `sr.Microphone()` with test audio playback
- Verify end-to-end pipeline with known utterances
- Test sample rate conversion accuracy
- Measure dropped frame rates under load

### Performance Benchmarks
- Target platform: Raspberry Pi 4 (4GB)
- Metric: <1% dropped frames during 60s continuous speech
- Compare against baseline `listen_in_background()` (10-30% drops expected)

## Common Pitfalls

1. **Blocking the capture thread**: Never call slow operations in `_capture_loop()`. Queue puts must be non-blocking.

2. **Ignoring VAD constraints**: WebRTC VAD will raise exceptions if sample rate or frame size is wrong. Validate early in `start()`.

3. **Forgetting padding buffer**: Speech detection is reactive - without pre-roll padding, you'll miss the first 300ms of utterances.

4. **Not resetting VAD state**: Some VADs maintain internal state. Call `vad.reset()` when starting a new utterance.

5. **Assuming numpy dtypes**: Audio data is bytes. Convert to `np.int16` explicitly before processing.

## Dependencies

### Required
- `speech_recognition` >= 3.8 (recognition backends, Microphone abstraction)
- `pyaudio` >= 0.2.11 (audio capture, installed via speech_recognition)
- `numpy` >= 1.20 (audio processing, energy calculation)

### Optional
- `webrtcvad` >= 2.0.10 (production-quality VAD, recommended)
- `scipy` >= 1.7 (high-quality resampling via `signal.resample`)

## POC Status

A minimal proof of concept is implemented in `sr_pipeline_poc/` to validate the three-thread architecture.

**Running the POC:**
```bash
# Basic demo
python -m sr_pipeline_poc.demo

# Stress test (2s artificial delay per transcription)
python -m sr_pipeline_poc.stress_test
```

**Success criteria:**
- Drop rate < 1% during normal speech
- Drop rate < 1% even with artificial 2s transcription delay
- Works on both MacOS and Raspberry Pi 4

**What's included in POC:**
- EnergyVAD (no external dependencies)
- Simplified 2-state FSM (IDLE ↔ SPEAKING)
- Hardcoded configuration
- Google API transcription only

**What's NOT in POC (for full implementation):**
- WebRTC VAD
- Full 4-state FSM (SPEECH_STARTING, TRAILING_SILENCE states)
- Sample rate conversion
- Configuration system
- Unit tests

## Design Document

The complete architectural design, including motivation, alternatives considered, and implementation phases, is in `docs/plans/sr-pipeline-design-document.md`. Refer to it for:
- Problem analysis and root cause (why `listen_in_background()` drops frames)
- Detailed pipeline data flow diagrams
- Complete code examples for all components
- Performance comparison methodology
- Future work (Silero VAD, streaming recognition)
