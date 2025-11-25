# Speech Recognition Pipeline POC Design

**Date:** 2025-11-25
**Author:** Nick Hehr
**Status:** Ready for Implementation
**Goal:** Validate three-thread architecture with minimal complexity

---

## Overview

This proof of concept implements the core three-thread architecture from the main design document with minimal complexity to validate the key hypothesis: **the capture thread never blocks during transcription**.

### Success Criteria

- ✅ Drop rate < 1% during normal speech
- ✅ Drop rate < 1% even during continuous 30s speech
- ✅ Transcriptions are accurate (Google API working)
- ✅ Graceful handling of errors (API timeout, etc.)
- ✅ Works on both MacOS and Raspberry Pi

---

## Architecture

### Structure

```
sr_pipeline_poc/
├── types.py          # AudioChunk, Utterance, PipelineMetrics dataclasses
├── energy_vad.py     # Simple RMS energy-based VAD
├── pipeline.py       # AudioPipeline with 3 threads + simplified 2-state FSM
└── demo.py           # Example script with metrics output
```

### What's Included

- Three independent threads (capture, detect, transcribe) with queue-based communication
- EnergyVAD with dynamic threshold adjustment (ambient noise calibration)
- Simplified 2-state detector (IDLE ↔ SPEAKING)
- PipelineMetrics tracking (chunks captured/dropped, utterances detected/transcribed)
- Google Speech API for transcription

### What's Excluded (POC Simplifications)

- WebRTC VAD (no external C dependencies)
- Full 4-state FSM (no SPEECH_STARTING or TRAILING_SILENCE states)
- Sample rate conversion (assume mic is compatible)
- Configuration files or CLI flags (hardcoded sensible defaults)
- Comprehensive error recovery (just hybrid fatal/warn approach)
- Unit tests (end-to-end validation only)

---

## Data Flow

### Pipeline Overview

```
Microphone → [Capture Thread] → capture_queue → [Detect Thread] → utterance_queue → [Transcribe Thread] → Callback
                   ↓                                    ↓                                    ↓
            AudioChunk (30ms)                    Utterance (complete)              print(text + metrics)
```

### Queue Configuration

**capture_queue:**
- maxsize=100 (~3 seconds of audio buffer)
- When full: Drop newest chunk, increment `metrics.chunks_dropped`
- Capture thread uses `put_nowait()` - never blocks

**utterance_queue:**
- maxsize=10 (enough for ~10 pending transcriptions)
- When full: Drop utterance with warning print
- Detect thread uses `put_nowait()` - never blocks

### Threading Details

- All threads are daemon threads (auto-cleanup on exit)
- Graceful shutdown via `stop()`: sets stop event, sends poison pills (None), joins with timeout
- Each thread catches exceptions, calls error callback, continues or exits based on error type

**Critical invariant:** Capture thread read loop has ZERO blocking operations except the PyAudio device read (which releases GIL). No locks, no blocking queue operations, no I/O.

---

## Simplified 2-State Detector

### States

- **IDLE**: Waiting for speech to begin
- **SPEAKING**: Accumulating audio for an utterance

### State Transitions

**IDLE:**
- Maintain ring buffer of last N chunks (for speech padding)
- When VAD detects speech → SPEAKING
- Copy padding buffer into utterance accumulator

**SPEAKING:**
- Accumulate all chunks into utterance
- Track last_speech_time when VAD says "speech"
- When silence duration > silence_timeout → emit utterance, return to IDLE
- When utterance duration > max_speech_duration → force emit, return to IDLE

### Configuration (Hardcoded)

```python
ENERGY_THRESHOLD = 300.0      # RMS energy level for speech
SILENCE_TIMEOUT = 0.8         # seconds of silence to end utterance
SPEECH_PADDING = 0.3          # seconds of audio to prepend
MAX_SPEECH_DURATION = 30.0    # force split after 30s
```

### What We're Skipping from Full FSM

- No SPEECH_STARTING state (no filtering of transient noise < 250ms)
- No TRAILING_SILENCE state (can't resume speech after brief pause)
- No min_speech_duration check (may emit very short utterances)

This simplified FSM still segments continuous audio into utterances, which is enough to validate the pipeline architecture.

---

## EnergyVAD Implementation

### Algorithm

```python
class EnergyVAD:
    def __init__(self, threshold=300.0, dynamic=True):
        self.threshold = threshold
        self.dynamic = dynamic
        self._ambient_energy = None
        self._samples_seen = 0

    def process(self, chunk: AudioChunk) -> VADResult:
        # Convert bytes to int16 samples
        samples = np.frombuffer(chunk.data, dtype=np.int16)

        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        # Dynamic threshold: calibrate during first ~1.5s
        if self.dynamic and self._samples_seen < 50:
            # Exponential moving average of ambient noise
            if self._ambient_energy is None:
                self._ambient_energy = energy
            else:
                self._ambient_energy = 0.9 * self._ambient_energy + 0.1 * energy
            self._samples_seen += 1
            effective_threshold = max(self.threshold, self._ambient_energy * 1.5)
        else:
            effective_threshold = self.threshold

        is_speech = energy > effective_threshold
        return VADResult(is_speech, confidence=1.0 if is_speech else 0.0)
```

### Key Features

- Simple RMS (root mean square) energy calculation - fast, no complex dependencies
- Dynamic threshold adjustment during startup (mimics `adjust_for_ambient_noise()`)
- After calibration (~1.5s), uses fixed threshold
- No reset needed (stateless after calibration)

### Trade-offs

- Less accurate than WebRTC VAD (will trigger on loud non-speech sounds)
- Good enough to validate pipeline architecture
- Matches speech_recognition's built-in behavior, so it's a fair baseline

---

## Demo Script

### Flow

```python
# demo.py
def main():
    # 1. Setup
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # 2. Ambient noise calibration
    print("Calibrating for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold: {recognizer.energy_threshold}")

    # 3. Create and start pipeline
    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=mic,
        on_transcript=handle_transcript,
        on_error=handle_error,
    )

    print("Listening... (Ctrl+C to stop)")
    pipeline.start()

    # 4. Periodic metrics output
    try:
        while True:
            time.sleep(5)
            print_metrics(pipeline.metrics)
    except KeyboardInterrupt:
        pipeline.stop()
```

### Expected Output

```
Calibrating for ambient noise...
Energy threshold: 287.3
Listening... (Ctrl+C to stop)

[0.8s] You said: hello there
[1.2s] You said: testing the microphone
Stats: captured=400, dropped=0 (0.0%), utterances=2/2

[2.1s] You said: this is working great
Stats: captured=700, dropped=0 (0.0%), utterances=3/3

^C
Stopping...
Final stats: captured=1234, dropped=0 (0.0%), utterances=5/5
```

### Error Handling

- **Fatal errors** (crash with helpful message):
  - Microphone not found
  - Invalid configuration

- **Warnings** (log and continue):
  - Google API timeout → print "Transcription error: <msg>", continue listening
  - Dropped frame → increment metric silently, show in stats

---

## Testing Strategy

### Manual Testing Approach

**Test 1: Basic functionality**
- Run `demo.py` on MacOS laptop
- Speak a few sentences with natural pauses
- Verify: transcriptions appear, metrics show 0% drop rate

**Test 2: Stress test (validate threading)**
- Speak continuously for 30+ seconds without pausing
- Verify: capture continues during transcription (no dropped frames)
- Expected: drop_rate stays at 0.0% even during slow API responses

**Test 3: Network latency simulation**
- Add artificial delay in transcription (sleep before API call)
- Verify: capture_queue continues filling, no frames dropped
- This proves capture truly doesn't block on transcription

**Test 4: Raspberry Pi comparison**
- Run same tests on Pi 4
- Compare drop rates between MacOS and Pi
- Baseline: run speech_recognition's `listen_in_background()` for comparison

---

## Dependencies

### Required

```toml
[project]
dependencies = [
    "SpeechRecognition>=3.8",
    "PyAudio>=0.2.11",
    "numpy>=1.20",
]
```

### Installation

```bash
# MacOS (may need portaudio)
brew install portaudio
uv sync

# Raspberry Pi OS
sudo apt-get install portaudio19-dev python3-pyaudio
uv sync
```

---

## Implementation Notes

### Critical Design Principles

1. **Capture thread never blocks** - Use `put_nowait()` for all queue operations in capture loop
2. **Daemon threads** - All worker threads should be daemon threads for clean shutdown
3. **Poison pills** - Use `None` sentinels to signal thread shutdown
4. **GIL awareness** - PyAudio device reads and Google API calls both release the GIL

### Common Pitfalls to Avoid

1. Don't use `queue.put()` (blocking) in capture thread - always `put_nowait()`
2. Don't forget to handle `queue.Full` exceptions when dropping frames
3. Don't forget poison pills in `stop()` or threads may hang on `queue.get()`
4. Don't forget to join threads with timeout to avoid infinite waits

### Performance Expectations

- **MacOS**: Should achieve 0% drop rate under normal conditions
- **Raspberry Pi 4**: Should achieve <1% drop rate under normal conditions
- **Baseline comparison**: speech_recognition's `listen_in_background()` typically shows 10-30% drop rate on Pi during transcription

---

## Next Steps After POC

If POC validates the architecture:

1. Add WebRTC VAD support
2. Implement full 4-state FSM
3. Add sample rate conversion
4. Add comprehensive unit tests
5. Add configuration system
6. Performance benchmarking suite
7. Package for PyPI distribution

---

## References

- Main design document: `docs/plans/sr-pipeline-design-document.md`
- speech_recognition library: https://github.com/Uberi/speech_recognition
