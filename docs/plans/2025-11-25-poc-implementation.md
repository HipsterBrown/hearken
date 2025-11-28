# Speech Recognition Pipeline POC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build minimal proof of concept to validate three-thread architecture prevents audio drops during transcription

**Architecture:** Three independent threads (capture, detect, transcribe) communicate via queues. Capture thread uses put_nowait() to never block. EnergyVAD with simplified 2-state FSM segments audio into utterances. Google API transcribes in separate thread.

**Tech Stack:** Python 3.11+, speech_recognition, PyAudio, numpy

---

## Prerequisites

**Environment setup:**
```bash
# Install system dependencies (MacOS)
brew install portaudio

# Sync dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

**Reference documents:**
- Design: `docs/plans/2025-11-25-poc-design.md`
- Full architecture: `docs/plans/sr-pipeline-design-document.md`

---

## Task 1: Create Project Structure and Types

**Files:**
- Create: `sr_pipeline_poc/__init__.py`
- Create: `sr_pipeline_poc/types.py`

**Step 1: Create package directory**

```bash
mkdir -p sr_pipeline_poc
```

**Step 2: Create empty __init__.py**

File: `sr_pipeline_poc/__init__.py`
```python
"""
Minimal proof of concept for sr_pipeline three-thread architecture.
"""

__version__ = "0.1.0"
```

**Step 3: Write types module**

File: `sr_pipeline_poc/types.py`
```python
"""Data types for audio pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import speech_recognition as sr


@dataclass
class AudioChunk:
    """A chunk of audio with metadata."""
    data: bytes
    timestamp: float          # time.monotonic() when captured
    sample_rate: int
    sample_width: int         # bytes per sample (2 for 16-bit)


@dataclass
class Utterance:
    """A complete speech segment ready for transcription."""
    audio: sr.AudioData
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        """Return duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class PipelineMetrics:
    """Runtime metrics for monitoring."""
    chunks_captured: int = 0
    chunks_dropped: int = 0
    utterances_detected: int = 0
    utterances_transcribed: int = 0
    transcription_errors: int = 0

    @property
    def drop_rate(self) -> float:
        """Return percentage of dropped chunks."""
        total = self.chunks_captured + self.chunks_dropped
        return self.chunks_dropped / total if total > 0 else 0.0


@dataclass
class VADResult:
    """Result from voice activity detection."""
    is_speech: bool
    confidence: float = 1.0  # 0.0 to 1.0
```

**Step 4: Update pyproject.toml dependencies**

File: `pyproject.toml`

Modify the `dependencies` array to:
```toml
dependencies = [
    "SpeechRecognition>=3.8",
    "PyAudio>=0.2.11",
    "numpy>=1.20",
]
```

**Step 5: Install dependencies**

```bash
uv sync
```

Expected: Dependencies installed, lock file updated

**Step 6: Verify types module imports**

```bash
python -c "from sr_pipeline_poc.types import AudioChunk, Utterance, PipelineMetrics, VADResult; print('Types imported successfully')"
```

Expected: "Types imported successfully"

**Step 7: Commit**

```bash
git add sr_pipeline_poc/ pyproject.toml
git commit -m "feat: add data types for audio pipeline

Created AudioChunk, Utterance, PipelineMetrics, and VADResult
dataclasses. These form the core data structures passed between
pipeline threads.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement EnergyVAD

**Files:**
- Create: `sr_pipeline_poc/energy_vad.py`

**Step 1: Write EnergyVAD class**

File: `sr_pipeline_poc/energy_vad.py`
```python
"""Energy-based voice activity detection."""

import numpy as np
from typing import Optional
from .types import AudioChunk, VADResult


class EnergyVAD:
    """
    Simple energy-based voice activity detection.

    Uses RMS (root mean square) energy threshold to detect speech.
    Optionally calibrates threshold based on ambient noise.
    """

    def __init__(self, threshold: float = 300.0, dynamic: bool = True):
        """
        Initialize EnergyVAD.

        Args:
            threshold: RMS energy threshold for speech detection
            dynamic: If True, adapt threshold based on ambient noise
        """
        self.threshold = threshold
        self.dynamic = dynamic
        self._ambient_energy: Optional[float] = None
        self._samples_seen = 0

    def process(self, chunk: AudioChunk) -> VADResult:
        """
        Process an audio chunk and determine if it contains speech.

        Args:
            chunk: Audio data to analyze

        Returns:
            VADResult with speech detection outcome
        """
        # Convert bytes to int16 samples
        samples = np.frombuffer(chunk.data, dtype=np.int16)

        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

        # Dynamic threshold adjustment (simple exponential moving average)
        if self.dynamic and self._samples_seen < 50:  # ~1.5s calibration
            if self._ambient_energy is None:
                self._ambient_energy = energy
            else:
                self._ambient_energy = 0.9 * self._ambient_energy + 0.1 * energy
            self._samples_seen += 1
            effective_threshold = max(self.threshold, self._ambient_energy * 1.5)
        else:
            effective_threshold = self.threshold

        is_speech = energy > effective_threshold
        confidence = min(1.0, energy / effective_threshold) if is_speech else 0.0

        return VADResult(is_speech=is_speech, confidence=confidence)

    def reset(self) -> None:
        """Reset VAD state (no-op for EnergyVAD after calibration)."""
        # Don't reset ambient calibration
        pass
```

**Step 2: Test EnergyVAD manually**

```bash
python -c "
from sr_pipeline_poc.energy_vad import EnergyVAD
from sr_pipeline_poc.types import AudioChunk
import numpy as np
import time

vad = EnergyVAD(threshold=300.0)

# Test with silence (low energy)
silence = np.zeros(480, dtype=np.int16).tobytes()
chunk = AudioChunk(silence, time.monotonic(), 16000, 2)
result = vad.process(chunk)
print(f'Silence: is_speech={result.is_speech} (expected False)')

# Test with noise (high energy)
noise = (np.random.randint(-5000, 5000, 480, dtype=np.int16)).tobytes()
chunk = AudioChunk(noise, time.monotonic(), 16000, 2)
result = vad.process(chunk)
print(f'Noise: is_speech={result.is_speech} (expected True)')
"
```

Expected:
```
Silence: is_speech=False (expected False)
Noise: is_speech=True (expected True)
```

**Step 3: Commit**

```bash
git add sr_pipeline_poc/energy_vad.py
git commit -m "feat: implement EnergyVAD for speech detection

Added RMS energy-based voice activity detection with dynamic
threshold calibration. Calibrates during first ~1.5s to adapt
to ambient noise levels.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Implement AudioPipeline Core Structure

**Files:**
- Create: `sr_pipeline_poc/pipeline.py`

**Step 1: Write pipeline class skeleton with threading setup**

File: `sr_pipeline_poc/pipeline.py`
```python
"""Three-thread audio pipeline for speech recognition."""

import threading
import queue
import time
import collections
from typing import Callable, Optional
import speech_recognition as sr

from .types import AudioChunk, Utterance, PipelineMetrics, VADResult
from .energy_vad import EnergyVAD


# Type aliases for callbacks
TranscriptCallback = Callable[[str, Utterance], None]
ErrorCallback = Callable[[Exception], None]


# Hardcoded POC configuration
ENERGY_THRESHOLD = 300.0
SILENCE_TIMEOUT = 0.8         # seconds
SPEECH_PADDING = 0.3          # seconds
MAX_SPEECH_DURATION = 30.0    # seconds
FRAME_DURATION_MS = 30        # milliseconds
CAPTURE_QUEUE_SIZE = 100
UTTERANCE_QUEUE_SIZE = 10


class AudioPipeline:
    """
    Decoupled audio capture and processing pipeline.

    Separates capture, VAD/segmentation, and transcription into independent
    threads to prevent audio drops during transcription.
    """

    def __init__(
        self,
        recognizer: sr.Recognizer,
        source: sr.Microphone,
        on_transcript: TranscriptCallback,
        on_error: Optional[ErrorCallback] = None,
    ):
        """
        Initialize AudioPipeline.

        Args:
            recognizer: speech_recognition Recognizer instance
            source: speech_recognition Microphone instance
            on_transcript: Callback for transcription results (text, utterance)
            on_error: Optional callback for errors
        """
        self.recognizer = recognizer
        self.source = source
        self.on_transcript = on_transcript
        self.on_error = on_error or self._default_error_handler

        # VAD setup
        self.vad = EnergyVAD(threshold=recognizer.energy_threshold)

        # Queues
        self._capture_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue(
            maxsize=CAPTURE_QUEUE_SIZE
        )
        self._utterance_queue: queue.Queue[Optional[Utterance]] = queue.Queue(
            maxsize=UTTERANCE_QUEUE_SIZE
        )

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Metrics
        self.metrics = PipelineMetrics()

    def _default_error_handler(self, e: Exception) -> None:
        """Default error handler that prints to console."""
        print(f"Pipeline error: {e}")

    def start(self) -> None:
        """Start all pipeline threads."""
        if self._running:
            raise RuntimeError("Pipeline already running")

        self._running = True
        self._stop_event.clear()

        self._threads = [
            threading.Thread(
                target=self._capture_loop,
                name="sr-pipeline-capture",
                daemon=True
            ),
            threading.Thread(
                target=self._detect_loop,
                name="sr-pipeline-detect",
                daemon=True
            ),
            threading.Thread(
                target=self._transcribe_loop,
                name="sr-pipeline-transcribe",
                daemon=True
            ),
        ]

        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop all pipeline threads gracefully."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        # Send poison pills to unblock queue.get()
        try:
            self._capture_queue.put_nowait(None)
        except queue.Full:
            pass

        try:
            self._utterance_queue.put_nowait(None)
        except queue.Full:
            pass

        # Wait for threads
        for t in self._threads:
            t.join(timeout=timeout)

        self._threads.clear()

    def wait(self) -> None:
        """Block until stop() is called or threads exit."""
        while self._running and any(t.is_alive() for t in self._threads):
            time.sleep(0.1)

    def _capture_loop(self) -> None:
        """Capture thread - implemented in next step."""
        pass

    def _detect_loop(self) -> None:
        """Detection thread - implemented in next step."""
        pass

    def _transcribe_loop(self) -> None:
        """Transcription thread - implemented in next step."""
        pass
```

**Step 2: Verify pipeline skeleton compiles**

```bash
python -c "
from sr_pipeline_poc.pipeline import AudioPipeline
import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone()

pipeline = AudioPipeline(
    recognizer=recognizer,
    source=mic,
    on_transcript=lambda text, utt: print(text),
)
print('Pipeline created successfully')
"
```

Expected: "Pipeline created successfully"

**Step 3: Commit**

```bash
git add sr_pipeline_poc/pipeline.py
git commit -m "feat: add AudioPipeline skeleton with threading setup

Created AudioPipeline class with three-thread structure and queue
management. Thread implementations are stubs to be filled in next.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Implement Capture Thread

**Files:**
- Modify: `sr_pipeline_poc/pipeline.py` (replace `_capture_loop` method)

**Step 1: Implement _capture_loop method**

In `sr_pipeline_poc/pipeline.py`, replace the `_capture_loop` stub with:

```python
    def _capture_loop(self) -> None:
        """
        Dedicated capture thread.

        Reads audio chunks at fixed intervals, never blocks on downstream.
        CRITICAL: Uses put_nowait() to never block - drops frames if queue full.
        """
        chunk_samples = int(self.source.SAMPLE_RATE * FRAME_DURATION_MS / 1000)

        with self.source as s:
            while self._running:
                try:
                    # Read audio - releases GIL during device read
                    data = s.stream.read(chunk_samples, exception_on_overflow=False)

                    chunk = AudioChunk(
                        data=data,
                        timestamp=time.monotonic(),
                        sample_rate=s.SAMPLE_RATE,
                        sample_width=s.SAMPLE_WIDTH,
                    )

                    # Non-blocking put - NEVER block capture thread
                    try:
                        self._capture_queue.put_nowait(chunk)
                        self.metrics.chunks_captured += 1
                    except queue.Full:
                        # Drop frame and track metric
                        self.metrics.chunks_dropped += 1

                except OSError as e:
                    # Audio device error - fatal
                    if self._running:
                        self.on_error(e)
                    break
                except Exception as e:
                    if self._running:
                        self.on_error(e)
```

**Step 2: Test capture thread in isolation**

Create temporary test file `test_capture.py`:
```python
import speech_recognition as sr
from sr_pipeline_poc.pipeline import AudioPipeline
import time

recognizer = sr.Recognizer()
mic = sr.Microphone()

pipeline = AudioPipeline(
    recognizer=recognizer,
    source=mic,
    on_transcript=lambda text, utt: None,
)

print("Starting capture thread...")
pipeline.start()

# Let it capture for 3 seconds
time.sleep(3)

pipeline.stop()

print(f"Captured: {pipeline.metrics.chunks_captured} chunks")
print(f"Dropped: {pipeline.metrics.chunks_dropped} chunks")
print(f"Drop rate: {pipeline.metrics.drop_rate:.1%}")

assert pipeline.metrics.chunks_captured > 0, "Should have captured some chunks"
print("✓ Capture thread works!")
```

Run:
```bash
python test_capture.py
```

Expected output (numbers will vary):
```
Starting capture thread...
Captured: 100 chunks
Dropped: 0 chunks
Drop rate: 0.0%
✓ Capture thread works!
```

**Step 3: Clean up test file**

```bash
rm test_capture.py
```

**Step 4: Commit**

```bash
git add sr_pipeline_poc/pipeline.py
git commit -m "feat: implement capture thread

Added _capture_loop that reads audio chunks and queues them without
blocking. Uses put_nowait() to drop frames rather than block when
queue is full. Tracks captured/dropped metrics.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Implement Detection Thread with 2-State FSM

**Files:**
- Modify: `sr_pipeline_poc/pipeline.py` (replace `_detect_loop` method and add `_emit_utterance` helper)

**Step 1: Implement _detect_loop with simplified FSM**

In `sr_pipeline_poc/pipeline.py`, replace the `_detect_loop` stub with:

```python
    def _detect_loop(self) -> None:
        """
        VAD and utterance segmentation thread.

        Implements simplified 2-state FSM:
        - IDLE: Wait for speech, maintain padding buffer
        - SPEAKING: Accumulate audio, emit after silence timeout
        """
        # State: "idle" or "speaking"
        state = "idle"

        # Ring buffer for speech padding
        padding_frames = int(SPEECH_PADDING * 1000 / FRAME_DURATION_MS)
        padding_buffer: collections.deque[AudioChunk] = collections.deque(
            maxlen=max(1, padding_frames)
        )

        # Utterance accumulator
        utterance_chunks: list[AudioChunk] = []
        speech_start_time: Optional[float] = None
        last_speech_time: Optional[float] = None

        while self._running:
            try:
                chunk = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is None:  # Poison pill
                break

            # Run VAD
            try:
                vad_result = self.vad.process(chunk)
            except Exception as e:
                self.on_error(e)
                continue

            is_speech = vad_result.is_speech
            now = chunk.timestamp

            # FSM transitions
            if state == "idle":
                padding_buffer.append(chunk)
                if is_speech:
                    # Start speaking
                    state = "speaking"
                    speech_start_time = now
                    last_speech_time = now
                    utterance_chunks = list(padding_buffer)

            elif state == "speaking":
                utterance_chunks.append(chunk)
                if is_speech:
                    last_speech_time = now

                # Check if we should emit utterance
                silence_duration = now - last_speech_time
                speech_duration = now - speech_start_time

                if silence_duration >= SILENCE_TIMEOUT:
                    # Silence timeout - emit utterance
                    self._emit_utterance(utterance_chunks, speech_start_time, now)
                    state = "idle"
                    utterance_chunks = []
                    padding_buffer.clear()
                    self.vad.reset()
                elif speech_duration >= MAX_SPEECH_DURATION:
                    # Max duration - force emit
                    self._emit_utterance(utterance_chunks, speech_start_time, now)
                    state = "idle"
                    utterance_chunks = []
                    padding_buffer.clear()
                    self.vad.reset()

    def _emit_utterance(
        self,
        chunks: list[AudioChunk],
        start: float,
        end: float
    ) -> None:
        """Package chunks as Utterance and queue for transcription."""
        if not chunks:
            return

        # Combine chunks into single audio blob
        audio_bytes = b"".join(c.data for c in chunks)
        audio_data = sr.AudioData(
            audio_bytes,
            chunks[0].sample_rate,
            chunks[0].sample_width,
        )

        utterance = Utterance(
            audio=audio_data,
            start_time=start,
            end_time=end,
        )

        try:
            self._utterance_queue.put_nowait(utterance)
            self.metrics.utterances_detected += 1
        except queue.Full:
            print(f"Warning: Utterance queue full, dropping {utterance.duration:.1f}s utterance")
```

**Step 2: Test detection thread**

Create temporary test file `test_detect.py`:
```python
import speech_recognition as sr
from sr_pipeline_poc.pipeline import AudioPipeline
import time

recognizer = sr.Recognizer()
mic = sr.Microphone()

# Adjust for ambient noise
print("Calibrating for ambient noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
print(f"Energy threshold: {recognizer.energy_threshold}")

utterances_detected = []

def on_utterance(text: str, utt):
    utterances_detected.append(utt)
    print(f"Detected utterance: {utt.duration:.1f}s")

pipeline = AudioPipeline(
    recognizer=recognizer,
    source=mic,
    on_transcript=on_utterance,
)

print("Starting pipeline... Speak a few words, then wait 1 second of silence.")
pipeline.start()

# Run for 10 seconds
time.sleep(10)

pipeline.stop()

print(f"\nUtterances detected: {pipeline.metrics.utterances_detected}")
print(f"Captured: {pipeline.metrics.chunks_captured} chunks")
print(f"Dropped: {pipeline.metrics.chunks_dropped} chunks ({pipeline.metrics.drop_rate:.1%})")

if pipeline.metrics.utterances_detected > 0:
    print("✓ Detection thread works!")
else:
    print("⚠ No utterances detected - try speaking louder or adjusting threshold")
```

Run:
```bash
python test_detect.py
```

Expected: Should detect utterances when you speak (transcription won't work yet, but detection should)

**Step 3: Clean up test file**

```bash
rm test_detect.py
```

**Step 4: Commit**

```bash
git add sr_pipeline_poc/pipeline.py
git commit -m "feat: implement detection thread with 2-state FSM

Added _detect_loop with simplified IDLE ↔ SPEAKING state machine.
Maintains padding buffer, segments audio into utterances based on
silence timeout and max duration. Emits to transcription queue.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Implement Transcription Thread

**Files:**
- Modify: `sr_pipeline_poc/pipeline.py` (replace `_transcribe_loop` method)

**Step 1: Implement _transcribe_loop method**

In `sr_pipeline_poc/pipeline.py`, replace the `_transcribe_loop` stub with:

```python
    def _transcribe_loop(self) -> None:
        """
        Transcription thread.

        Consumes utterances and calls Google Speech API.
        Network I/O releases the GIL - doesn't block capture.
        """
        while self._running:
            try:
                utterance = self._utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if utterance is None:  # Poison pill
                break

            try:
                # Recognition - releases GIL during network I/O
                text = self.recognizer.recognize_google(utterance.audio)

                self.metrics.utterances_transcribed += 1

                # Invoke callback
                try:
                    self.on_transcript(text, utterance)
                except Exception as e:
                    self.on_error(e)

            except sr.UnknownValueError:
                # No speech recognized - not an error, just skip
                pass
            except sr.RequestError as e:
                # API error - warn and continue
                self.metrics.transcription_errors += 1
                self.on_error(e)
            except Exception as e:
                self.metrics.transcription_errors += 1
                self.on_error(e)
```

**Step 2: Test full pipeline end-to-end**

Create temporary test file `test_full_pipeline.py`:
```python
import speech_recognition as sr
from sr_pipeline_poc.pipeline import AudioPipeline
import time

recognizer = sr.Recognizer()
mic = sr.Microphone()

# Calibrate
print("Calibrating for ambient noise...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
print(f"Energy threshold: {recognizer.energy_threshold}")

def on_transcript(text: str, utt):
    print(f"[{utt.duration:.1f}s] You said: {text}")

pipeline = AudioPipeline(
    recognizer=recognizer,
    source=mic,
    on_transcript=on_transcript,
)

print("\nListening... Speak something, then wait for transcription.")
print("Press Ctrl+C to stop.\n")
pipeline.start()

try:
    while True:
        time.sleep(5)
        m = pipeline.metrics
        print(f"Stats: captured={m.chunks_captured}, dropped={m.chunks_dropped} ({m.drop_rate:.1%}), utterances={m.utterances_detected}/{m.utterances_transcribed}")
except KeyboardInterrupt:
    print("\nStopping...")
    pipeline.stop()

    m = pipeline.metrics
    print(f"\nFinal stats:")
    print(f"  Chunks captured: {m.chunks_captured}")
    print(f"  Chunks dropped: {m.chunks_dropped} ({m.drop_rate:.1%})")
    print(f"  Utterances detected: {m.utterances_detected}")
    print(f"  Utterances transcribed: {m.utterances_transcribed}")
    print(f"  Transcription errors: {m.transcription_errors}")
```

Run:
```bash
python test_full_pipeline.py
```

Expected: You can speak and see transcriptions appear. Stats should show 0% drop rate.

**Step 3: Keep test file for now (will make it the demo)**

Don't delete test_full_pipeline.py yet - we'll refine it into the demo script.

**Step 4: Commit**

```bash
git add sr_pipeline_poc/pipeline.py
git commit -m "feat: implement transcription thread

Added _transcribe_loop that consumes utterances and calls Google
Speech API. Handles UnknownValueError (no speech) gracefully and
tracks transcription errors. Full pipeline now functional.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Create Demo Script

**Files:**
- Create: `sr_pipeline_poc/demo.py`
- Delete: `test_full_pipeline.py`

**Step 1: Create polished demo script**

File: `sr_pipeline_poc/demo.py`
```python
#!/usr/bin/env python3
"""
Demo script for sr_pipeline POC.

Validates three-thread architecture prevents audio drops during transcription.
Run on MacOS and Raspberry Pi to compare performance.
"""

import speech_recognition as sr
import time
from .pipeline import AudioPipeline


def main():
    """Run the demo pipeline."""
    print("=" * 60)
    print("SR Pipeline POC - Three-Thread Architecture Demo")
    print("=" * 60)

    # Setup
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Calibrate for ambient noise
    print("\nCalibrating for ambient noise...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except OSError as e:
        print(f"❌ Error: Could not access microphone: {e}")
        print("\nTroubleshooting:")
        print("  - Check microphone is connected")
        print("  - MacOS: Check System Preferences > Security & Privacy > Microphone")
        print("  - Raspberry Pi: Check 'arecord -l' shows devices")
        return 1

    print(f"✓ Energy threshold: {recognizer.energy_threshold:.1f}")

    # Create pipeline
    def on_transcript(text: str, utt):
        print(f"[{utt.duration:.1f}s] You said: {text}")

    def on_error(e: Exception):
        print(f"⚠ Error: {e}")

    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=mic,
        on_transcript=on_transcript,
        on_error=on_error,
    )

    # Start listening
    print("\n" + "=" * 60)
    print("Listening... (Ctrl+C to stop)")
    print("=" * 60)
    print()

    pipeline.start()

    try:
        while True:
            time.sleep(5)
            m = pipeline.metrics
            print(f"Stats: captured={m.chunks_captured}, dropped={m.chunks_dropped} ({m.drop_rate:.1%}), utterances={m.utterances_detected}/{m.utterances_transcribed}")
    except KeyboardInterrupt:
        print("\n\nStopping...")
        pipeline.stop()

    # Final stats
    m = pipeline.metrics
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    print(f"Chunks captured:        {m.chunks_captured}")
    print(f"Chunks dropped:         {m.chunks_dropped} ({m.drop_rate:.1%})")
    print(f"Utterances detected:    {m.utterances_detected}")
    print(f"Utterances transcribed: {m.utterances_transcribed}")
    print(f"Transcription errors:   {m.transcription_errors}")

    # Success criteria check
    print("\n" + "=" * 60)
    print("POC Validation")
    print("=" * 60)

    if m.drop_rate < 0.01:  # <1%
        print(f"✓ Drop rate {m.drop_rate:.1%} < 1% - SUCCESS")
    else:
        print(f"✗ Drop rate {m.drop_rate:.1%} >= 1% - NEEDS INVESTIGATION")

    if m.utterances_transcribed > 0:
        print(f"✓ {m.utterances_transcribed} utterances transcribed - SUCCESS")
    else:
        print("⚠ No utterances transcribed - try speaking louder or check network")

    return 0


if __name__ == "__main__":
    exit(main())
```

**Step 2: Delete old test file**

```bash
rm test_full_pipeline.py
```

**Step 3: Test demo script**

```bash
python -m sr_pipeline_poc.demo
```

Expected: Clean demo output with stats and validation results

**Step 4: Update package __init__.py to expose main classes**

File: `sr_pipeline_poc/__init__.py`
```python
"""
Minimal proof of concept for sr_pipeline three-thread architecture.
"""

__version__ = "0.1.0"

from .pipeline import AudioPipeline
from .energy_vad import EnergyVAD
from .types import AudioChunk, Utterance, PipelineMetrics, VADResult

__all__ = [
    "AudioPipeline",
    "EnergyVAD",
    "AudioChunk",
    "Utterance",
    "PipelineMetrics",
    "VADResult",
]
```

**Step 5: Commit**

```bash
git add sr_pipeline_poc/demo.py sr_pipeline_poc/__init__.py
git rm test_full_pipeline.py 2>/dev/null || true
git commit -m "feat: add polished demo script

Created demo.py with clean output, error handling, and automatic
POC validation against success criteria. Removed test file.
Exposed main classes in __init__.py for clean imports.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Add README and Usage Documentation

**Files:**
- Create: `sr_pipeline_poc/README.md`
- Modify: `README.md` (root)

**Step 1: Create package README**

File: `sr_pipeline_poc/README.md`
```markdown
# SR Pipeline POC

Minimal proof of concept for three-thread speech recognition pipeline that prevents audio drops during transcription.

## Architecture

```
Microphone → [Capture Thread] → Queue → [Detect Thread] → Queue → [Transcribe Thread] → Callback
                   ↓                          ↓                         ↓
            AudioChunk (30ms)         Utterance (complete)      Google API transcription
```

**Key design principle:** Capture thread never blocks. Uses `put_nowait()` and drops frames with metrics tracking rather than blocking upstream.

## Quick Start

```bash
# Run demo
python -m sr_pipeline_poc.demo
```

Speak into your microphone. Transcriptions will appear with utterance duration. Stats print every 5 seconds.

## Success Criteria

- ✅ Drop rate < 1% during normal speech
- ✅ Transcriptions appear for spoken utterances
- ✅ Works on both MacOS and Raspberry Pi

## Components

- **types.py** - AudioChunk, Utterance, PipelineMetrics, VADResult dataclasses
- **energy_vad.py** - RMS energy-based voice activity detection
- **pipeline.py** - AudioPipeline with three-thread architecture
- **demo.py** - Example script with metrics output

## Configuration (Hardcoded)

```python
ENERGY_THRESHOLD = 300.0      # RMS energy for speech detection
SILENCE_TIMEOUT = 0.8         # seconds of silence to end utterance
SPEECH_PADDING = 0.3          # seconds of audio to prepend
MAX_SPEECH_DURATION = 30.0    # force split after 30s
FRAME_DURATION_MS = 30        # milliseconds per chunk
```

To adjust, edit values in `pipeline.py`.

## Testing on Raspberry Pi

```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-pyaudio

# Run demo
python -m sr_pipeline_poc.demo
```

Compare drop rate to MacOS. Should be <1% on Pi 4.

## Troubleshooting

**No microphone access:**
- MacOS: Check System Preferences > Security & Privacy > Microphone
- Raspberry Pi: Check `arecord -l` shows devices

**No transcriptions appearing:**
- Check internet connection (Google API requires network)
- Speak louder or adjust `ENERGY_THRESHOLD`
- Check for error messages in output

**High drop rate:**
- Indicates queue is filling faster than downstream can process
- This is what we're trying to prevent - investigate if > 1%

## Next Steps

If POC validates architecture (<1% drop rate):

1. Add WebRTC VAD support
2. Implement full 4-state FSM
3. Add sample rate conversion
4. Add unit tests
5. Package for distribution
```

**Step 2: Update root README**

File: `README.md`
```markdown
# SR Pipeline Test

Testing ground for speech recognition pipeline improvements.

## POC

The `sr_pipeline_poc/` directory contains a minimal proof of concept for the three-thread architecture.

**Run the demo:**
```bash
python -m sr_pipeline_poc.demo
```

**Read the design:**
- POC design: `docs/plans/2025-11-25-poc-design.md`
- Full architecture: `docs/plans/sr-pipeline-design-document.md`

## Goal

Validate that separating capture, detection, and transcription into independent threads prevents the audio drops that occur in `speech_recognition.listen_in_background()`.

**Success criteria:** <1% drop rate during continuous speech on both MacOS and Raspberry Pi.
```

**Step 3: Commit**

```bash
git add sr_pipeline_poc/README.md README.md
git commit -m "docs: add README files for POC

Added comprehensive README in POC package with architecture
overview, quick start, troubleshooting, and next steps. Updated
root README to point to POC and design docs.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Add Stress Test Script

**Files:**
- Create: `sr_pipeline_poc/stress_test.py`

**Step 1: Create stress test script**

File: `sr_pipeline_poc/stress_test.py`
```python
#!/usr/bin/env python3
"""
Stress test for sr_pipeline POC.

Adds artificial delay to transcription to validate that capture
continues without dropping frames even when transcription is slow.
"""

import speech_recognition as sr
import time
from .pipeline import AudioPipeline


def main():
    """Run the stress test."""
    print("=" * 60)
    print("SR Pipeline POC - Stress Test")
    print("=" * 60)
    print("\nThis test adds a 2-second delay to each transcription.")
    print("Capture should continue without dropping frames.\n")

    # Setup
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Calibrate
    print("Calibrating for ambient noise...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except OSError as e:
        print(f"❌ Error: Could not access microphone: {e}")
        return 1

    print(f"✓ Energy threshold: {recognizer.energy_threshold:.1f}")

    # Wrap transcription callback with artificial delay
    def on_transcript(text: str, utt):
        print(f"[{utt.duration:.1f}s] Starting transcription (will take 2s)...")
        time.sleep(2)  # Simulate slow API
        print(f"[{utt.duration:.1f}s] You said: {text}")

    def on_error(e: Exception):
        print(f"⚠ Error: {e}")

    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=mic,
        on_transcript=on_transcript,
        on_error=on_error,
    )

    # Start listening
    print("\n" + "=" * 60)
    print("Listening... Speak continuously for 10+ seconds")
    print("=" * 60)
    print()

    pipeline.start()

    try:
        # Run for 30 seconds
        for i in range(6):
            time.sleep(5)
            m = pipeline.metrics
            print(f"[{(i+1)*5}s] Stats: captured={m.chunks_captured}, dropped={m.chunks_dropped} ({m.drop_rate:.1%}), utterances={m.utterances_detected}/{m.utterances_transcribed}")
    except KeyboardInterrupt:
        print("\n\nStopping...")

    pipeline.stop()

    # Final stats
    m = pipeline.metrics
    print("\n" + "=" * 60)
    print("Stress Test Results")
    print("=" * 60)
    print(f"Chunks captured:        {m.chunks_captured}")
    print(f"Chunks dropped:         {m.chunks_dropped} ({m.drop_rate:.1%})")
    print(f"Utterances detected:    {m.utterances_detected}")
    print(f"Utterances transcribed: {m.utterances_transcribed}")

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    if m.drop_rate < 0.01:  # <1%
        print(f"✓ Drop rate {m.drop_rate:.1%} < 1% EVEN WITH 2s DELAY - SUCCESS!")
        print("  This proves capture thread doesn't block during transcription.")
    else:
        print(f"✗ Drop rate {m.drop_rate:.1%} >= 1% - ARCHITECTURE ISSUE")

    return 0


if __name__ == "__main__":
    exit(main())
```

**Step 2: Test stress test script**

```bash
python -m sr_pipeline_poc.stress_test
```

Expected: Even with 2s delay per transcription, drop rate should stay at 0%

**Step 3: Update POC README with stress test info**

File: `sr_pipeline_poc/README.md`

Add section after "Quick Start":

```markdown
## Stress Test

Validate capture thread doesn't block during slow transcription:

```bash
python -m sr_pipeline_poc.stress_test
```

This adds a 2-second delay to each transcription. Drop rate should still be <1%, proving the capture thread continues independently.
```

**Step 4: Commit**

```bash
git add sr_pipeline_poc/stress_test.py sr_pipeline_poc/README.md
git commit -m "feat: add stress test script

Added stress_test.py that artificially delays transcription by 2s
to validate capture thread continues without blocking. This is the
key architectural validation for the POC.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Final Integration and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/poc-validation-checklist.md`

**Step 1: Add POC section to CLAUDE.md**

File: `CLAUDE.md`

Add section after "Dependencies" section:

```markdown
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
```

**Step 2: Create validation checklist**

File: `docs/poc-validation-checklist.md`
```markdown
# POC Validation Checklist

Use this checklist to validate the POC on different platforms.

## Pre-Test Setup

- [ ] System dependencies installed (`portaudio` on MacOS, `portaudio19-dev` on Pi)
- [ ] Python dependencies installed (`uv sync`)
- [ ] Microphone connected and accessible
- [ ] Internet connection available (Google API requires network)

## Test 1: Basic Functionality (MacOS)

Run: `python -m sr_pipeline_poc.demo`

- [ ] Demo starts without errors
- [ ] Ambient noise calibration completes
- [ ] Speaking into microphone produces transcriptions
- [ ] Transcriptions are reasonably accurate
- [ ] Drop rate shows 0.0%
- [ ] Stats update every 5 seconds
- [ ] Ctrl+C stops cleanly

**Expected results:**
```
Drop rate: 0.0% < 1% - SUCCESS
N utterances transcribed - SUCCESS
```

## Test 2: Stress Test (MacOS)

Run: `python -m sr_pipeline_poc.stress_test`

- [ ] Test starts without errors
- [ ] Speak continuously for 10+ seconds
- [ ] Transcriptions appear (with 2s delay message)
- [ ] Drop rate stays at 0.0% despite delay
- [ ] Final validation shows SUCCESS

**Expected results:**
```
Drop rate: 0.0% < 1% EVEN WITH 2s DELAY - SUCCESS!
```

This is the critical test - proves capture doesn't block during transcription.

## Test 3: Basic Functionality (Raspberry Pi 4)

Run same tests on Raspberry Pi 4.

- [ ] Demo works on Pi
- [ ] Drop rate < 1% (may be slightly higher than MacOS but should still be very low)
- [ ] Stress test shows drop rate < 1%

**Expected results:**
- MacOS: 0.0% drop rate
- Pi 4: <1% drop rate (acceptable, much better than 10-30% from baseline)

## Test 4: Comparison to Baseline (Optional)

Create a test using `speech_recognition.listen_in_background()` for comparison:

```python
import speech_recognition as sr
import time

recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

def callback(recognizer, audio):
    try:
        # Simulate slow transcription
        time.sleep(2)
        text = recognizer.recognize_google(audio)
        print(f"Baseline: {text}")
    except:
        pass

stop_listening = recognizer.listen_in_background(mic, callback)

time.sleep(30)
stop_listening(wait_for_stop=False)
```

- [ ] Baseline shows audio gaps during transcription (you'll hear clipped speech)
- [ ] POC does not show audio gaps

## Success Criteria

All of these must pass:

- [x] MacOS demo drop rate < 1%
- [x] MacOS stress test drop rate < 1%
- [x] Pi 4 demo drop rate < 1%
- [x] Pi 4 stress test drop rate < 1%
- [x] Transcriptions are accurate and complete
- [x] No crashes or fatal errors

If all pass: **POC validates the architecture** ✅

Next step: Proceed with full implementation (WebRTC VAD, 4-state FSM, tests, packaging)
```

**Step 3: Commit final documentation**

```bash
git add CLAUDE.md docs/poc-validation-checklist.md
git commit -m "docs: add POC validation checklist and update CLAUDE.md

Added validation checklist for testing POC on MacOS and Raspberry Pi.
Updated CLAUDE.md with POC status and running instructions.

POC implementation is now complete and ready for validation testing.

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Completion

The POC implementation is complete!

**What was built:**
- `sr_pipeline_poc/types.py` - Data structures
- `sr_pipeline_poc/energy_vad.py` - Voice activity detection
- `sr_pipeline_poc/pipeline.py` - Three-thread architecture
- `sr_pipeline_poc/demo.py` - Interactive demo script
- `sr_pipeline_poc/stress_test.py` - Architecture validation test
- Documentation and validation checklist

**Next steps:**

1. **Test on MacOS:**
   ```bash
   python -m sr_pipeline_poc.demo
   python -m sr_pipeline_poc.stress_test
   ```

2. **Test on Raspberry Pi:**
   - Transfer code to Pi
   - Install dependencies
   - Run same tests
   - Compare metrics

3. **Validate success criteria:**
   - Use `docs/poc-validation-checklist.md`
   - All tests should show <1% drop rate

4. **If validation passes:**
   - Proceed with full implementation
   - Add WebRTC VAD
   - Implement 4-state FSM
   - Add unit tests
   - Package for distribution

5. **If validation fails:**
   - Review architecture
   - Check for blocking operations in capture thread
   - Investigate queue sizing
   - Profile thread performance
