# Python speech_recognition Pipeline Improvements

**Module:** `sr_pipeline` (userland wrapper for speech_recognition)  
**Author:** Nick Hehr  
**Date:** November 2025  
**Status:** Draft  
**Version:** 1.0  

---

## 1. Executive Summary

This document describes architectural improvements to work around limitations in Python's `speech_recognition` library. The core issue is that `listen_in_background()` uses a synchronous architecture where audio capture blocks during transcription, causing dropped frames.

The solution is a userland wrapper that decouples capture, detection, and transcription into independent threads with queue-based communication. This reuses existing `speech_recognition` components (`Microphone`, `Recognizer`, recognition backends) while fixing the pipeline architecture.

---

## 2. Problem Analysis

### 2.1 Current Architecture (speech_recognition)

```python
def listen_in_background(self, source, callback, phrase_time_limit=None):
    def threaded_listen():
        with source as s:
            while self.running:
                audio = self.listen(s, phrase_time_limit)  # BLOCKS
                callback(self, audio)                       # BLOCKS
```

**Issues:**

1. **Serial execution**: While `callback()` runs (often doing synchronous transcription), nothing captures audio
2. **Coupled VAD and capture**: `listen()` combines energy detection with audio reading
3. **No backpressure**: If processing is slow, audio is simply lost
4. **Fixed VAD**: Only energy-based detection, no option for WebRTC or neural VAD

### 2.2 Why the GIL Isn't the Main Problem

The GIL is often blamed, but:

- **PyAudio releases the GIL** during blocking device reads (C extension)
- **Network I/O releases the GIL** during socket operations
- **The actual Python work** (energy calculation) is trivial (~1ms per chunk)

The real problem is architectural—synchronous pipeline stages that should be concurrent.

### 2.3 Where the GIL Does Matter

CPU-bound work like neural network inference (Silero VAD) would be GIL-bound. Solutions:

- Use C extensions that release the GIL (WebRTC VAD, ONNX with proper config)
- Move CPU-bound work to a separate process via `multiprocessing`

---

## 3. Proposed Architecture

### 3.1 Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Capture   │───▶│     VAD     │───▶│ Transcribe  │───▶│  Callback   │
│   Thread    │    │   Thread    │    │   Thread    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
 capture_queue      utterance_queue    User function
  (AudioChunk)        (Utterance)         (text)
   maxsize=100        maxsize=10
```

**Key properties:**

- Each stage runs in its own thread
- Queues decouple stages with explicit backpressure
- Capture thread never blocks on downstream—drops oldest if queue full
- Transcription latency doesn't affect audio capture

### 3.2 Data Flow

1. **Capture Thread**: Reads 30ms audio chunks from microphone, pushes to `capture_queue`
2. **VAD Thread**: Consumes chunks, runs voice activity detection + FSM segmentation, emits complete utterances to `utterance_queue`
3. **Transcribe Thread**: Consumes utterances, calls recognition backend, invokes user callback with text

### 3.3 Backpressure Strategy

| Queue | Full Behavior | Rationale |
|-------|---------------|-----------|
| `capture_queue` | Drop newest chunk | Capture must never block; old audio more valuable than new during overflow |
| `utterance_queue` | Drop utterance with warning | Transcription backlog indicates systemic issue |

---

## 4. Core Components

### 4.1 Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
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
        return self.end_time - self.start_time


@dataclass
class TranscriptResult:
    """Result from transcription."""
    text: str
    utterance: Utterance
    confidence: Optional[float] = None  # If available from backend


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
        total = self.chunks_captured + self.chunks_dropped
        return self.chunks_dropped / total if total > 0 else 0.0
```

### 4.2 VAD Interface

```python
from abc import ABC, abstractmethod
from typing import Optional


class VADResult:
    """Result from voice activity detection."""
    def __init__(self, is_speech: bool, confidence: float = 1.0):
        self.is_speech = is_speech
        self.confidence = confidence  # 0.0 to 1.0


class VAD(ABC):
    """Voice Activity Detection interface."""
    
    @abstractmethod
    def process(self, chunk: AudioChunk) -> VADResult:
        """
        Process an audio chunk and determine if it contains speech.
        
        Args:
            chunk: Audio data to analyze
            
        Returns:
            VADResult with speech detection outcome
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state. Called between utterances."""
        pass
    
    @property
    @abstractmethod
    def required_sample_rate(self) -> Optional[int]:
        """
        Return required sample rate, or None if any rate is acceptable.
        WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz.
        """
        pass
    
    @property
    @abstractmethod
    def required_frame_duration_ms(self) -> Optional[int]:
        """
        Return required frame duration in ms, or None if flexible.
        WebRTC VAD requires 10, 20, or 30ms frames.
        """
        pass
```

### 4.3 Detector State Machine

```python
class DetectorState(Enum):
    """FSM states for utterance segmentation."""
    IDLE = auto()            # Waiting for speech
    SPEECH_STARTING = auto() # Speech detected, confirming it's not noise
    SPEAKING = auto()        # Confirmed speech, accumulating audio
    TRAILING_SILENCE = auto() # Speech may have ended, waiting to confirm


@dataclass
class DetectorConfig:
    """Configuration for utterance detection."""
    
    # Energy threshold for fallback energy-based VAD
    energy_threshold: float = 300.0
    
    # Minimum speech duration to consider valid (filters transients)
    min_speech_duration: float = 0.25  # seconds
    
    # Maximum speech duration before forced segmentation
    max_speech_duration: float = 30.0  # seconds
    
    # Silence duration to end an utterance
    silence_timeout: float = 0.8  # seconds
    
    # Audio to prepend before detected speech start
    speech_padding: float = 0.3  # seconds
    
    # No-input timeout (0 = disabled)
    no_input_timeout: float = 5.0  # seconds
```

---

## 5. VAD Implementations

### 5.1 Energy-Based VAD (Baseline)

Simple RMS energy threshold. Matches `speech_recognition`'s built-in behavior.

```python
import numpy as np


class EnergyVAD(VAD):
    """
    Simple energy-based voice activity detection.
    Matches speech_recognition's built-in approach.
    """
    
    def __init__(self, threshold: float = 300.0, dynamic: bool = True):
        """
        Args:
            threshold: RMS energy threshold for speech detection
            dynamic: If True, adapt threshold based on ambient noise
        """
        self.threshold = threshold
        self.dynamic = dynamic
        self._ambient_energy: Optional[float] = None
        self._samples_seen = 0
    
    def process(self, chunk: AudioChunk) -> VADResult:
        # Convert bytes to samples
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
        # Don't reset ambient calibration
        pass
    
    @property
    def required_sample_rate(self) -> Optional[int]:
        return None  # Works with any sample rate
    
    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        return None  # Works with any frame size
```

### 5.2 WebRTC VAD

Google's production VAD from WebRTC. Releases the GIL during processing.

```python
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


class WebRTCVADMode(Enum):
    """WebRTC VAD aggressiveness modes."""
    QUALITY = 0          # Least aggressive, highest false positive rate
    LOW_BITRATE = 1      # 
    AGGRESSIVE = 2       #
    VERY_AGGRESSIVE = 3  # Most aggressive, lowest false positive rate


class WebRTCVAD(VAD):
    """
    WebRTC Voice Activity Detection.
    
    Uses Google's production VAD from the WebRTC project.
    Releases the GIL during processing (C extension).
    
    Requirements:
        pip install webrtcvad
        
    Constraints:
        - Sample rate must be 8000, 16000, 32000, or 48000 Hz
        - Frame duration must be 10, 20, or 30 ms
        - Audio must be 16-bit mono PCM
    """
    
    VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
    VALID_FRAME_DURATIONS = {10, 20, 30}
    
    def __init__(
        self, 
        mode: WebRTCVADMode = WebRTCVADMode.AGGRESSIVE,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ):
        if not WEBRTC_AVAILABLE:
            raise ImportError(
                "webrtcvad not installed. Install with: pip install webrtcvad"
            )
        
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                f"Sample rate must be one of {self.VALID_SAMPLE_RATES}, got {sample_rate}"
            )
        
        if frame_duration_ms not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f"Frame duration must be one of {self.VALID_FRAME_DURATIONS}ms, "
                f"got {frame_duration_ms}ms"
            )
        
        self._vad = webrtcvad.Vad(mode.value)
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # bytes
    
    def process(self, chunk: AudioChunk) -> VADResult:
        # Validate chunk format
        if chunk.sample_rate != self._sample_rate:
            raise ValueError(
                f"Chunk sample rate {chunk.sample_rate} doesn't match "
                f"configured rate {self._sample_rate}"
            )
        
        if len(chunk.data) != self._frame_size:
            raise ValueError(
                f"Chunk size {len(chunk.data)} doesn't match expected "
                f"frame size {self._frame_size} bytes"
            )
        
        # WebRTC VAD returns True/False - releases GIL during this call
        is_speech = self._vad.is_speech(chunk.data, self._sample_rate)
        
        # WebRTC doesn't provide confidence, use binary
        return VADResult(is_speech=is_speech, confidence=1.0 if is_speech else 0.0)
    
    def reset(self) -> None:
        # WebRTC VAD is stateless, nothing to reset
        pass
    
    @property
    def required_sample_rate(self) -> Optional[int]:
        return self._sample_rate
    
    @property
    def required_frame_duration_ms(self) -> Optional[int]:
        return self._frame_duration_ms


def create_vad(
    vad_type: str = "webrtc",
    **kwargs
) -> VAD:
    """
    Factory function to create VAD instances.
    
    Args:
        vad_type: "energy" or "webrtc"
        **kwargs: Passed to VAD constructor
        
    Returns:
        VAD instance
    """
    if vad_type == "energy":
        return EnergyVAD(**kwargs)
    elif vad_type == "webrtc":
        return WebRTCVAD(**kwargs)
    else:
        raise ValueError(f"Unknown VAD type: {vad_type}")
```

### 5.3 Future: Silero VAD

Silero VAD provides higher accuracy but requires ONNX Runtime and is CPU-bound. To avoid GIL issues, it should run in a separate process.

```python
# NOTE: Out of scope for initial implementation
# 
# Silero VAD would require:
# 1. multiprocessing.Process for inference
# 2. multiprocessing.Queue for communication
# 3. ONNX Runtime with proper threading config
#
# The VAD interface supports this - just implement process() to
# send chunks to the inference process and receive results.
```

---

## 6. Pipeline Implementation

### 6.1 Main Pipeline Class

```python
import threading
import queue
import time
import collections
from typing import Callable, Optional, Union
import speech_recognition as sr


# Type alias for callbacks
TranscriptCallback = Callable[[str, Utterance], None]
ErrorCallback = Callable[[Exception], None]


class AudioPipeline:
    """
    Decoupled audio capture and processing pipeline for speech_recognition.
    
    Separates capture, VAD/segmentation, and transcription into independent
    threads to prevent audio drops during transcription.
    
    Example:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        pipeline = AudioPipeline(
            recognizer=recognizer,
            source=mic,
            on_transcript=lambda text, utt: print(f"You said: {text}"),
            vad=WebRTCVAD(mode=WebRTCVADMode.AGGRESSIVE),
        )
        
        pipeline.start()
        try:
            pipeline.wait()
        except KeyboardInterrupt:
            pipeline.stop()
    """
    
    def __init__(
        self,
        recognizer: sr.Recognizer,
        source: sr.Microphone,
        on_transcript: TranscriptCallback,
        on_error: Optional[ErrorCallback] = None,
        # VAD configuration
        vad: Optional[VAD] = None,
        detector_config: Optional[DetectorConfig] = None,
        # Recognition configuration
        recognize_method: str = "recognize_google",
        recognize_kwargs: Optional[dict] = None,
        # Queue sizes
        capture_queue_size: int = 100,
        utterance_queue_size: int = 10,
    ):
        self.recognizer = recognizer
        self.source = source
        self.on_transcript = on_transcript
        self.on_error = on_error or (lambda e: print(f"Pipeline error: {e}"))
        
        # VAD setup
        self.vad = vad or EnergyVAD(threshold=recognizer.energy_threshold)
        self.detector_config = detector_config or DetectorConfig()
        
        # Recognition setup
        self._recognize_method = getattr(recognizer, recognize_method)
        self._recognize_kwargs = recognize_kwargs or {}
        
        # Queues
        self._capture_queue: queue.Queue[Optional[AudioChunk]] = queue.Queue(
            maxsize=capture_queue_size
        )
        self._utterance_queue: queue.Queue[Optional[Utterance]] = queue.Queue(
            maxsize=utterance_queue_size
        )
        
        # Control
        self._running = False
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        
        # Metrics
        self.metrics = PipelineMetrics()
    
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
        """
        Dedicated capture thread.
        
        Reads audio chunks at fixed intervals, never blocks on downstream.
        """
        # Determine frame size based on VAD requirements
        frame_duration_ms = self.vad.required_frame_duration_ms or 30
        
        with self.source as s:
            # Validate/adjust sample rate for VAD
            actual_rate = s.SAMPLE_RATE
            required_rate = self.vad.required_sample_rate
            
            if required_rate and actual_rate != required_rate:
                # In production, you'd resample. For now, warn.
                print(
                    f"Warning: Microphone rate {actual_rate} doesn't match "
                    f"VAD required rate {required_rate}. Detection may fail."
                )
            
            chunk_samples = int(actual_rate * frame_duration_ms / 1000)
            
            while self._running:
                try:
                    # Read audio - releases GIL during device read
                    data = s.stream.read(chunk_samples, exception_on_overflow=False)
                    
                    chunk = AudioChunk(
                        data=data,
                        timestamp=time.monotonic(),
                        sample_rate=actual_rate,
                        sample_width=s.SAMPLE_WIDTH,
                    )
                    
                    # Non-blocking put
                    try:
                        self._capture_queue.put_nowait(chunk)
                        self.metrics.chunks_captured += 1
                    except queue.Full:
                        self.metrics.chunks_dropped += 1
                        
                except OSError as e:
                    # Audio device error
                    if self._running:
                        self.on_error(e)
                    break
                except Exception as e:
                    if self._running:
                        self.on_error(e)
    
    def _detect_loop(self) -> None:
        """
        VAD and utterance segmentation thread.
        
        Implements FSM to segment continuous audio into discrete utterances.
        """
        config = self.detector_config
        state = DetectorState.IDLE
        
        # Ring buffer for speech padding
        frame_duration_ms = self.vad.required_frame_duration_ms or 30
        padding_frames = int(config.speech_padding * 1000 / frame_duration_ms)
        padding_buffer: collections.deque[AudioChunk] = collections.deque(
            maxlen=max(1, padding_frames)
        )
        
        # Utterance accumulator
        utterance_chunks: list[AudioChunk] = []
        speech_start_time: Optional[float] = None
        last_speech_time: Optional[float] = None
        last_activity_time = time.monotonic()
        
        while self._running:
            try:
                chunk = self._capture_queue.get(timeout=0.1)
            except queue.Empty:
                # Check no-input timeout
                if (
                    config.no_input_timeout > 0 
                    and state == DetectorState.IDLE
                    and time.monotonic() - last_activity_time > config.no_input_timeout
                ):
                    # Could emit a "no input" event here
                    last_activity_time = time.monotonic()
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
            if state == DetectorState.IDLE:
                padding_buffer.append(chunk)
                if is_speech:
                    state = DetectorState.SPEECH_STARTING
                    speech_start_time = now
                    last_speech_time = now
                    last_activity_time = now
                    # Include padding
                    utterance_chunks = list(padding_buffer)
                    
            elif state == DetectorState.SPEECH_STARTING:
                utterance_chunks.append(chunk)
                if is_speech:
                    last_speech_time = now
                    if now - speech_start_time >= config.min_speech_duration:
                        state = DetectorState.SPEAKING
                else:
                    if now - last_speech_time >= config.silence_timeout:
                        # False start, reset
                        state = DetectorState.IDLE
                        utterance_chunks = []
                        padding_buffer.clear()
                        self.vad.reset()
                        
            elif state == DetectorState.SPEAKING:
                utterance_chunks.append(chunk)
                if is_speech:
                    last_speech_time = now
                    
                # Check max duration
                if now - speech_start_time >= config.max_speech_duration:
                    self._emit_utterance(utterance_chunks, speech_start_time, now)
                    state = DetectorState.IDLE
                    utterance_chunks = []
                    padding_buffer.clear()
                    self.vad.reset()
                elif not is_speech:
                    state = DetectorState.TRAILING_SILENCE
                    
            elif state == DetectorState.TRAILING_SILENCE:
                utterance_chunks.append(chunk)
                if is_speech:
                    last_speech_time = now
                    state = DetectorState.SPEAKING
                elif now - last_speech_time >= config.silence_timeout:
                    self._emit_utterance(utterance_chunks, speech_start_time, now)
                    state = DetectorState.IDLE
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
        
        # Combine chunks
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
    
    def _transcribe_loop(self) -> None:
        """
        Transcription thread.
        
        Consumes utterances and calls recognition backend.
        Network I/O releases the GIL.
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
                text = self._recognize_method(
                    utterance.audio, 
                    **self._recognize_kwargs
                )
                
                self.metrics.utterances_transcribed += 1
                
                # Invoke callback
                try:
                    self.on_transcript(text, utterance)
                except Exception as e:
                    self.on_error(e)
                    
            except sr.UnknownValueError:
                # No speech recognized - not an error
                pass
            except sr.RequestError as e:
                self.metrics.transcription_errors += 1
                self.on_error(e)
            except Exception as e:
                self.metrics.transcription_errors += 1
                self.on_error(e)
```

### 6.2 Convenience Functions

```python
def listen_in_background_improved(
    recognizer: sr.Recognizer,
    source: sr.Microphone,
    callback: Callable[[sr.Recognizer, str], None],
    vad_type: str = "webrtc",
    **kwargs
) -> AudioPipeline:
    """
    Drop-in improvement for speech_recognition.Recognizer.listen_in_background().
    
    Args:
        recognizer: speech_recognition Recognizer instance
        source: speech_recognition Microphone instance  
        callback: Function called with (recognizer, transcript_text)
        vad_type: "energy" or "webrtc"
        **kwargs: Additional arguments for AudioPipeline
        
    Returns:
        AudioPipeline instance (call .stop() to end)
        
    Example:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        # Adjust for ambient noise
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
        
        # Start listening
        pipeline = listen_in_background_improved(
            recognizer, mic,
            callback=lambda r, text: print(f"You said: {text}"),
            vad_type="webrtc",
        )
        
        # ... do other work ...
        
        pipeline.stop()
    """
    # Create VAD
    if vad_type == "energy":
        vad = EnergyVAD(threshold=recognizer.energy_threshold)
    elif vad_type == "webrtc":
        vad = WebRTCVAD(mode=WebRTCVADMode.AGGRESSIVE)
    else:
        raise ValueError(f"Unknown vad_type: {vad_type}")
    
    # Wrap callback to match expected signature
    def on_transcript(text: str, utterance: Utterance) -> None:
        callback(recognizer, text)
    
    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=source,
        on_transcript=on_transcript,
        vad=vad,
        **kwargs,
    )
    
    pipeline.start()
    return pipeline
```

---

## 7. Audio Preprocessing

### 7.1 Sample Rate Conversion

WebRTC VAD requires specific sample rates. If the microphone doesn't match, resample.

```python
import numpy as np
from scipy import signal  # Optional dependency


def resample_audio(
    data: bytes,
    from_rate: int,
    to_rate: int,
    sample_width: int = 2,
) -> bytes:
    """
    Resample audio data to a different sample rate.
    
    Args:
        data: Raw audio bytes (16-bit PCM)
        from_rate: Source sample rate
        to_rate: Target sample rate
        sample_width: Bytes per sample (2 for 16-bit)
        
    Returns:
        Resampled audio bytes
    """
    if from_rate == to_rate:
        return data
    
    # Convert to numpy
    samples = np.frombuffer(data, dtype=np.int16)
    
    # Calculate resampling ratio
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    
    # Resample using scipy (high quality) or linear interpolation (fallback)
    try:
        resampled = signal.resample(samples, new_length)
    except ImportError:
        # Fallback to linear interpolation
        x_old = np.linspace(0, 1, len(samples))
        x_new = np.linspace(0, 1, new_length)
        resampled = np.interp(x_new, x_old, samples)
    
    # Convert back to bytes
    return resampled.astype(np.int16).tobytes()


class ResamplingAudioSource:
    """
    Wrapper that resamples audio from a Microphone to a target rate.
    
    Use when the microphone sample rate doesn't match VAD requirements.
    """
    
    def __init__(self, source: sr.Microphone, target_rate: int):
        self.source = source
        self.target_rate = target_rate
        self._stream = None
        
    def __enter__(self):
        self.source.__enter__()
        self.SAMPLE_RATE = self.target_rate
        self.SAMPLE_WIDTH = self.source.SAMPLE_WIDTH
        self._source_rate = self.source.SAMPLE_RATE
        return self
    
    def __exit__(self, *args):
        return self.source.__exit__(*args)
    
    @property
    def stream(self):
        """Return a wrapper that resamples on read."""
        return _ResamplingStream(
            self.source.stream,
            self._source_rate,
            self.target_rate,
        )


class _ResamplingStream:
    """Internal stream wrapper that resamples audio."""
    
    def __init__(self, stream, from_rate: int, to_rate: int):
        self._stream = stream
        self._from_rate = from_rate
        self._to_rate = to_rate
        self._ratio = from_rate / to_rate
    
    def read(self, num_frames: int, exception_on_overflow: bool = False) -> bytes:
        # Read more frames from source to get equivalent duration
        source_frames = int(num_frames * self._ratio)
        data = self._stream.read(source_frames, exception_on_overflow)
        return resample_audio(data, self._from_rate, self._to_rate)
```

---

## 8. Usage Examples

### 8.1 Basic Usage

```python
import speech_recognition as sr
from sr_pipeline import AudioPipeline, WebRTCVAD, WebRTCVADMode

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Calibrate for ambient noise
    print("Calibrating for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"Energy threshold: {recognizer.energy_threshold}")
    
    # Create pipeline with WebRTC VAD
    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=mic,
        on_transcript=lambda text, utt: print(f"[{utt.duration:.1f}s] {text}"),
        on_error=lambda e: print(f"Error: {e}"),
        vad=WebRTCVAD(mode=WebRTCVADMode.AGGRESSIVE),
    )
    
    print("Listening... (Ctrl+C to stop)")
    pipeline.start()
    
    try:
        while True:
            time.sleep(5)
            m = pipeline.metrics
            print(
                f"Stats: captured={m.chunks_captured}, "
                f"dropped={m.chunks_dropped} ({m.drop_rate:.1%}), "
                f"utterances={m.utterances_detected}/{m.utterances_transcribed}"
            )
    except KeyboardInterrupt:
        print("\nStopping...")
        pipeline.stop()


if __name__ == "__main__":
    main()
```

### 8.2 Google Cloud Speech-to-Text

```python
import speech_recognition as sr
from sr_pipeline import AudioPipeline, WebRTCVAD

def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    pipeline = AudioPipeline(
        recognizer=recognizer,
        source=mic,
        on_transcript=lambda text, utt: print(f"You said: {text}"),
        vad=WebRTCVAD(),
        # Use Google Cloud instead of free API
        recognize_method="recognize_google_cloud",
        recognize_kwargs={
            "credentials_json": "/path/to/credentials.json",
            "language": "en-US",
            "preferred_phrases": ["turn on", "turn off", "hey assistant"],
        },
    )
    
    pipeline.start()
    pipeline.wait()
```

### 8.3 Drop-in Replacement

```python
import speech_recognition as sr
from sr_pipeline import listen_in_background_improved

def callback(recognizer, text):
    print(f"Recognized: {text}")

recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)

# Instead of: recognizer.listen_in_background(mic, callback)
pipeline = listen_in_background_improved(
    recognizer, mic, callback,
    vad_type="webrtc",
)

# Later...
pipeline.stop()
```

### 8.4 Custom Detection Config

```python
from sr_pipeline import AudioPipeline, WebRTCVAD, DetectorConfig

pipeline = AudioPipeline(
    recognizer=recognizer,
    source=mic,
    on_transcript=on_transcript,
    vad=WebRTCVAD(mode=WebRTCVADMode.VERY_AGGRESSIVE),
    detector_config=DetectorConfig(
        min_speech_duration=0.5,      # Require 500ms of speech to trigger
        max_speech_duration=60.0,     # Allow up to 60s utterances
        silence_timeout=1.0,          # Wait 1s of silence before ending
        speech_padding=0.5,           # Keep 500ms before speech start
        no_input_timeout=10.0,        # Timeout after 10s of no speech
    ),
)
```

---

## 9. Module Structure

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

---

## 10. Dependencies

### 10.1 Required

| Package | Version | Purpose |
|---------|---------|---------|
| `speech_recognition` | >=3.8 | Base library, recognition backends |
| `pyaudio` | >=0.2.11 | Audio capture (via speech_recognition) |
| `numpy` | >=1.20 | Audio processing, energy calculation |

### 10.2 Optional

| Package | Version | Purpose |
|---------|---------|---------|
| `webrtcvad` | >=2.0.10 | WebRTC voice activity detection |
| `scipy` | >=1.7 | High-quality resampling |

### 10.3 Installation

```bash
# Core
pip install SpeechRecognition numpy

# With WebRTC VAD (recommended)
pip install SpeechRecognition numpy webrtcvad

# With high-quality resampling
pip install SpeechRecognition numpy webrtcvad scipy
```

---

## 11. Performance Comparison

### 11.1 Test Methodology

Test on Raspberry Pi 4 (4GB) with USB microphone:
- 60 second continuous speech test
- Measure dropped frames, transcription latency, CPU usage

### 11.2 Expected Results

| Metric | listen_in_background() | AudioPipeline (Energy) | AudioPipeline (WebRTC) |
|--------|------------------------|------------------------|------------------------|
| Dropped frames | 10-30% | <1% | <1% |
| Transcription latency | 1-3s + drops | 1-3s (no drops) | 1-3s (no drops) |
| CPU usage | ~5% | ~6% | ~8% |
| Detection accuracy | Low | Low | High |

---

## 12. Limitations and Tradeoffs

### 12.1 What This Fixes

- Audio drops during transcription network I/O
- Missed utterance beginnings (via padding buffer)
- Poor VAD accuracy (via WebRTC VAD)
- Lack of metrics/observability

### 12.2 What This Doesn't Fix

- **Recognition accuracy**: Still depends on the backend (Google, etc.)
- **Network latency**: Transcription still takes 1-3s round-trip
- **GIL for CPU-bound VAD**: Silero VAD would need multiprocessing (out of scope)

### 12.3 When to Use Go Instead

Consider the Go implementation (`gostt`) when:

- You need Silero VAD (CPU-bound neural network)
- You want a single static binary for deployment
- You're building a long-running production service
- You need streaming recognition
- Target hardware is very constrained (Pi Zero)

---

## 13. Implementation Phases

### Phase 1: Core Pipeline (Week 1)

- [ ] Data classes (`AudioChunk`, `Utterance`, etc.)
- [ ] `EnergyVAD` implementation
- [ ] `AudioPipeline` with three-thread architecture
- [ ] Basic FSM detector
- [ ] Unit tests with mock audio

### Phase 2: WebRTC Integration (Week 2)

- [ ] `WebRTCVAD` implementation
- [ ] Sample rate validation/conversion
- [ ] Integration tests with real microphone
- [ ] `listen_in_background_improved()` compat wrapper

### Phase 3: Polish (Week 3)

- [ ] Metrics and observability
- [ ] Error handling edge cases
- [ ] Documentation and examples
- [ ] Performance benchmarks on Raspberry Pi

---

## 14. References

### Libraries

- [speech_recognition](https://github.com/Uberi/speech_recognition) - Base library
- [py-webrtcvad](https://github.com/wiseman/py-webrtcvad) - WebRTC VAD Python bindings

### Background

- [WebRTC VAD paper](https://www.researchgate.net/publication/255667085_A_Study_of_Voice_Activity_Detection_Techniques_for_NIST_Speaker_Recognition_Evaluations) - Technical details on WebRTC's VAD algorithm
- [Silero VAD](https://github.com/snakers4/silero-vad) - Reference for future neural VAD integration

---

*End of Design Document*
