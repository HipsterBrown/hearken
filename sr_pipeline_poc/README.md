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

## Stress Test

Validate capture thread doesn't block during slow transcription:

```bash
python -m sr_pipeline_poc.stress_test
```

This adds a 2-second delay to each transcription. Drop rate should still be <1%, proving the capture thread continues independently.

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
