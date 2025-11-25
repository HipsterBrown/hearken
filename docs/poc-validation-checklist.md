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

- [ ] MacOS demo drop rate < 1%
- [ ] MacOS stress test drop rate < 1%
- [ ] Pi 4 demo drop rate < 1%
- [ ] Pi 4 stress test drop rate < 1%
- [ ] Transcriptions are accurate and complete
- [ ] No crashes or fatal errors

If all pass: **POC validates the architecture** ✅

Next step: Proceed with full implementation (WebRTC VAD, 4-state FSM, tests, packaging)
