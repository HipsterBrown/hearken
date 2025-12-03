# Manual Testing Checklist for WebRTC VAD

**Feature:** WebRTC VAD Support (v0.2)
**Date:** 2025-12-02
**Status:** Ready for Manual Testing

## Prerequisites

- [ ] Hardware microphone available
- [ ] Internet connection (for Google Speech Recognition API)
- [ ] Installed: `pip install hearken[webrtc,sr]`

## Test 1: Basic Import

**Objective:** Verify WebRTCVAD can be imported and instantiated.

**Steps:**
```bash
cd ~/.config/superpowers/worktrees/hearken/feature-webrtc-vad
python -c "from hearken import WebRTCVAD; vad = WebRTCVAD(); print('✓ Import successful')"
```

**Expected Result:**
- No ImportError
- Prints "✓ Import successful"

**Status:** ✅ PASSED (automated)

---

## Test 2: Run WebRTC VAD Example

**Objective:** Verify example runs without errors and detects speech.

**Steps:**
```bash
cd ~/.config/superpowers/worktrees/hearken/feature-webrtc-vad
python examples/webrtc_vad.py
```

**Expected Results:**
- [ ] Script starts without errors
- [ ] Shows "Calibrating for ambient noise..."
- [ ] Shows "Listening with WebRTC VAD..."
- [ ] Displays aggressiveness mode and sample rate

**When speaking:**
- [ ] Speech is detected
- [ ] Transcription appears (if internet available)
- [ ] Format: `[X.Xs] You said: <your words>`

**When silent:**
- [ ] No false detections during quiet periods

**Cleanup:**
- [ ] Ctrl+C stops cleanly
- [ ] No error messages on shutdown

**Notes:**
```
Date tested: _______
Tester: _______
Microphone: _______
OS: _______

Issues encountered:
```

---

## Test 3: Aggressiveness Modes Comparison

**Objective:** Verify different aggressiveness modes behave correctly.

**Test 3a: Mode 0 (Least Aggressive)**

Edit `examples/webrtc_vad.py` line 25:
```python
vad = WebRTCVAD(aggressiveness=0)  # Least aggressive
```

Run and test:
- [ ] More sensitive to background noise
- [ ] Picks up quieter speech
- [ ] May have more false positives

**Test 3b: Mode 1 (Quality - Default)**

Edit line 25:
```python
vad = WebRTCVAD(aggressiveness=1)  # Quality mode (default)
```

Run and test:
- [ ] Balanced sensitivity
- [ ] Good speech detection
- [ ] Reasonable noise rejection

**Test 3c: Mode 2 (Low Bitrate)**

Edit line 25:
```python
vad = WebRTCVAD(aggressiveness=2)  # Low bitrate mode
```

Run and test:
- [ ] More conservative
- [ ] Requires clearer speech
- [ ] Better noise rejection

**Test 3d: Mode 3 (Very Aggressive)**

Edit line 25:
```python
vad = WebRTCVAD(aggressiveness=3)  # Very aggressive
```

Run and test:
- [ ] Most conservative
- [ ] May miss softer speech
- [ ] Best noise rejection

**Notes:**
```
Mode 0 behavior: _______
Mode 1 behavior: _______
Mode 2 behavior: _______
Mode 3 behavior: _______

Recommended default: _______
```

---

## Test 4: Error Messages - Unsupported Sample Rate

**Objective:** Verify clear error message for unsupported sample rate.

**Steps:**

1. Create test file `test_error_44khz.py`:
```python
import speech_recognition as sr
from hearken import Listener, WebRTCVAD
from hearken.adapters.sr import SpeechRecognitionSource

recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=44100)  # Unsupported rate

source = SpeechRecognitionSource(mic)
vad = WebRTCVAD()
listener = Listener(source=source, vad=vad)

try:
    listener.start()
    import time
    time.sleep(2)
except Exception as e:
    print(f"Error caught: {e}")
finally:
    listener.stop()
```

2. Run:
```bash
python test_error_44khz.py
```

**Expected Result:**
- [ ] ValueError raised
- [ ] Message includes: "WebRTC VAD requires sample rate of [8000, 16000, 32000, 48000] Hz"
- [ ] Message shows actual rate: "Got 44100 Hz"
- [ ] Message includes guidance: "Configure your AudioSource with a supported sample rate"

**Actual Output:**
```
[paste error message here]
```

**Status:** [ ] PASS / [ ] FAIL

**Notes:**
```
Error message clarity: _______
Actionable guidance: _______
```

---

## Test 5: Error Messages - Unsupported Frame Duration

**Objective:** Verify clear error message for unsupported frame duration.

**Note:** This is harder to test with speech_recognition library which uses fixed frame sizes. Document if encountered during regular use.

**Expected Behavior:**
- ValueError with clear message
- Lists supported durations: [10, 20, 30] ms
- Shows actual duration received
- Provides configuration guidance

**Status:** [ ] Tested / [ ] Skipped (hard to trigger)

---

## Test 6: Comparison with EnergyVAD

**Objective:** Compare WebRTC VAD accuracy vs EnergyVAD in noisy environment.

**Test Setup:**
- Use same audio environment for both tests
- Introduce background noise (music, fan, typing, etc.)

**Test 6a: EnergyVAD Baseline**

```python
from hearken import EnergyVAD
vad = EnergyVAD(dynamic=True)
# ... rest of setup same as example
```

Run and observe:
- [ ] False positives from background noise
- [ ] Missed speech during noise
- [ ] Overall accuracy: _______

**Test 6b: WebRTCVAD Comparison**

```python
from hearken import WebRTCVAD
vad = WebRTCVAD(aggressiveness=1)
# ... rest of setup same as example
```

Run and observe:
- [ ] Better noise rejection than EnergyVAD
- [ ] Fewer false positives
- [ ] More accurate speech detection
- [ ] Overall accuracy: _______

**Comparison Notes:**
```
Noise environment: _______
EnergyVAD performance: _______
WebRTCVAD performance: _______
Improvement: _______
```

---

## Test 7: Long Running Stability

**Objective:** Verify WebRTC VAD remains stable over extended use.

**Steps:**
1. Run `examples/webrtc_vad.py`
2. Let it run for 5-10 minutes
3. Speak intermittently throughout
4. Monitor for issues

**Expected Results:**
- [ ] No memory leaks
- [ ] No degraded performance over time
- [ ] Clean transcriptions throughout
- [ ] No crashes or hangs

**Monitor:**
- [ ] CPU usage stable
- [ ] Memory usage stable
- [ ] Response time consistent

**Notes:**
```
Duration: _______ minutes
CPU usage: _______
Memory usage: _______
Issues: _______
```

---

## Test 8: Reset Functionality

**Objective:** Verify reset() works correctly in practice.

This is already tested in automated tests, but verify in real usage:

**Expected Behavior:**
- VAD can be reset between utterances
- No state pollution between resets
- Performance consistent after reset

**Status:** ✅ PASSED (automated tests cover this)

---

## Summary Checklist

After completing all manual tests above:

- [ ] All imports work
- [ ] Example runs successfully
- [ ] Speech detection works correctly
- [ ] All aggressiveness modes tested
- [ ] Error messages are clear and helpful
- [ ] WebRTC VAD shows improvement over EnergyVAD
- [ ] Long running stability confirmed
- [ ] No major issues found

**Ready for release:** [ ] YES / [ ] NO

**Issues to address before release:**
```
1. _______
2. _______
3. _______
```

**Final Notes:**
```
Tested by: _______
Date: _______
OS: _______
Python version: _______
Microphone: _______

Overall assessment:
```

---

## Post-Testing

After manual testing is complete and issues are addressed:

1. Document any findings in GitHub issues (if applicable)
2. Update this checklist with results
3. Proceed with Task 13: Final integration and version bump
4. Consider adding any discovered edge cases to automated tests

**Manual Testing Documentation Updated:** [ ] YES / [ ] NO
