# Silero VAD Manual Testing Checklist

## Prerequisites

- Microphone configured for 16kHz
- Internet connection (for first run, model download)
- Install: `pip install hearken[all]`

## Test Scenarios

### 1. Basic Functionality (Clean Environment)

**Setup:**
```bash
python examples/silero_vad.py
```

**Test:**
- [ ] Speak clearly in quiet room
- [ ] Verify speech is detected and transcribed
- [ ] Verify silence periods don't trigger detection
- [ ] Stop and restart, verify model not re-downloaded

**Expected:**
- Accurate speech detection
- Clear transcriptions
- No false positives during silence
- Fast startup on second run (cached model)

---

### 2. Noisy Environment Comparison

**Setup:**
Run with background noise (music, fan, etc.)

**Test:**
- [ ] Run with EnergyVAD: `python examples/basic_usage.py`
- [ ] Run with WebRTCVAD: `python examples/webrtc_vad.py`
- [ ] Run with SileroVAD: `python examples/silero_vad.py`
- [ ] Compare false positive rates

**Expected:**
- SileroVAD should have fewer false positives than EnergyVAD
- SileroVAD comparable or better than WebRTCVAD

---

### 3. Threshold Sensitivity

**Test different threshold values:**

**Low threshold (0.3) - More Sensitive:**
```python
vad=SileroVAD(threshold=0.3)
```
- [ ] Verify catches soft speech
- [ ] May have more false positives

**Default threshold (0.5) - Balanced:**
```python
vad=SileroVAD(threshold=0.5)
```
- [ ] Verify good balance
- [ ] Normal speech detected reliably

**High threshold (0.7) - Conservative:**
```python
vad=SileroVAD(threshold=0.7)
```
- [ ] Verify fewer false positives
- [ ] May miss very soft speech

---

### 4. Model Caching

**Test:**
- [ ] Delete cache: `rm ~/.cache/hearken/silero_vad_v5.onnx`
- [ ] Run example, observe model download
- [ ] Check file exists: `ls -lh ~/.cache/hearken/silero_vad_v5.onnx`
- [ ] Run again, verify no re-download (fast startup)

**Expected:**
- First run downloads ~2MB model
- Subsequent runs use cached model
- File size ~2MB

---

### 5. Custom Model Path

**Test:**
```python
# Download model manually
mkdir -p /tmp/custom_models
# Copy model to /tmp/custom_models/silero.onnx

vad=SileroVAD(model_path="/tmp/custom_models/silero.onnx")
```

- [ ] Verify uses custom path
- [ ] Verify works correctly

---

### 6. Environment Variable

**Test:**
```bash
export HEARKEN_SILERO_MODEL_PATH="/tmp/custom_models/silero.onnx"
python examples/silero_vad.py
```

- [ ] Verify uses env var path
- [ ] Verify works correctly

---

### 7. Error Messages

**Test wrong sample rate:**
```python
mic = sr.Microphone(sample_rate=8000)  # Wrong rate
vad=SileroVAD()
```

- [ ] Verify clear error message
- [ ] Message mentions 16kHz requirement

**Test offline mode (no model, no internet):**
```bash
rm ~/.cache/hearken/silero_vad_v5.onnx
# Disconnect internet
python examples/silero_vad.py
```

- [ ] Verify clear error message
- [ ] Message includes manual download instructions
- [ ] Message includes URL and paths

---

### 8. Performance

**Test:**
- [ ] Run for 5+ minutes continuously
- [ ] Monitor CPU usage
- [ ] Monitor memory usage
- [ ] Verify no memory leaks
- [ ] Verify real-time performance (no lag)

**Expected:**
- CPU usage reasonable (<50% on modern CPU)
- Memory stable (no growth over time)
- No audio drops or lag

---

## Success Criteria

- [ ] All 8 test scenarios pass
- [ ] No crashes or unexpected errors
- [ ] Performance acceptable for real-time use
- [ ] Better accuracy than EnergyVAD in noisy conditions
- [ ] Model caching works reliably
- [ ] Error messages are clear and actionable
