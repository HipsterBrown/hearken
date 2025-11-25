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
