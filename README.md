---
title: VoiceGuard - Deepfake Voice Detection
emoji: shield
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# VoiceGuard

VoiceGuard is a production-ready deepfake voice detection system that classifies uploaded audio as **real** or **fake** using transfer learning from a DCGAN discriminator backbone. The application turns speech into normalized mel-spectrograms, runs a trained PyTorch classifier, and serves predictions through a FastAPI backend with a clean browser interface.

The project is framed around the core problem discussed in the AI plagiarism/deepfake report: generated speech can preserve language content while introducing subtle spectral and temporal artifacts. VoiceGuard focuses on those artifacts rather than transcript content, making it useful for audio authenticity screening, AI-generated voice checks, and batch analysis workflows.

## Demo

> Video walkthrough: add your demo video link here.

## Highlights

- Detects deepfake or synthetic speech from common audio formats: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, `.aac`, and `.webm`.
- Uses a transfer-learned DCGAN discriminator backbone with a lightweight attention pooling classification head.
- Converts audio into fixed-size 64 x 64 mel-spectrogram representations for stable model input.
- Supports single-file and batch inference through both the web UI and REST API.
- Includes Grad-CAM style explainability outputs for frequency, time, CAM heatmap, and attention grid inspection.
- Ships with Docker support for simple deployment to Hugging Face Spaces or any container platform.

## Results

The current TL-v2 checkpoint reports the following evaluation performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9827 |
| Accuracy | 93.5% |
| Precision | 95.8% |
| Recall | 91.0% |
| F1 Score | 93.3% |

Model metadata:

| Item | Value |
|---|---|
| Model name | VoiceGuard TL-v2 |
| Backbone | DCGAN discriminator |
| Head | AttentionPool + fully connected classifier |
| Parameters | 2,903,682 |
| Training data | LibriSpeech real speech + ASVspoof 2021 fake speech |
| Default model file | `tl_v2_best.pth` |
| Inference threshold | Configurable with `THRESHOLD` |

## Methodology

VoiceGuard follows a report-style experimental pipeline:

1. **Audio ingestion**
   - Accept user-uploaded audio files.
   - Validate extension and file size.
   - Decode audio with `soundfile` first and fall back to `librosa` when required.

2. **Signal normalization**
   - Convert multi-channel input to mono.
   - Resample audio to 16 kHz.
   - Trim leading and trailing silence.
   - Normalize amplitude to a consistent range.
   - Center-crop or zero-pad each clip to 3 seconds.

3. **Feature extraction**
   - Generate mel-spectrograms with 64 mel bins.
   - Convert power values to decibel scale.
   - Normalize the spectrogram to `[0, 1]`.
   - Resize the time-frequency map to a fixed 64 x 64 grid.

4. **Transfer learning**
   - Reuse a DCGAN discriminator as the spectral feature extractor.
   - Attach an attention pooling layer to summarize high-level time-frequency features.
   - Use a compact fully connected head for binary real/fake classification.
   - Apply sigmoid activation to obtain `P(real)`.

5. **Decision layer**
   - Compare `P(real)` against the configured threshold.
   - Return `REAL` when the probability is above the threshold.
   - Return `FAKE` when it is below the threshold.
   - Report confidence as the stronger side of the binary decision.

6. **Explainability**
   - Compute Grad-CAM style activation summaries from the final convolutional feature map.
   - Expose frequency and temporal profiles to show where the model concentrated.
   - Return the attention pooling grid for additional interpretability.

## Architecture

```text
Audio upload
    |
    v
Decode + resample + trim + normalize
    |
    v
64 x 64 mel-spectrogram
    |
    v
DCGAN discriminator backbone
    |
    v
AttentionPool2d
    |
    v
FC classifier head
    |
    v
P(real), label, confidence, waveform bars, explanation data
```

## Repository Structure

```text
voiceguard_prod/
|-- app.py                # FastAPI backend, model architecture, inference, explainability
|-- templates/
|   `-- index.html        # Browser UI for upload, batch results, history, and explanations
|-- tl_v2_best.pth        # Best trained checkpoint
|-- tl_v2_final.pth       # Alternate/final checkpoint
|-- requirements.txt      # Python dependencies
|-- Dockerfile            # Container build for production/Hugging Face Spaces
|-- QUICKSTART.md         # Short run and deployment notes
`-- README.md             # Project documentation
```

## Run Locally

```bash
cd voiceguard_prod
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

## API

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model-info
```

### Predict One File

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@audio.wav"
```

### Predict a Batch

```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@voice1.wav" \
  -F "files=@voice2.flac"
```

### Explainability

```bash
curl -X POST http://localhost:8000/explain \
  -F "files=@audio.wav"
```

Example prediction response:

```json
[
  {
    "filename": "audio.wav",
    "label": "FAKE",
    "probability": 0.1203,
    "confidence": 0.8797,
    "inference_ms": 38,
    "waveform": [0.12, 0.34, 0.41],
    "error": null
  }
]
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `tl_v2_best.pth` | Path to the checkpoint used at startup |
| `THRESHOLD` | `0.92` in local app config, `0.48` in Dockerfile | Decision cutoff for `P(real)` |

> Note: align `THRESHOLD` between `app.py` and the deployment environment before final evaluation or public release.

## Deployment

- Deploy with Docker by pushing this repository to GitHub, creating a Hugging Face Space with **SDK: Docker**, and adding the demo video link in the README after recording the walkthrough.

## Ethical Use

VoiceGuard should be used as a decision-support tool, not as the only evidence for accusing a speaker or content creator. Deepfake detectors can be sensitive to recording quality, background noise, codecs, unseen synthesis models, and dataset shift. For high-stakes use cases, combine model output with metadata review, human inspection, and additional forensic checks.

## License

MIT
