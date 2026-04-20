# VoiceGuard — Quick Start

## Files
```
voiceguard_prod/
├── app.py                  ← FastAPI backend
├── tl_v2_best.pth          ← Your trained model (AUC 0.9827)
├── tl_v2_final.pth         ← Alternate weights
├── requirements.txt
├── Dockerfile
├── README.md
└── templates/
    └── index.html          ← Production UI
```

## Run locally (2 minutes)
```bash
cd voiceguard_prod
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

## Deploy to HuggingFace Spaces (free public URL)
1. Go to https://huggingface.co/new-space
2. Name: `voiceguard` | SDK: **Docker**
3. Upload all files (drag & drop in Files tab)
4. Wait ~3 min → live at `yourname-voiceguard.hf.space`

## API
```bash
# Single file
curl -X POST http://localhost:8000/predict \
  -F "files=@audio.wav"

# Batch
curl -X POST http://localhost:8000/predict \
  -F "files=@voice1.wav" -F "files=@voice2.flac"

# Health check
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```

## Response
```json
[{
  "filename":     "audio.wav",
  "label":        "FAKE",
  "probability":  0.1203,
  "confidence":   0.8797,
  "inference_ms": 38,
  "waveform":     [0.12, 0.34, ...],
  "error":        null
}]
```

## Environment variables
| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `tl_v2_best.pth` | Path to model |
| `THRESHOLD` | `0.48` | P(real) cutoff |
