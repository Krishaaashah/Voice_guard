"""
VoiceGuard - Production Backend
Model: Transfer Learning v2 (DCGAN backbone + new head)
AUC: 0.9827 | Accuracy: 93.5% | Precision: 95.8%

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

HuggingFace Spaces:
    Dockerfile exposes port 7860 automatically
"""

import os, time, tempfile, traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "tl_v2_best.pth")
THRESHOLD  = float(os.getenv("THRESHOLD", "0.92"))   # optimised on val set
MAX_MB     = 50
ALLOWED    = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}

AUDIO_CFG = dict(
    sample_rate=16000, duration=3.0,
    n_mels=64, n_fft=1024, hop_length=256,
    fmin=0, fmax=8000, spec_size=64,
)

MODEL_META = dict(
    name       = "VoiceGuard TL-v2",
    backbone   = "DCGAN Discriminator (pretrained)",
    head       = "AttentionPool + FC(512→256→64→1)",
    auc        = 0.9827,
    accuracy   = 0.9350,
    precision  = 0.9579,
    recall     = 0.9100,
    f1         = 0.9333,
    threshold  = THRESHOLD,
    params     = 2_903_682,
    trained_on = "LibriSpeech (real) + ASVspoof 2021 (fake)",
)

# ─── Model Architecture ───────────────────────────────────────────────────────
class AttentionPool2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, 1)
        self.last_weights = None

    def forward(self, x):
        w = torch.softmax(self.attn(x).view(x.size(0), -1), dim=1)
        self.last_weights = w.detach().cpu()
        w = w.view(x.size(0), 1, x.size(2), x.size(3))
        return (x * w).sum(dim=[2, 3])


class TransferClassifier(nn.Module):
    """
    DCGAN discriminator backbone + new classification head.
    Trained with transfer learning from discriminator_final.pth.
    Output: raw logit — sigmoid applied externally.
    """
    def __init__(self, ndf=64, dropout=0.4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1,      ndf,   4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf,    ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2,  ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4,  ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = AttentionPool2d(ndf * 8)
        fd = ndf * 8
        self.head = nn.Sequential(
            nn.Linear(fd, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.pool(self.backbone(x))).view(-1)


# ─── Model Loader ─────────────────────────────────────────────────────────────
_model  = None
_device = None


def load_model():
    global _model, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(MODEL_PATH).exists():
        print(f"[WARN] Model not found at '{MODEL_PATH}' - running in DEMO mode")
        _model = "demo"
        return

    print(f"Loading model from '{MODEL_PATH}' on {_device}...")
    ck = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    # Handle both checkpoint formats:
    # tl_v2_best.pth  → has 'model' key (full checkpoint)
    # tl_v2_final.pth → has 'state' key
    state_dict = ck.get("model") or ck.get("state")
    cfg        = ck.get("config", {})

    m = TransferClassifier(
        ndf     = cfg.get("ndf", 64),
        dropout = cfg.get("dropout", 0.4),
    ).to(_device)
    m.load_state_dict(state_dict)
    m.eval()
    _model = m

    n_params = sum(p.numel() for p in m.parameters())
    print(f"OK: Model loaded - {n_params:,} params | threshold={THRESHOLD}")


# ─── Audio Pipeline ──────────────────────────────────────────────────────────
def decode_audio(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            audio, sr = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != AUDIO_CFG["sample_rate"]:
                audio = librosa.resample(audio, orig_sr=sr,
                                         target_sr=AUDIO_CFG["sample_rate"])
        except Exception:
            audio, _ = librosa.load(tmp_path, sr=AUDIO_CFG["sample_rate"], mono=True)
    finally:
        os.unlink(tmp_path)

    audio, _ = librosa.effects.trim(audio, top_db=25)
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max()

    target = int(AUDIO_CFG["sample_rate"] * AUDIO_CFG["duration"])
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        mid   = max(0, (len(audio) - target) // 2)
        audio = audio[mid : mid + target]
    return audio


def make_mel(audio: np.ndarray) -> np.ndarray:
    sr, size = AUDIO_CFG["sample_rate"], AUDIO_CFG["spec_size"]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=AUDIO_CFG["n_mels"],
        n_fft=AUDIO_CFG["n_fft"],
        hop_length=AUDIO_CFG["hop_length"],
        fmin=AUDIO_CFG["fmin"],
        fmax=AUDIO_CFG["fmax"],
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    t_new    = np.linspace(0, mel_norm.shape[1] - 1, size)
    mel_r    = np.array([np.interp(t_new, np.arange(mel_norm.shape[1]), row)
                         for row in mel_norm])
    f_new    = np.linspace(0, mel_r.shape[0] - 1, size)
    mel_f    = np.array([np.interp(f_new, np.arange(mel_r.shape[0]), mel_r[:, t])
                         for t in range(size)]).T
    return mel_f.astype(np.float32)


def mel_to_bars(mel: np.ndarray, n: int = 60) -> List[float]:
    energy = mel.mean(axis=0)
    idxs   = np.linspace(0, len(energy) - 1, n).astype(int)
    bars   = energy[idxs]
    bars   = (bars - bars.min()) / (bars.max() - bars.min() + 1e-8)
    return [round(float(b), 4) for b in bars]


def run_inference(mel: np.ndarray) -> float:
    t = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
    t = (t * 2.0 - 1.0).to(_device)
    with torch.no_grad():
        logit = _model(t)
        prob  = torch.sigmoid(logit).item()
    return float(prob)


# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "VoiceGuard API",
    description = "Deepfake voice detection — Transfer Learning v2",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    load_model()


# ─── Schemas ─────────────────────────────────────────────────────────────────
class PredResult(BaseModel):
    filename    : str
    label       : str        # "REAL" | "FAKE" | "ERROR"
    probability : float      # P(real) ∈ [0,1]
    confidence  : float      # max(prob, 1-prob)
    inference_ms: int
    waveform    : List[float]
    error       : str | None = None


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path("templates/index.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VoiceGuard</h1><p>UI not found. POST to /predict</p>")


@app.get("/health")
async def health():
    return {
        "status"     : "ok",
        "model"      : "demo" if _model == "demo" else "loaded",
        "device"     : str(_device),
        "threshold"  : THRESHOLD,
        "model_path" : MODEL_PATH,
    }


@app.get("/model-info")
async def model_info():
    return MODEL_META


def compute_gradcam(mel: np.ndarray) -> dict:
    """
    Compute Grad-CAM on backbone last conv layer + attention weights.
    Returns frequency profile, time profile, and 4x4 attention grid.
    """
    import scipy.ndimage

    t = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
    t = (t * 2.0 - 1.0).to(_device)

    with torch.no_grad():
        feats  = _model.backbone(t)          # (1, 512, 4, 4)
        _       = _model.pool(feats)         # populates last_weights
        cam_raw = feats.squeeze(0)           # (512, 4, 4)
        weights = cam_raw.mean(dim=[1, 2])   # channel-wise global avg
        cam     = (weights[:, None, None] * cam_raw).sum(dim=0)
        cam     = torch.relu(cam)
        cmax    = cam.max()
        if cmax > 0:
            cam = cam / cmax
        cam_np = cam.cpu().numpy()           # (4, 4)

    # Upsample to 64x64 for smooth freq/time profiles
    cam64        = np.clip(scipy.ndimage.zoom(cam_np, 16, order=1), 0, 1)
    freq_profile = cam64.mean(axis=1).tolist()   # 64 mel-freq bins
    time_profile = cam64.mean(axis=0).tolist()   # 64 time frames

    attn_grid = None
    if _model.pool.last_weights is not None:
        attn_grid = _model.pool.last_weights.reshape(4, 4).tolist()

    return {
        "cam_4x4"     : cam_np.tolist(),
        "freq_profile": freq_profile,
        "time_profile": time_profile,
        "attn_grid"   : attn_grid,
    }


@app.post("/explain")
async def explain(files: List[UploadFile] = File(...)):
    """Return Grad-CAM explanation for the first valid audio file."""
    if _model == "demo":
        # Return synthetic data for demo mode
        import random
        fp = [abs(np.sin(i * 0.3 + 1.2) * 0.6 + random.uniform(0, 0.2)) for i in range(64)]
        tp = [abs(np.sin(i * 0.15) * 0.5 + random.uniform(0, 0.15)) for i in range(64)]
        return JSONResponse({"freq_profile": fp, "time_profile": tp,
                             "cam_4x4": [[random.random() for _ in range(4)] for _ in range(4)],
                             "attn_grid": [[random.random() for _ in range(4)] for _ in range(4)],
                             "label": "FAKE", "probability": 0.12})

    upload = files[0]
    fname  = upload.filename or "audio"
    data   = await upload.read()
    try:
        audio = decode_audio(data, fname)
        mel   = make_mel(audio)
        prob  = run_inference(mel)
        label = "REAL" if prob >= THRESHOLD else "FAKE"
        expl  = compute_gradcam(mel)
        expl["label"]       = label
        expl["probability"] = round(prob, 4)
        return JSONResponse(expl)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict", response_model=List[PredResult])
async def predict(files: List[UploadFile] = File(...)):
    results = []
    for upload in files:
        fname = upload.filename or "audio"
        ext   = Path(fname).suffix.lower()

        if ext not in ALLOWED:
            results.append(PredResult(
                filename=fname, label="ERROR", probability=0,
                confidence=0, inference_ms=0, waveform=[],
                error=f"Unsupported format '{ext}'. Use: {', '.join(sorted(ALLOWED))}",
            ))
            continue

        data = await upload.read()
        if len(data) > MAX_MB * 1024 * 1024:
            results.append(PredResult(
                filename=fname, label="ERROR", probability=0,
                confidence=0, inference_ms=0, waveform=[],
                error=f"File too large (max {MAX_MB} MB)",
            ))
            continue

        try:
            t0    = time.time()
            audio = decode_audio(data, fname)
            mel   = make_mel(audio)
            bars  = mel_to_bars(mel)

            if _model == "demo":
                import random
                prob = random.uniform(0.05, 0.95)
            else:
                prob  = run_inference(mel)

            ms    = int((time.time() - t0) * 1000)
            label = "REAL" if prob >= THRESHOLD else "FAKE"
            conf  = prob if label == "REAL" else 1.0 - prob

            results.append(PredResult(
                filename=fname, label=label,
                probability=round(prob, 4),
                confidence=round(conf, 4),
                inference_ms=ms,
                waveform=bars,
            ))

        except Exception as e:
            traceback.print_exc()
            results.append(PredResult(
                filename=fname, label="ERROR",
                probability=0, confidence=0,
                inference_ms=0, waveform=[],
                error=str(e),
            ))

    return results
