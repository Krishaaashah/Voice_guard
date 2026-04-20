import torch
from pathlib import Path

MODEL_PATH = "tl_v2_best.pth"
print(f"Checking model: {MODEL_PATH}")
if not Path(MODEL_PATH).exists():
    print("Model file does not exist")
else:
    try:
        ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        print("Keys in checkpoint:", ck.keys())
        print("Model loaded successfully on CPU")
    except Exception as e:
        print(f"Error loading model: {e}")
