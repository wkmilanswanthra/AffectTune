from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, io, cv2
from PIL import Image
import time, os, base64

EMOTIONS = ["happy","sad","angry","fear","surprise","neutral","disgust"]

class EmotionOut(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: str

app = FastAPI(title="emotion-face", version="0.1.0")

def _softmax(x):
    x = np.array(x, dtype=np.float32)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).tolist()

def mock_infer(img: np.ndarray) -> EmotionOut:
    h, w = img.shape[:2]
    m = float(img.mean()) / 255.0
    s = float(img.std()) / 255.0
    seed = int((m * 1000) + (s * 1000)) % 9973
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, size=(7,))
    probs = _softmax(logits)
    valence = float(np.clip((probs[0] + probs[4] + probs[5]) - (probs[1] + probs[2] + probs[3] + probs[6]), -1, 1))
    arousal = float(np.clip(s * 2.0, 0, 1))
    conf = float(np.max(probs))
    return EmotionOut(probs=probs, valence=valence, arousal=arousal, confidence=conf, model_version="face-mock@0.1.0")

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/infer", response_model=EmotionOut)
async def infer(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    out = mock_infer(img)
    return out
