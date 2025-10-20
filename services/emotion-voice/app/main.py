from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, io, soundfile as sf, librosa

EMOTIONS = ["happy","sad","angry","fear","surprise","neutral","disgust"]

class EmotionOut(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: str

app = FastAPI(title="emotion-voice", version="0.1.0")

def _softmax(x):
    x = np.array(x, dtype=np.float32); x -= x.max()
    e = np.exp(x); return (e / e.sum()).tolist()

def features(y, sr):
    rms = float(np.sqrt(np.mean(y**2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return rms, zcr, sc

def mock_infer(y, sr):
    rms, zcr, sc = features(y, sr)
    seed = int((rms*1e4) + (zcr*1e4) + (sc/100)) % 10007
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, size=(7,))
    probs = _softmax(logits)
    valence = float(np.clip((probs[0] + probs[4] + probs[5]) - (probs[1] + probs[2] + probs[3] + probs[6]), -1, 1))
    arousal = float(np.clip(rms * 8.0, 0, 1))
    conf = float(np.max(probs))
    return EmotionOut(probs=probs, valence=valence, arousal=arousal, confidence=conf, model_version="voice-mock@0.1.0")

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/infer", response_model=EmotionOut)
async def infer(file: UploadFile = File(...), sample_rate: int = 16000):
    data = await file.read()
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != sample_rate: y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate); sr = sample_rate
    y = y[: 2*sr]
    return mock_infer(y, sr)
