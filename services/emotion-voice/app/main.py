from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, io, soundfile as sf, librosa, os, time
import onnxruntime as ort

CLASSES = ["happy","sad","neutral","angry","fear","surprise","disgust"]

SR       = 16000
DUR_S    = 2.0
SAMPLES  = int(SR * DUR_S)
N_MELS   = 64
N_FFT    = 1024
HOP      = 256
FMIN     = 50
FMAX     = 8000

VA = {
    "happy": (+0.8, +0.6), "sad": (-0.7, -0.4), "neutral": (0.0, 0.0),
    "angry": (-0.6, +0.7), "fear": (-0.7, +0.7), "surprise": (+0.4, +0.8),
    "disgust": (-0.6, +0.3),
}
VA_W = np.array([VA[c] for c in CLASSES], dtype=np.float32) 

MODEL_PATH = os.getenv("VOICE_ONNX_PATH", "G:\\Work\\Github Projects\\Emotion-Driven Music Recommendation System\\emotion-music-reco\\services\\emotion-voice\\app\\models\\tinycnn.onnx")

class EmotionOut(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: str

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z, dtype=np.float32)
    return e / e.sum(axis=1, keepdims=True)

def load_wave_bytes(data: bytes) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1) 
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    return y

def to_logmel(y: np.ndarray) -> np.ndarray:
    M = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP,
        fmin=FMIN, fmax=FMAX
    )
    L = librosa.power_to_db(M).astype(np.float32)       
    L = (L - L.mean()) / (L.std() + 1e-6)               
    return L[None, None, ...]                           

app = FastAPI(title="emotion-voice", version="1.0.0")

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=providers)
INP = session.get_inputs()[0].name
OUT = session.get_outputs()[0].name

@app.get("/health")
def health():
    try:
        opset = session._model_meta.custom_metadata_map.get("onnx_opset")  
    except Exception:
        opset = None
    return {"status":"ok","providers":session.get_providers(),"opset":opset}

@app.post("/infer", response_model=EmotionOut)
async def infer(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await file.read()
    y = load_wave_bytes(data)
    x = to_logmel(y)                            
    logits = session.run([OUT], {INP: x})[0]    
    probs = softmax(logits)[0].astype(np.float32)
    conf = float(probs.max())
    va = probs @ VA_W                          

    t1 = time.perf_counter()
    return EmotionOut(
        probs=probs.tolist(),
        valence=float(va[0]),
        arousal=float(np.clip(va[1], 0.0, 1.0)),
        confidence=conf,
        model_version=f"voice-onnx@{app.version} {(t1-t0)*1000:.1f}ms"
    )