from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, io, json, os, time, cv2
from PIL import Image
import onnxruntime as ort

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "models")
PREPROC = json.load(open(os.path.join(MODEL_DIR, "preprocess.json")))
IMG_SIZE = int(PREPROC["img_size"])
MEAN = np.array(PREPROC["mean"], dtype=np.float32).reshape(1,1,3)
STD  = np.array(PREPROC["std"],  dtype=np.float32).reshape(1,1,3)
CLASSES = PREPROC["classes"]
assert len(CLASSES) == 7, "We expect 7 classes in fixed order."

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession(os.path.join(MODEL_DIR, "face.onnx"), sess_options=so, providers=providers)
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

class EmotionOut(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: str

app = FastAPI(title="emotion-face", version="1.0.0")

VA = {
 "happy": (+0.8, +0.6),
 "sad": (-0.7, -0.4),
 "angry": (-0.6, +0.7),
 "fear": (-0.7, +0.7),
 "surprise": (+0.4, +0.8),
 "neutral": (0.0,  0.0),
 "disgust": (-0.6, +0.3),
}
VA_W = np.array([VA[c] for c in CLASSES], dtype=np.float32)  

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def preprocess_pil(pil: Image.Image) -> np.ndarray:
    img = np.array(pil.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2,0,1))   
    return np.expand_dims(img, 0)   

@app.get("/health")
def health():
    return {"status":"ok","provider":providers[0]}

@app.post("/infer", response_model=EmotionOut)
async def infer(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess_pil(pil)
    logits = session.run([out_name], {inp_name: x})[0] 
    probs = softmax(logits).astype(np.float32)[0]      
    confidence = float(probs.max())
    va = probs @ VA_W  # (2,)
    t1 = time.perf_counter()
    return EmotionOut(
        probs=probs.tolist(),
        valence=float(va[0]),
        arousal=float(np.clip(va[1], 0.0, 1.0)),
        confidence=confidence,
        model_version=f"face-onnx@{app.version} ({(t1-t0)*1000:.1f}ms)"
    )
