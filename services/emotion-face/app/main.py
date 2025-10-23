from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np, io, json, os, time
from PIL import Image, ImageTk
import onnxruntime as ort

import tkinter as tk

def _preview_for_2s(pil_img: Image.Image, title="Preview", seconds=2):
    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    tk_img = ImageTk.PhotoImage(pil_img)
    lbl = tk.Label(root, image=tk_img)
    lbl.pack()
    root.after(int(seconds * 1000), root.destroy)
    root.mainloop()

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "models")
PREPROC = json.load(open(os.path.join(MODEL_DIR, "preprocess-fer.json")))
IMG_SIZE = int(PREPROC["img_size"])
CHANNELS = int(PREPROC["channels"])
MEAN = np.array(PREPROC["mean"], dtype=np.float32).reshape(1,1,CHANNELS)
STD  = np.array(PREPROC["std"],  dtype=np.float32).reshape(1,1,CHANNELS)
CLASSES = PREPROC["classes"]
assert len(CLASSES) in (7,8), "We expect 7 or 8 classes."

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
session = ort.InferenceSession(os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx"), sess_options=so, providers=providers)
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
    "happy": (+0.8, +0.6), "happiness": (+0.8, +0.6),
    "sad": (-0.7, -0.4),   "sadness": (-0.7, -0.4),
    "angry": (-0.6, +0.7), "anger": (-0.6, +0.7),
    "fear": (-0.7, +0.7),
    "surprise": (+0.4, +0.8),
    "neutral": (0.0,  0.0),
    "disgust": (-0.6, +0.3),
    "contempt": (-0.3, +0.2),
}
VA_W = np.array([VA[c] for c in CLASSES], dtype=np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def preprocess_pil(pil: Image.Image, visualize: bool = False) -> np.ndarray:
    print(CHANNELS)
    if CHANNELS == 1:
        print("cahnnel 1")
        arr = np.array(pil.convert("L"))
        arr = np.array(Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        norm = (arr - MEAN.squeeze()) / STD.squeeze()    
        chw = np.expand_dims(norm, 0)                    

        if visualize:
            vis = norm * STD.squeeze() + MEAN.squeeze()
            vis = np.clip(vis, 0.0, 1.0)
            vis_img = Image.fromarray((vis * 255).astype(np.uint8), mode="L")
            _preview_for_2s(vis_img)

        return np.expand_dims(chw, 0)       

    else:
        print("channel else")
        arr = np.array(pil.convert("RGB"))
        arr = np.array(Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        norm = (arr - MEAN) / STD
        chw = np.transpose(norm, (2, 0, 1))              

        if visualize:
            vis = norm * STD + MEAN
            vis = np.clip(vis, 0.0, 1.0)
            vis_img = Image.fromarray((vis * 255).astype(np.uint8), mode="RGB")
            _preview_for_2s(vis_img)

        return np.expand_dims(chw, 0) 

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
    va = probs @ VA_W
    print(np.argsort(np.squeeze(probs))[::-1])
    pred_idx = int(np.argmax(probs))
    pred_emotion = CLASSES[pred_idx]
    print(f"[emotion-face] Predicted: {pred_emotion} | confidence={confidence:.3f}")

    t1 = time.perf_counter()
    return EmotionOut(
        probs=probs.tolist(),
        valence=float(va[0]),
        arousal=float(np.clip(va[1], 0.0, 1.0)),
        confidence=confidence,
        model_version=f"face-onnx@{app.version} ({(t1-t0)*1000:.1f}ms)"
    )