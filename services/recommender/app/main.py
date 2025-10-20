from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, httpx

EMOTIONS = ["happy","sad","angry","fear","surprise","neutral","disgust"]

class EmotionOut(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: str

class FusionIn(BaseModel):
    face: EmotionOut | None = None
    voice: EmotionOut | None = None
    w_face: float = 0.6
    w_voice: float = 0.4
    smooth: float = 0.2  

app = FastAPI(title="fusion", version="0.1.0")
_last = None  

def _ema(prev, curr, a):
    if prev is None: return curr
    return (1-a)*np.array(prev) + a*np.array(curr)

@app.post("/fuse", response_model=EmotionOut)
def fuse(inp: FusionIn):
    global _last
    z = np.zeros(7, dtype=np.float32); z[5] = 1.0
    p_face = np.array(inp.face.probs if inp.face else z)
    p_voice = np.array(inp.voice.probs if inp.voice else z)
    probs = (inp.w_face * p_face + inp.w_voice * p_voice)
    probs = (probs / probs.sum()).tolist()

    val = float(np.clip(((inp.face.valence if inp.face else 0)+(inp.voice.valence if inp.voice else 0))/2, -1, 1))
    aro = float(np.clip(((inp.face.arousal if inp.face else 0)+(inp.voice.arousal if inp.voice else 0))/2, 0, 1))
    conf = float(max(inp.face.confidence if inp.face else 0, inp.voice.confidence if inp.voice else 0))
    out = [probs, val, aro, conf]

    if inp.smooth > 0:
        _last = _ema(_last, out, inp.smooth)
        probs, val, aro, conf = _last[0].tolist(), float(_last[1]), float(_last[2]), float(_last[3])

    return EmotionOut(probs=probs, valence=val, arousal=aro, confidence=conf, model_version="fusion@0.1.0")
