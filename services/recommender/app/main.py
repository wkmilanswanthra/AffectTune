from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import Literal, Optional

# -------------------------
# Constants / Labels
# -------------------------
EMOTIONS = ["happy","sad","angry","fear","surprise","neutral","disgust"]
N = len(EMOTIONS)

# -------------------------
# Pydantic Schemas
# -------------------------
class EmotionOut(BaseModel):
    probs: list[float]                 # length N
    valence: float                     # [-1, +1]
    arousal: float                     # [0, 1]
    confidence: float                  # [0, 1]
    model_version: str

class EmotionIn(BaseModel):
    probs: list[float]
    valence: float
    arousal: float
    confidence: float
    model_version: Optional[str] = None

class FusionIn(BaseModel):
    face: Optional[EmotionIn] = None
    voice: Optional[EmotionIn] = None

    # Base weights (will be normalized)
    w_face: float = Field(0.6, ge=0.0)
    w_voice: float = Field(0.4, ge=0.0)

    # EMA smoothing factor [0..1]; 0 disables smoothing
    smooth: float = Field(0.2, ge=0.0, le=1.0)

    # How to combine modalities
    strategy: Literal["fixed", "conf"] = "conf"
    # Temperature for softmax-like sharpening/softening (1=no change; <1 sharper; >1 softer)
    temperature: float = Field(1.0, gt=0.0)

    # If a modality confidence < min_conf, treat as missing
    min_conf: float = Field(0.0, ge=0.0, le=1.0)

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="fusion", version="1.1.0")

# Keep an EMA of the last fused output: [probs(N), val, aro, conf]
_last: Optional[list] = None

# -------------------------
# Helpers
# -------------------------
def _validate_probs(p: np.ndarray) -> np.ndarray:
    if p.ndim != 1 or p.size != N:
        raise HTTPException(status_code=400, detail=f"probs must be a length-{N} list")
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        # replace invalid with uniform
        p = np.ones(N, dtype=np.float32) / N
    s = p.sum()
    if s <= 0:
        p = np.ones(N, dtype=np.float32) / N
    else:
        p = p / s
    return p.astype(np.float32)

def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    if abs(T - 1.0) < 1e-6:
        return p
    # log + scale + exp trick
    # clamp to avoid log(0)
    p = np.clip(p, 1e-8, 1.0)
    z = np.log(p) / T
    z = z - z.max()                 # stability
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)

def _ema(prev, curr, a: float):
    if prev is None:
        return curr
    return (1.0 - a) * np.array(prev, dtype=np.float32) + a * np.array(curr, dtype=np.float32)

def _val_arousal_avg(face: Optional[EmotionIn], voice: Optional[EmotionIn]) -> tuple[float, float]:
    vals = []; aros = []
    if face is not None:
        vals.append(float(np.clip(face.valence, -1.0, 1.0)))
        aros.append(float(np.clip(face.arousal, 0.0, 1.0)))
    if voice is not None:
        vals.append(float(np.clip(voice.valence, -1.0, 1.0)))
        aros.append(float(np.clip(voice.arousal, 0.0, 1.0)))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.mean(aros))

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "labels": EMOTIONS,
        "ema_initialized": _last is not None,
    }

@app.post("/reset")
def reset_state():
    global _last
    _last = None
    return {"status": "cleared"}

@app.post("/fuse", response_model=EmotionOut)
def fuse(inp: FusionIn):
    global _last

    # Prepare per-modality distributions (or default neutral)
    neutral = np.zeros(N, dtype=np.float32); neutral[EMOTIONS.index("neutral")] = 1.0

    face_probs = None
    voice_probs = None
    face_conf = 0.0
    voice_conf = 0.0

    if inp.face is not None:
        face_probs = _validate_probs(np.array(inp.face.probs, dtype=np.float32))
        face_conf = float(np.clip(inp.face.confidence, 0.0, 1.0))
        if face_conf < inp.min_conf:
            face_probs, face_conf = None, 0.0

    if inp.voice is not None:
        voice_probs = _validate_probs(np.array(inp.voice.probs, dtype=np.float32))
        voice_conf = float(np.clip(inp.voice.confidence, 0.0, 1.0))
        if voice_conf < inp.min_conf:
            voice_probs, voice_conf = None, 0.0

    # If both missing -> neutral
    if face_probs is None and voice_probs is None:
        probs = neutral
        val, aro = 0.0, 0.0
        conf = 0.0
    else:
        # Temperature scaling
        T = max(1e-6, float(inp.temperature))
        if face_probs is not None:
            face_probs = _apply_temperature(face_probs, T)
        if voice_probs is not None:
            voice_probs = _apply_temperature(voice_probs, T)

        # Base weights (normalized)
        w_face = max(0.0, float(inp.w_face))
        w_voice = max(0.0, float(inp.w_voice))
        denom = w_face + w_voice
        if denom <= 0:
            w_face, w_voice = 0.5, 0.5
        else:
            w_face, w_voice = w_face / denom, w_voice / denom

        # Strategy: confidence-aware weighting (default) or fixed
        if inp.strategy == "conf":
            w_face *= face_conf
            w_voice *= voice_conf
            d = w_face + w_voice
            if d <= 1e-12:
                # if confidences both zero, fall back to equal/non-missing
                if face_probs is not None and voice_probs is not None:
                    w_face = w_voice = 0.5
                elif face_probs is not None:
                    w_face, w_voice = 1.0, 0.0
                else:
                    w_face, w_voice = 0.0, 1.0
            else:
                w_face, w_voice = w_face / d, w_voice / d

        # Compose distributions
        p_face = face_probs if face_probs is not None else neutral
        p_voice = voice_probs if voice_probs is not None else neutral

        probs = (w_face * p_face + w_voice * p_voice)
        s = probs.sum()
        probs = (probs / s) if s > 0 else neutral
        probs = probs.astype(np.float32)

        # Valence/Arousal: mean of available
        val, aro = _val_arousal_avg(inp.face if face_probs is not None else None,
                                    inp.voice if voice_probs is not None else None)

        # Confidence: weighted max of each modality's confidence
        conf = float(max(face_conf * w_face, voice_conf * w_voice))

    # Optional EMA smoothing
    if inp.smooth > 0.0:
        curr = [probs, np.array([val], np.float32), np.array([aro], np.float32), np.array([conf], np.float32)]
        fused = _ema(_last, curr, float(inp.smooth))
        _last = fused
        probs = fused[0].astype(np.float32)
        val   = float(fused[1])
        aro   = float(fused[2])
        conf  = float(np.clip(fused[3], 0.0, 1.0))

    # Ensure well-formed output
    probs = _validate_probs(probs)

    return EmotionOut(
        probs=probs.tolist(),
        valence=float(np.clip(val, -1.0, 1.0)),
        arousal=float(np.clip(aro, 0.0, 1.0)),
        confidence=float(np.clip(conf, 0.0, 1.0)),
        model_version=f"fusion@{app.version}"
    )
