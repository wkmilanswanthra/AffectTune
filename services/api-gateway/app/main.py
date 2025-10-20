from fastapi import UploadFile, File
from fastapi import HTTPException
import httpx, os

FACE_URL  = os.getenv("FACE_URL",  "http://localhost:8001")
VOICE_URL = os.getenv("VOICE_URL", "http://localhost:8002")
FUSE_URL  = os.getenv("FUSE_URL",  "http://localhost:8003")

@app.post("/infer/face")
async def infer_face(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        r = await client.post(f"{FACE_URL}/infer", files=files)
        if r.status_code != 200: raise HTTPException(r.status_code, r.text)
        return r.json()

@app.post("/infer/voice")
async def infer_voice(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=15.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        r = await client.post(f"{VOICE_URL}/infer", files=files)
        if r.status_code != 200: raise HTTPException(r.status_code, r.text)
        return r.json()

@app.post("/infer/fusion")
async def infer_fusion(payload: dict):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{FUSE_URL}/fuse", json=payload)
        if r.status_code != 200: raise HTTPException(r.status_code, r.text)
        return r.json()
