from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi import HTTPException
import httpx, os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Emotion Reco API Gateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("API Gateway starting...")
class Health(BaseModel):
    status: str

@app.get("/health", response_model=Health)
def health():
    print("Health check received")
    return {"status": "ok"}

@app.websocket("/ws/echo")
async def ws_echo(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            await ws.send_text(f"echo:{msg}")
    except Exception:
        await ws.close()


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
