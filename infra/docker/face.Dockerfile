FROM python:3.11-slim
WORKDIR /app
COPY services/emotion-face/. /app
RUN pip install -r /app/requirements.txt || true
RUN pip install fastapi uvicorn[standard] pillow numpy onnxruntime opencv-python-headless
EXPOSE 8001
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8001"]
