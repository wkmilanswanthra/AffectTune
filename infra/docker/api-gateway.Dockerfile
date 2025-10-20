FROM python:3.11-slim
WORKDIR /app
COPY services/api-gateway/requirements.txt .
RUN pip install -r requirements.txt
COPY services/api-gateway/app ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
