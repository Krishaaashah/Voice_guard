FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces uses 7860
EXPOSE 7860

ENV MODEL_PATH=/app/tl_v2_best.pth
ENV THRESHOLD=0.48

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
