FROM python:3.11-slim

# PuLP's CBC solver needs libgomp1 (OpenMP runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Slim runtime deps — no torch/transformers since LLM is opt-in via env var.
# Keeps the image <300 MB and boot time ~30s on Render free tier.
COPY requirements-render.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY llm /app/llm
COPY optimizer /app/optimizer
COPY api /app/api
COPY data/raw /app/data/raw

# Render sets $PORT automatically; uvicorn reads it via the shell.
ENV PYTHONUNBUFFERED=1
ENV GRIDWISE_USE_LLM=0
EXPOSE 10000

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-10000}"]
