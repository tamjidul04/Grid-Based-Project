FROM python:3.11-slim

# System deps: PuLP's CBC solver needs gfortran runtime, libstdc++ etc.
# We pull them in once here so the image boots cleanly on HF Spaces.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cache-friendly: install deps before copying the rest of the source
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . /app

# HF Spaces expects port 7860 by default
ENV GRIDWISE_PORT=7860
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# Run the FastAPI app via uvicorn. The Space runs as root in the container
# which is fine for a public demo.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
