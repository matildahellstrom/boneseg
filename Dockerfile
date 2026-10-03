# boneseg web app. Build: docker build -t boneseg .
# Run:   docker run -p 8000:8000 -v "$PWD/projects:/data" boneseg
# With an NVIDIA GPU, swap the CPU wheel index below for a CUDA one and add --gpus all to docker run.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BONESEG_DATA_DIR=/data TORCH_HOME=/data/torch-cache

WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY boneseg ./boneseg

VOLUME /data
EXPOSE 8000
# Listens on all interfaces inside the container; the -p flag decides who can reach it
CMD ["python", "-m", "boneseg", "serve", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/data"]
