# One image for both roles. Build CPU image (default) or GPU image:
#   docker build -f deploy/docker/Dockerfile -t grid:cpu .
#   docker build -f deploy/docker/Dockerfile --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -t grid:gpu .
ARG BASE=python:3.11-slim
FROM ${BASE}
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY grid ./grid
COPY scripts ./scripts
# Default = worker. The gateway overrides the command (see compose / swarm / k8s files).
EXPOSE 50051 8000 29500
CMD ["python", "-m", "grid.worker.server"]
