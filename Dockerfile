# ============================================================
#  BrainTumorXAI — Dockerfile
#  CPU-only inference (no CUDA required)
#  Runs Gradio dashboard on port 7860
#  Base: python:3.12-slim (matches host runtime, required by numpy>=2.3)
# ============================================================

FROM python:3.12-slim

# System-level dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container
WORKDIR /app

# Copy dependency manifest first (layer-caching optimization)
COPY requirements.txt .

# Install Python dependencies (no cache to keep image lean)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app.py .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY test_images/ ./test_images/

# Expose Gradio port
EXPOSE 7860

# Force CPU-only execution inside container and disable output buffering
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Launch Gradio application
CMD ["python", "app.py"]
