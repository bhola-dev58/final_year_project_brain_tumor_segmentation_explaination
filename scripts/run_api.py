"""
Launcher script for the BrainTumorXAI FastAPI REST server.

Runs independently from the Gradio UI (app.py).

Usage:
    ./venv/bin/python scripts/run_api.py

Then open:
    http://localhost:8000/docs   — Interactive Swagger UI
    http://localhost:8000/redoc  — ReDoc documentation
    GET  http://localhost:8000/api/health
    POST http://localhost:8000/api/predict  (upload file=<mri.jpg>)
"""
import os
import sys

# Allow imports from the project root (e.g., src.inference, src.config)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    print("Starting BrainTumorXAI REST API server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,         # Disable hot-reload in production
        log_level="info",
    )
