# BrainTumorXAI — Documentation Index

Welcome to the full technical documentation for **BrainTumorXAI**, an explainable deep learning system for brain tumor detection, classification, and segmentation from CE-MRI scans.

---

## 📚 Documentation Pages

| Document | Description |
|---|---|
| [01 — Project Overview](./01_project_overview.md) | Goals, scope, system design, and novelty |
| [02 — Architecture](./02_architecture.md) | Full system architecture and data flow |
| [03 — Dataset](./03_dataset.md) | Dataset structure, classes, and preprocessing |
| [04 — Models & Training](./04_models_and_training.md) | DenseNet121, InceptionV3, ensemble logic, and training setup |
| [05 — Inference Pipeline](./05_inference_pipeline.md) | Step-by-step prediction, Grad-CAM, segmentation |
| [06 — API Reference](./06_api_reference.md) | FastAPI REST endpoints, request/response schemas |
| [07 — Dashboard (UI)](./07_dashboard_ui.md) | Gradio interface, components, and layout |
| [08 — Configuration](./08_configuration.md) | All constants, thresholds, and paths in `config.py` |
| [09 — Testing](./09_testing.md) | Test suite coverage, running tests, expected results |
| [10 — Deployment](./10_deployment.md) | Local setup, Docker, and production deployment guide |
| [11 — Results & Metrics](./11_results_and_metrics.md) | Classification accuracy, confusion matrix, and baselines |
| [12 — Research Paper](./12_research_paper.md) | Summary of the accompanying IEEE-format research paper |

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/bhola-dev58/final_year_project_brain_tumor_segmentation_explaination.git
cd Brain_Tumor_Project

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Gradio dashboard
python app.py
# → Open http://127.0.0.1:7860
```

---

**Author:** Bhola Yadav  
**Institution:** CMR Institute of Technology  
**Year:** 2025
