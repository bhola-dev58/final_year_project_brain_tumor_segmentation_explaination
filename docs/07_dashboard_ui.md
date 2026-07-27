# 07 — Dashboard (UI)

The interactive dashboard is built with **Gradio 6.x** and provides a dark-themed, clinical-grade interface for brain tumor analysis. It is defined in `src/dashboard.py` and launched via `app.py`.

---

## Launching the Dashboard

```bash
python app.py
```

- Local URL: `http://127.0.0.1:7860`
- Public shareable URL: Printed to the terminal automatically (valid for 1 week via Gradio's share tunnel)

---

## Layout Overview

The dashboard is divided into three main zones:

```
┌──────────────────────────────────────────────────────────────────┐
│  HEADER: BrainTumorXAI title + badges (DenseNet121 + InceptionV3)│
└──────────────────────────────────────────────────────────────────┘
┌─────────────────────────────┐  ┌───────────────────────────────┐
│       LEFT COLUMN (70%)     │  │     RIGHT COLUMN (30%)        │
│                             │  │                               │
│  [Upload MRI] [Preview]     │  │  [AI Diagnosis]               │
│  [Analyze] [Clear]          │  │  [Model Confidence]           │
│  [Quick Examples]           │  │  [Prediction Breakdown]       │
│                             │  │  [Tumor Properties]           │
│  [Segmentation] [Grad-CAM]  │  │  [AI Explanation]             │
└─────────────────────────────┘  └───────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  DISCLAIMER: Research/educational use only                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## UI Components

### Header Bar (`#app-header`)
- Displays the system name with a gradient accent
- Shows two status badges: "DenseNet121 + InceptionV3" and "Ensemble (Soft Voting)"
- Dark gradient background (`#0f172a → #1e1b4b → #0f172a`)
- Responsive: stacks vertically on mobile

### Upload Panel
- `gr.Image(type="numpy")` — uploads an image and returns it as a numpy array
- The image is NOT displayed here — only the upload widget
- **Analyze MRI** button (blue gradient) — triggers full inference
- **Clear** button (dark gray) — resets all panels
- **Quick Examples** (`gr.Examples`) — 4 pre-loaded MRI scan thumbnails. Click to load instantly

### Preview Panel (Uploaded Image)
- `gr.Image(interactive=False)` — displays the original uploaded scan after analysis

### Segmentation Panel
- Displays the red-overlay segmentation image output
- Includes a legend: `■ Tumor` (red) and `■ Background` (dark)

### Grad-CAM Panel
- Displays the JET-colormap heatmap overlay
- Includes a gradient bar legend: Low → High activation

### Results Sidebar (Right Column)

All result panels are `gr.HTML` components with custom HTML/CSS rendering:

| Panel | Shows |
|---|---|
| **AI Diagnosis** | "TUMOR DETECTED" (red) or "NO TUMOR DETECTED" (green), inference time |
| **Model Confidence** | Large percentage number + animated progress bar |
| **Prediction Breakdown** | Per-class probability bars for all 4 classes |
| **Tumor Properties** | 2×2 grid: Type, Location, Tumor Area %, Severity |
| **AI Explanation** | Natural language summary of what the AI found |

---

## Styling (Dark Theme CSS)

Custom CSS is applied through `get_custom_css()`. Key design choices:

| Element | Style |
|---|---|
| Background | `#0a0f1a` (very dark navy) |
| Card panels | `#0f172a` with `#1e293b` border |
| Analyze button | Blue-to-indigo gradient (`#3b82f6 → #6366f1`) |
| Clear button | Dark gray (`#1e293b`) |
| Positive (No Tumor) | Green `#22c55e` |
| Negative (Tumor) | Red `#ef4444` |
| Severity High | Red `#ef4444` |
| Severity Moderate | Amber `#f59e0b` |
| Severity Low | Green `#22c55e` |
| Severity Uncertain | Gray `#6b7280` |
| Typography | Inter, Segoe UI, system-ui |

All panels have `border-radius: 12px` for a modern card look.

---

## Quick Examples

The dashboard preloads 4 real MRI scans from the `test_images/` folder as clickable examples:

```python
EXAMPLE_IMAGES = [
    ["test_images/Tr-me_0025.jpg"],   # Meningioma
    ["test_images/Tr-me_0070.jpg"],   # Meningioma
    ["test_images/Tr-me_0080.jpg"],   # Meningioma
    ["test_images/Tr-pi_0050.jpg"],   # Pituitary
]
```

Clicking any example automatically loads the image into the upload box — no file browser needed.

---

## Event Handlers

### Analyze Button
```python
predict_btn.click(
    fn=predict_and_format,
    inputs=[input_img],
    outputs=[
        uploaded_preview,     # shows original image
        seg_output,           # shows segmentation overlay
        gradcam_output,       # shows Grad-CAM overlay
        diagnosis_output,     # updates diagnosis HTML
        confidence_output,    # updates confidence HTML
        breakdown_output,     # updates breakdown HTML
        properties_output,    # updates properties HTML
        explanation_output    # updates explanation HTML
    ]
)
```

### Clear Button
```python
clear_btn.click(
    fn=clear_outputs,
    inputs=[],
    outputs=[
        input_img,            # clears upload box
        uploaded_preview,     # clears preview
        seg_output,           # clears segmentation
        gradcam_output,       # clears Grad-CAM
        diagnosis_output,     # resets to "Awaiting..."
        confidence_output,
        breakdown_output,
        properties_output,
        explanation_output
    ]
)
```

---

## Output HTML Builder

The `build_html_outputs()` function converts the raw prediction dictionary into 5 HTML strings. It handles both the tumor and no-tumor cases:

```python
# Tumor detected → red status, tumor property grid, detailed explanation
# No tumor → green status, "Brain scan appears normal" message
```

The AI Explanation panel generates natural language text like:

> *"The AI has detected a **Glioma Tumor** with **92.4% confidence**. The tumor is primarily located in the **Left Frontal Lobe (Superior)** and covers approximately **4.2%** of the brain area shown. Based on the size and confidence, the estimated severity is **Moderate**. The Grad-CAM heatmap indicates the regions the model focused on to make this diagnosis."*

---

## Disclaimer

The dashboard footer always displays:

> **Note:** This system is for research and educational purposes only. Always consult a qualified healthcare professional for medical diagnosis. This tool is designed to assist radiologists and clinicians — it does not replace professional medical advice.
