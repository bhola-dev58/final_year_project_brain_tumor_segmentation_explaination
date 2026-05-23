# BrainTumorXAI: Explainable Brain Tumor Detection

An advanced, clinical-grade medical imaging application that uses **Ensemble Deep Learning** to detect, classify, and segment brain tumors from MRI scans. This project focuses on **Explainable AI (XAI)**, providing clinicians with visual evidence for every prediction.

## Key Features

*   **Ensemble Prediction:** Combines the strengths of **DenseNet121** and **InceptionV3** using Soft Voting for high-reliability results.
*   **Explainable AI (XAI):** Implements **Grad-CAM** heatmaps to highlight exactly which regions the model focused on.
*   **Automated Segmentation:** Generates tumor masks using morphological image processing.
*   **Tumor Analytics:** Estimates tumor location (Lobe/Quadrant), area percentage, and severity.
*   **Premium Dashboard:** A responsive, dark-themed interface built with Gradio for seamless interaction.

## Technology Stack

*   **Framework:** TensorFlow 2.x / Keras 3
*   **Models:** DenseNet121, InceptionV3
*   **UI:** Gradio
*   **Vision:** OpenCV, NumPy, Matplotlib
*   **Language:** Python 3.10+

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/BrainTumorXAI.git
cd BrainTumorXAI
```

### 2. Set Up Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Once the server starts, open the local URL (usually `http://127.0.0.1:7860`) in your browser.

## Project Structure

*   `app.py`: Main entry point for the Gradio application.
*   `model_core.py`: AI logic, ensemble prediction, and Grad-CAM generation.
*   `image_processing.py`: Image segmentation and tumor property estimation.
*   `ui_components.py`: Custom CSS and HTML dashboard components.
*   `models/`: Pre-trained weights for DenseNet121 and InceptionV3.

## Dataset
The models were trained on a comprehensive Brain Tumor MRI dataset consisting of 4 classes:
1.  **Glioma**
2.  **Meningioma**
3.  **Pituitary**
4.  **No Tumor**

## Medical Disclaimer
This software is for **educational and research purposes only**. It is not intended for clinical use or to replace professional medical diagnosis. Always consult a qualified healthcare professional for any medical concerns.

---
**Author:** Bhola Yadav
**Project:** Final Year/Major Project - Explainable Brain Tumor Detection
