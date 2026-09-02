# 12 — Research Paper

## Summary

The accompanying IEEE-format research paper documents the complete theoretical foundation, methodology, experimental setup, and results of the BrainTumorXAI project.

**Paper Title:** Explainable Ensemble Deep Learning with Grad-CAM Guided Morphological Segmentation for Brain Tumor Detection and Classification  
**Author:** Bhola Yadav  
**Institution:** CMR Institute of Technology  
**Paper File:** [`brain_tumor_xai_paper.tex`](../brain_tumor_xai_paper.tex)  
**Format:** IEEE Transactions style (LaTeX)

---

## Paper Structure

| Section | Title | Content |
|---|---|---|
| I | Introduction | Brain tumor importance, XAI motivation, paper contribution |
| II | Literature Survey | 18 related works covering CNN, ensemble, and XAI approaches |
| III | Methodology | Full pipeline: preprocessing, models, ensemble, Grad-CAM, segmentation algorithms |
| IV | Experimental Setup | Dataset, training config, evaluation protocol |
| V | Results & Discussion | Performance tables, confusion matrix, comparison with SOTA |
| VI | Conclusion & Future Work | Summary of achievements and future directions |
| References | Bibliography | 18 IEEE-formatted references |

---

## Key Algorithms in the Paper

The paper includes 3 formal Algorithm blocks:

### Algorithm 1: Image Preprocessing and Data Augmentation Pipeline
- `REQUIRE`: Raw CE-MRI image of size H × W
- `ENSURE`: Normalized, augmented batch tensor
- Covers: ImageDataGenerator initialization, per-image resize to 224×224, /255 normalization, and conditional augmentation (rotation, flip, zoom) during training only

### Algorithm 2: Soft-Voting Ensemble Classification
- `REQUIRE`: Preprocessed image tensor, trained DenseNet121 and InceptionV3 models
- `ENSURE`: Final class prediction, confidence score, probability vector
- Covers: Individual model forward passes, probability averaging, argmax class selection

### Algorithm 3: Grad-CAM Guided Morphological Tumor Segmentation
- `REQUIRE`: Heatmap H_r, original image I, τ_ROI=0.70, κ_max=25.0%
- `ENSURE`: Segmented overlay S, mask M, area percentage ρ
- Covers: Heatmap normalization, ROI masking, morphological close, largest contour selection, Otsu thresholding inside ROI, OPEN+CLOSE cleanup, red fill overlay

---

## Tables in the Paper

| Table | Content |
|---|---|
| Table I | Comparative Related Work summary (12 papers) |
| Table II | Technology stack (framework, library versions) |
| Table III | Model Performance Comparison (DenseNet121 vs InceptionV3 vs Ensemble) |
| Table IV | Per-class classification report (Precision, Recall, F1, Support) |
| Table V | Confusion Matrix (4×4) |
| Table VI | Comparison with State-of-the-Art methods |

---

## Base Paper

The methodology of BrainTumorXAI is directly based on:

> **[1]** K. M. Hosny, M. A. Mohammed, R. A. Salama, and A. M. Elshewey, "Explainable ensemble deep learning-based model for brain tumor detection and classification," *Neural Computing and Applications*, vol. 37, pp. 1289–1306, 2025. DOI: 10.1007/s00521-024-10401-0

This base paper establishes the dual-backbone ensemble framework (DenseNet + InceptionNet) with XAI integration as the primary methodology. BrainTumorXAI extends this with:
1. Grad-CAM-driven morphological segmentation (avoiding a dedicated segmentation model)
2. Anatomical location estimation from heatmap peak coordinates
3. Severity classification combining confidence + area metrics
4. Production-ready deployment (Gradio + FastAPI + Docker)

---

## Citation Style

The paper uses **IEEE citation format** with `\bibitem` entries in standard `thebibliography` environment.

### Citing This Work (BibTeX)

```bibtex
@misc{yadav2025braintumor,
  author    = {Bhola Yadav},
  title     = {Explainable Ensemble Deep Learning with Grad-CAM Guided Morphological
               Segmentation for Brain Tumor Detection and Classification},
  year      = {2025},
  institution = {CMR Institute of Technology},
  note      = {Final Year Major Project}
}
```

---

## Key Findings (Abstract Summary)

1. **Problem:** Standard deep learning tumor classifiers lack transparency — clinicians cannot verify model reasoning.
2. **Solution:** Ensemble of DenseNet121 + InceptionV3 with Grad-CAM-guided morphological segmentation.
3. **Novelty:** Segmentation without a dedicated segmentation model — the Grad-CAM heatmap doubles as a spatial prior for Otsu-based tumor extraction.
4. **Results:** Ensemble accuracy of **78.88%**, Macro F1 of **80%**, with visual explainability and tumor location/severity output.
5. **Deployment:** Fully functional Gradio dashboard + FastAPI REST API deployable via Docker.

---

## Future Work

As stated in the paper's conclusion:

1. **3D Medical Images** — Extend the pipeline to process 3D volumetric MRI data (NIfTI format)
2. **Federated Learning** — Enable privacy-preserving distributed training across multiple hospitals without sharing patient data
3. **Multi-Modal Imaging** — Incorporate T1, T2, and FLAIR MRI sequences for richer diagnostic information
4. **DICOM Integration** — Direct `.dcm` file input support for hospital PACS compatibility
5. **Dedicated Segmentation Model** — Optionally replace Grad-CAM-guided segmentation with a full U-Net trained on ground-truth masks

---

## Compiling the Paper in Overleaf

1. Upload `brain_tumor_xai_paper.tex` to Overleaf
2. Upload any referenced image files (workflow diagram, ROC curve, segmentation examples)
3. Set compiler to **pdfLaTeX**
4. Required LaTeX packages: `IEEEtran`, `algorithm`, `algorithmic`, `booktabs`, `graphicx`, `amsmath`, `hyperref`
5. Compile — expected output: ~10-page IEEE-format PDF
