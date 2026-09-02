from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
import gradio as gr

from src.inference import predict_tumor_logic
from src.report_generator import generate_pdf_report
from src.theme import get_clinical_theme

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSS_PATH = os.path.join(_BASE_DIR, "assets", "styles.css")

EXAMPLE_IMAGES = [
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0025.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0070.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0080.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-pi_0050.jpg")],
]

def get_custom_css() -> str:
    """Reads and returns the external CSS stylesheet."""
    if os.path.exists(_CSS_PATH):
        with open(_CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_empty_states() -> Tuple[str, str, str, str]:
    diag = """
    <div class="result-card result-card-empty">
        <div class="result-empty-text">
            Upload an MRI axial slice and click <b>Analyze MRI Scan</b> to trigger Tri-Ensemble neural inference with multi-modal explainability.
        </div>
    </div>
    """
    conf = """
    <div class="result-card">
        <div class="card-title-sm">TRI-ENSEMBLE CONFIDENCE</div>
        <div class="conf-main-value conf-val-empty">-- %</div>
        <div class="conf-entropy-text">Awaiting scan input...</div>
    </div>
    """
    props = """
    <div class="result-card">
        <div class="card-title-sm">CLINICAL MEASUREMENTS</div>
        <div class="biomarkers-grid">
            <div class="biomarker-item"><div class="biomarker-label">Estimated Location</div><div class="biomarker-value">--</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Tumor Tissue Area</div><div class="biomarker-value">-- %</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Severity Grading</div><div class="biomarker-value">--</div></div>
            <div class="biomarker-item"><div class="biomarker-label">Uncertainty Entropy</div><div class="biomarker-value">-- %</div></div>
        </div>
    </div>
    """
    breakdown = """
    <div class="result-card">
        <div class="card-title-sm">MULTI-CLASS PROBABILITIES</div>
        <div class="conf-entropy-text">Glioma | Meningioma | Pituitary | No Tumor</div>
    </div>
    """
    return diag, conf, props, breakdown


def format_results(data: Dict[str, Any]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    str, str, str, str, str
]:
    if not data["is_valid"]:
        err = f"<div class='result-card result-card-error'><b>Error:</b> {data.get('error', 'Inference failed')}</div>"
        return None, None, None, None, err, "", "", "", ""

    is_tumor = data["is_tumor"]
    cls_name = data["class_name"]
    conf = data["confidence"]
    uncertainty = data.get("uncertainty", 0.0)
    sev = data["severity"]
    loc = data["location"]
    pct = data["tumor_percentage"]
    inf_time = data["inference_time"]

    card_state_class = "result-card result-card-tumor" if is_tumor else "result-card result-card-normal"
    badge_class = "diag-badge diag-badge-tumor" if is_tumor else "diag-badge diag-badge-normal"
    conf_val_class = "conf-main-value conf-val-tumor" if is_tumor else "conf-main-value conf-val-normal"
    progress_fill_class = "progress-fill progress-fill-tumor" if is_tumor else "progress-fill progress-fill-normal"

    diag_html = f"""
    <div class="{card_state_class}">
        <div class="diag-header">
            <span class="card-title-sm">Primary Diagnosis</span>
            <span class="{badge_class}">{sev.upper()}</span>
        </div>
        <div class="diag-main-title">{cls_name}</div>
        <div class="diag-meta-text">Inference completed in <span class="diag-meta-time">{inf_time:.3f}s</span> via Tri-Ensemble Fusion.</div>
    </div>
    """

    conf_html = f"""
    <div class="result-card">
        <div class="card-title-sm">Ensemble Confidence & Stability</div>
        <div class="conf-value-group">
            <div class="{conf_val_class}">{conf:.2f}%</div>
            <div class="conf-entropy-text">(Entropy: {uncertainty:.1f}%)</div>
        </div>
        <div class="progress-track">
            <div class="{progress_fill_class}" style="width:{conf}%;"></div>
        </div>
    </div>
    """

    props_html = f"""
    <div class="result-card">
        <div class="card-title-sm">Clinical Biomarkers</div>
        <div class="biomarkers-grid">
            <div class="biomarker-item">
                <div class="biomarker-label">Anatomical Lobe</div>
                <div class="biomarker-value">{loc}</div>
            </div>
            <div class="biomarker-item">
                <div class="biomarker-label">Tumor Tissue Area</div>
                <div class="biomarker-value">{pct:.2f}%</div>
            </div>
        </div>
    </div>
    """

    # Probability Breakdown
    prob_bars = ""
    bar_colors = ["#22c55e", "#ef4444", "#f59e0b", "#8b5cf6"]
    for i, (cname, prob) in enumerate(zip(data["classes"], data["avg_pred"])):
        p_pct = prob * 100.0
        prob_bars += f"""
        <div class="prob-row">
            <div class="prob-row-header">
                <span>{cname}</span>
                <b>{p_pct:.2f}%</b>
            </div>
            <div class="prob-progress-track">
                <div class="prob-progress-fill" style="width:{p_pct}%; background:{bar_colors[i % len(bar_colors)]};"></div>
            </div>
        </div>
        """
    
    breakdown_html = f"""
    <div class="result-card">
        <div class="card-title-sm">Classification Breakdown</div>
        {prob_bars}
    </div>
    """

    # Model agreement analytics table with responsive scroll wrapper
    pred_dict = data.get("pred_dict", {})
    cnx_p = pred_dict.get("ConvNeXtSmall", [0]*4)
    inc_p = pred_dict.get("InceptionV3", [0]*4)
    den_p = pred_dict.get("DenseNet121", [0]*4)

    analytics_html = f"""
    <div class="result-card">
        <div class="card-title-sm card-title-table">TRI-ENSEMBLE VOTING ANALYSIS</div>
        <div class="table-responsive-wrapper">
            <table class="analytics-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>ConvNeXt (45%)</th>
                        <th>Inception (35%)</th>
                        <th>DenseNet (20%)</th>
                        <th>Fused Ensemble</th>
                    </tr>
                </thead>
                <tbody>
    """
    for i, cname in enumerate(data["classes"]):
        analytics_html += f"""
                <tr>
                    <td>{cname}</td>
                    <td>{cnx_p[i]*100:.2f}%</td>
                    <td>{inc_p[i]*100:.2f}%</td>
                    <td>{den_p[i]*100:.2f}%</td>
                    <td class="analytics-fused-col">{data['avg_pred'][i]*100:.2f}%</td>
                </tr>
        """
    analytics_html += """
                </tbody>
            </table>
        </div>
    </div>
    """

    return (
        data["img"],
        data["gradcam_overlay"],
        data["gradcam_pp_overlay"],
        data["segmentation_img"],
        diag_html,
        conf_html,
        props_html,
        breakdown_html,
        analytics_html
    )


def handle_prediction(img: Optional[np.ndarray]):
    data = predict_tumor_logic(img)
    return format_results(data)


def handle_pdf_export(
    img: Optional[np.ndarray],
    patient_id: str,
    patient_name: str,
    patient_age: str,
    patient_gender: str
) -> Optional[str]:
    if img is None:
        return None
    data = predict_tumor_logic(img)
    if not data["is_valid"]:
        return None
    pdf_path = generate_pdf_report(
        diag_result=data,
        patient_id=patient_id or "PT-8942",
        patient_name=patient_name or "Anonymous Patient",
        patient_age=patient_age or "45",
        patient_gender=patient_gender or "Unspecified"
    )
    return pdf_path


def create_app() -> gr.Blocks:
    with gr.Blocks(title="BrainTumorXAI Clinical Studio", fill_width=True) as demo:
        # Header Banner
        with gr.Row(elem_id="app-header"):
            gr.HTML("""
            <div class="header-container">
                <div class="header-brand">
                    <div class="header-logo-badge">
                        <span class="header-logo-text">BT</span>
                    </div>
                    <div>
                        <h1 class="header-title">
                            BrainTumorXAI <span class="header-title-highlight">Clinical Diagnostic Studio</span>
                        </h1>
                        <p class="header-subtitle">
                            Explainable AI Ensemble Diagnostic System (ConvNeXtSmall + InceptionV3 + DenseNet121)
                        </p>
                    </div>
                </div>
                <div class="header-badges-group">
                    <span class="badge-primary">SOTA Tri-Ensemble (98.79% Target)</span>
                    <span class="badge-primary">Grad-CAM and Grad-CAM++</span>
                    <span class="badge-primary">PDF Clinical Reports</span>
                </div>
            </div>
            """)

        with gr.Tabs():
            # ============================================================
            # TAB 1: Clinical Diagnostics & Explainable AI
            # ============================================================
            with gr.TabItem("Clinical Diagnostics & XAI"):
                with gr.Row():
                    # Left Column: Upload & Controls
                    with gr.Column(scale=5, min_width=240):
                        with gr.Column(elem_classes="card-panel"):
                            gr.HTML('<div class="card-heading">1. Upload Brain MRI Axial Slice</div>')
                            input_img = gr.Image(type="numpy", label=None, show_label=False, height=280)
                            
                            with gr.Row():
                                predict_btn = gr.Button("Analyze MRI Scan", elem_classes="action-btn", scale=3)
                                clear_btn = gr.Button("Reset", elem_classes="clear-btn", scale=1)

                            gr.Examples(
                                examples=EXAMPLE_IMAGES,
                                inputs=input_img,
                                label="Sample MRI Scans",
                                examples_per_page=4,
                            )

                    # Right Column: Live Diagnostic Outputs
                    with gr.Column(scale=7, min_width=260):
                        empty_diag, empty_conf, empty_props, empty_breakdown = get_empty_states()
                        diag_box = gr.HTML(value=empty_diag)
                        conf_box = gr.HTML(value=empty_conf)
                        props_box = gr.HTML(value=empty_props)
                        breakdown_box = gr.HTML(value=empty_breakdown)

                # Bottom Row: 4-Way Side-by-Side Visualizations
                gr.HTML('<div class="section-title">Multi-Modal Explainable AI (XAI) Overlays</div>')
                with gr.Row():
                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-orig">(A) Input MRI Slice</div>')
                        out_orig = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-grad">(B) Grad-CAM Attention</div>')
                        out_grad = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-pp">(C) Grad-CAM++ (Multi-Focus)</div>')
                        out_grad_pp = gr.Image(label=None, show_label=False, interactive=False, height=240)

                    with gr.Column(scale=1, min_width=180, elem_classes="card-panel"):
                        gr.HTML('<div class="overlay-heading overlay-heading-seg">(D) Morphological Segmentation</div>')
                        out_seg = gr.Image(label=None, show_label=False, interactive=False, height=240)

            # ============================================================
            # TAB 2: Tri-Ensemble Analytics & Deep Model Breakdown
            # ============================================================
            with gr.TabItem("Tri-Ensemble Analytics"):
                gr.HTML("""
                <div class="result-card">
                    <div class="card-title-sm">Ensemble Consensus Framework</div>
                    <p style="color:#94a3b8; font-size:13.5px; margin:0; line-height:1.6;">
                        Our framework fuses predictions from three distinct deep learning architectures using Weighted Soft Voting:
                        <b style="color:#cbd5e1;">ConvNeXtSmall (45%)</b> + <b style="color:#cbd5e1;">InceptionV3 (35%)</b> + <b style="color:#cbd5e1;">DenseNet121 (20%)</b>.
                    </p>
                </div>
                """)
                analytics_box = gr.HTML("""
                <div class="result-card analytics-empty-box">
                    <div>Run an analysis in the Diagnostics tab to view the live multi-model breakdown.</div>
                </div>
                """)

            # ============================================================
            # TAB 3: Clinical PDF Report Generator & Export
            # ============================================================
            with gr.TabItem("Clinical Report Export (PDF)"):
                with gr.Row():
                    with gr.Column(scale=6, min_width=240, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Patient & Clinical Details</div>')
                        with gr.Row():
                            p_id = gr.Textbox(label="Patient ID", value="PT-8942")
                            p_name = gr.Textbox(label="Patient Name", value="Anonymous Patient")
                        with gr.Row():
                            p_age = gr.Textbox(label="Age", value="45")
                            p_gender = gr.Radio(choices=["Male", "Female", "Other"], label="Gender", value="Male")

                        generate_pdf_btn = gr.Button("Generate Official PDF Report", elem_classes="download-btn")

                    with gr.Column(scale=6, min_width=240, elem_classes="card-panel"):
                        gr.HTML('<div class="card-heading">Downloadable PDF Medical Document</div>')
                        pdf_download_file = gr.File(label="Download Generated PDF", interactive=False)
                        gr.HTML("""
                        <div class="report-notice-box">
                            The generated PDF includes patient information, Tri-Ensemble classification confidence, Grad-CAM attention scans, and clinical signature blocks.
                        </div>
                        """)

            # ============================================================
            # TAB 4: Architecture & Benchmark Reference
            # ============================================================
            with gr.TabItem("Architecture & Citations"):
                gr.HTML("""
                <div class="result-card doc-card">
                    <h3 class="doc-title">Deep Learning & Explainable AI Specifications</h3>
                    <ul class="doc-list">
                        <li><b>Primary Backbone: ConvNeXtSmall (Meta AI, CVPR 2022)</b>: 50M parameter modernized ConvNet with 7x7 depthwise convolutions, GELU activations, and LayerNorm. Standalone accuracy ~97.5% - 98.5%.</li>
                        <li><b>Multi-Scale Context: InceptionV3</b>: Parallel 1x1, 3x3, 5x5 receptive fields at native 299x299 resolution to capture both microscopic micro-adenomas and broad astrocytoma margins.</li>
                        <li><b>Feature Saliency: DenseNet121</b>: Direct feature concatenation across all dense blocks for high-resolution Grad-CAM visual gradient propagation.</li>
                        <li><b>Optimization: 3-Phase Progressive Training</b>: Warmup heads (Phase 1) &rarr; Top layer unfreezing (Phase 2) &rarr; Full backbone fine-tuning with AMSGrad Adam (Phase 3).</li>
                        <li><b>Inference Optimization: 10-Pass TTA</b>: Test-Time Augmentation across rotation, shifts, and zooms for robust generalization above 98.79%.</li>
                    </ul>
                </div>
                """)

        empty_diag, empty_conf, empty_props, empty_breakdown = get_empty_states()

        # Event Handlers
        predict_btn.click(
            fn=handle_prediction,
            inputs=[input_img],
            outputs=[
                out_orig,
                out_grad,
                out_grad_pp,
                out_seg,
                diag_box,
                conf_box,
                props_box,
                breakdown_box,
                analytics_box
            ]
        )

        clear_btn.click(
            fn=lambda: (
                None,  # input_img
                None,  # out_orig
                None,  # out_grad
                None,  # out_grad_pp
                None,  # out_seg
                empty_diag,
                empty_conf,
                empty_props,
                empty_breakdown,
                "<div class='result-card analytics-empty-box'><div>Run an analysis in the Diagnostics tab to view the live multi-model breakdown.</div></div>"
            ),
            inputs=[],
            outputs=[
                input_img,
                out_orig,
                out_grad,
                out_grad_pp,
                out_seg,
                diag_box,
                conf_box,
                props_box,
                breakdown_box,
                analytics_box
            ]
        )

        generate_pdf_btn.click(
            fn=handle_pdf_export,
            inputs=[input_img, p_id, p_name, p_age, p_gender],
            outputs=[pdf_download_file]
        )

    return demo
