from typing import Tuple, Optional
import os
import numpy as np
import gradio as gr

from src.inference import predict_tumor_logic

# Absolute paths to real MRI scan examples (mask images are excluded)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_IMAGES = [
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0025.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0070.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-me_0080.jpg")],
    [os.path.join(_BASE_DIR, "test_images", "Tr-pi_0050.jpg")],
]

def get_custom_css() -> str:
    """
    Returns custom CSS rules for styling the Gradio clinical dashboard.
    """
    return """
    /* Global Dark Theme */
    .gradio-container {
        background: #0a0f1a !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        max-width: 100% !important;
        padding: 0 20px !important;
    }

    /* Make layout responsive */
    @media (max-width: 768px) {
        #app-header > div { flex-direction: column !important; align-items: flex-start !important; }
        .row-wrap { flex-direction: column !important; }
    }

    /* Header */
    #app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        padding: 28px 32px !important;
        margin-bottom: 16px !important;
    }

    /* Cards */
    .card-panel {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        width: 100% !important;
    }
    .card-panel .label-wrap {
        background: #1e293b !important;
        padding: 10px 16px !important;
    }
    .card-panel .label-wrap span {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* Image containers */
    .card-panel .image-container,
    .card-panel .upload-container {
        background: #0a0f1a !important;
        border: none !important;
        min-height: 280px !important;
    }

    /* Buttons */
    .action-btn {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .action-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    }

    .clear-btn {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    .clear-btn:hover {
        background: #334155 !important;
    }

    /* Result panels */
    .result-card {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        width: 100% !important;
        min-height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }

    /* Disclaimer */
    #disclaimer-box {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-top: 12px !important;
    }

    /* Hide footer elements */
    footer { display: none !important; }
    .built-with { display: none !important; }
    """

def get_empty_states() -> Tuple[str, str, str, str, str]:
    """
    Helper to return HTML templates for initial placeholder states of components.
    """
    empty_diag = """
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">AI Diagnosis</div>
        <div>Awaiting Scan Input...</div>
    </div>
    """
    empty_conf = """
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Model Confidence</div>
        <div>--</div>
    </div>
    """
    empty_break = """
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Prediction Breakdown</div>
        <div>Awaiting Analysis...</div>
    </div>
    """
    empty_prop = """
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Scan Properties</div>
        <div>N/A</div>
    </div>
    """
    empty_exp = """
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">AI Explanation</div>
        <div>Awaiting Analysis...</div>
    </div>
    """
    return empty_diag, empty_conf, empty_break, empty_prop, empty_exp


def build_html_outputs(data: dict) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    str, str, str, str, str
]:
    """
    Constructs HTML output string components based on model prediction dictionary data.
    """
    if not data["is_valid"]:
        error_html = f"<div style='color:#ef4444; padding:16px; font-weight:600;'>{data['error']}</div>"
        return None, None, None, error_html, "", "", "", ""
    
    is_tumor = data["is_tumor"]
    status_text = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"
    status_color = "#ef4444" if is_tumor else "#22c55e"

    # AI Diagnosis Panel
    diagnosis_html = f"""
    <div style="text-align:center; padding:16px;">
        <div style="font-size:14px; color:#94a3b8; margin-bottom:8px;">AI Diagnosis</div>
        <div style="font-size:28px; font-weight:800; color:{status_color};">
            {status_text}
        </div>
        <div style="font-size:13px; color:#64748b; margin-top:8px;">
            Model: DenseNet121 + InceptionV3 &bull; {data['inference_time']:.2f}s
        </div>
    </div>
    """

    # Model Confidence Panel
    conf_bar_color = "#ef4444" if is_tumor else "#22c55e"
    confidence_html = f"""
    <div style="padding:16px;">
        <div style="font-size:13px; color:#94a3b8; margin-bottom:4px;">Model Confidence</div>
        <div style="font-size:36px; font-weight:800; color:{conf_bar_color};">{data['confidence']:.2f}%</div>
        <div style="width:100%; background:#1e293b; border-radius:8px; height:10px; margin-top:8px; overflow:hidden;">
            <div style="width:{data['confidence']}%; height:100%; background:linear-gradient(90deg, {conf_bar_color}, {conf_bar_color}cc); border-radius:8px;"></div>
        </div>
    </div>
    """

    # Predictions Breakdown Panel
    breakdown_rows = ""
    for i, (cls, prob) in enumerate(zip(data['classes'], data['avg_pred'])):
        prob_pct = float(prob) * 100
        bar_color = "#ef4444" if cls != "No Tumor" and i == data['class_idx'] else ("#22c55e" if cls == "No Tumor" and i == data['class_idx'] else "#334155")
        dot_color = bar_color if i == data['class_idx'] else "#475569"
        breakdown_rows += f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="width:8px; height:8px; border-radius:50%; background:{dot_color}; flex-shrink:0;"></span>
            <span style="font-size:13px; color:#cbd5e1; min-width:130px;">{cls}</span>
            <div style="flex:1; background:#1e293b; border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{prob_pct}%; height:100%; background:{bar_color}; border-radius:4px;"></div>
            </div>
            <span style="font-size:13px; color:#94a3b8; min-width:50px; text-align:right;">{prob_pct:.2f}%</span>
        </div>
        """

    breakdown_html = f"""
    <div style="padding:16px;">
        <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">Prediction Breakdown</div>
        {breakdown_rows}
    </div>
    """

    # Properties & Explanation Panel
    if is_tumor:
        properties_html = f"""
        <div style="padding:16px;">
            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">Detected Tumor Properties</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#1e293b; padding:12px; border-radius:8px;">
                    <div style="font-size:11px; color:#64748b;">Tumor Type</div>
                    <div style="font-size:14px; font-weight:600; color:#f1f5f9;">{data['class_name']}</div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:8px;">
                    <div style="font-size:11px; color:#64748b;">Location</div>
                    <div style="font-size:14px; font-weight:600; color:#f1f5f9;">{data['location']}</div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:8px;">
                    <div style="font-size:11px; color:#64748b;">Tumor Area</div>
                    <div style="font-size:14px; font-weight:600; color:#f1f5f9;">{data['tumor_percentage']:.2f}%</div>
                </div>
                <div style="background:#1e293b; padding:12px; border-radius:8px;">
                    <div style="font-size:11px; color:#64748b;">Severity</div>
                    <div style="font-size:14px; font-weight:600; color:{data['severity_color']};">{data['severity']}</div>
                </div>
            </div>
        </div>
        """
        
        explanation_text = f"""
        The AI has detected a <strong style="color:#ef4444;">{data['class_name']}</strong> with <strong style="color:#ef4444;">{data['confidence']:.1f}% confidence</strong>.
        The tumor is primarily located in the <strong>{data['location']}</strong> and covers approximately <strong>{data['tumor_percentage']:.1f}%</strong> of the brain area shown. 
        Based on the size and confidence, the estimated severity is <strong style="color:{data['severity_color']};">{data['severity']}</strong>.
        The Grad-CAM heatmap indicates the regions the model focused on to make this diagnosis.
        """
        
    else:
        properties_html = f"""
        <div style="padding:16px;">
            <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">Scan Properties</div>
            <div style="background:#1e293b; padding:16px; border-radius:8px; text-align:center;">
                <div style="font-size:16px; font-weight:600; color:#22c55e;">No Tumor Detected</div>
                <div style="font-size:13px; color:#64748b; margin-top:4px;">Brain scan appears normal</div>
            </div>
        </div>
        """
        
        explanation_text = f"""
        The AI analyzed the MRI and found <strong style="color:#22c55e;">no distinct tumor patterns</strong> with a confidence of <strong style="color:#22c55e;">{data['confidence']:.1f}%</strong>.
        The brain scan appears normal. No anomalous regions or high-activation areas indicative of glioma, meningioma, or pituitary tumors were detected.
        """

    explanation_html = f"""
    <div style="padding:16px;">
        <div style="font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:12px;">AI Explanation</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.6; background:#1e293b; padding:12px; border-radius:8px; border-left:4px solid {status_color};">
            {explanation_text}
        </div>
    </div>
    """

    return (
        data["img"],
        data["segmentation_img"],
        data["gradcam_overlay"],
        diagnosis_html,
        confidence_html,
        breakdown_html,
        properties_html,
        explanation_html
    )


def predict_and_format(img: Optional[np.ndarray]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
    str, str, str, str, str
]:
    """
    Runs classification inference and returns formatted HTML dashboard content.
    """
    data = predict_tumor_logic(img)
    if not data["is_valid"]:
        error_html = f"<div style='padding:16px; color:#ef4444; font-weight:600;'>{data['error']}</div>"
        return None, None, None, error_html, "", "", "", ""
    return build_html_outputs(data)


def clear_outputs() -> Tuple[
    None, None, None, None, str, str, str, str, str
]:
    """
    Resets the input images and replaces outputs with initial empty states.
    """
    empty_diag, empty_conf, empty_break, empty_prop, empty_exp = get_empty_states()
    return None, None, None, None, empty_diag, empty_conf, empty_break, empty_prop, empty_exp


def create_app() -> gr.Blocks:
    """
    Constructs and returns the Gradio blocks layout ready to be run or mounted.
    """
    with gr.Blocks(title="BrainTumorXAI", fill_width=True) as demo:
        with gr.Row(elem_id="app-header"):
            gr.HTML("""
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="width:56px; height:56px; background:linear-gradient(135deg, #6366f1, #8b5cf6);
                                border-radius:14px; display:flex; align-items:center; justify-content:center;
                                box-shadow: 0 4px 15px rgba(99,102,241,0.3);">
                        <span style="font-size:24px; color:white; font-weight:bold;">BT</span>
                    </div>
                    <div>
                        <h1 style="margin:0; font-size:26px; font-weight:800; color:#f1f5f9; line-height:1.2;">
                            Explainable <span style="color:#818cf8;">Brain Tumor</span> Detection and Segmentation
                        </h1>
                        <p style="margin:4px 0 0 0; font-size:14px; color:#64748b;">
                            AI-powered detection, segmentation and explainability for brain MRI images
                        </p>
                    </div>
                </div>
                <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                    <span style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:6px 14px; font-size:12px; color:#94a3b8; white-space:nowrap;">DenseNet121 + InceptionV3</span>
                    <span style="background:#1e293b; border:1px solid #334155; border-radius:8px; padding:6px 14px; font-size:12px; color:#94a3b8; white-space:nowrap;">Ensemble (Soft Voting)</span>
                </div>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=7, min_width=300):
                with gr.Row():
                    with gr.Column(scale=1, min_width=250, elem_classes="card-panel"):
                        gr.HTML('<div style="background:#1e293b; padding:10px 16px; font-size:14px; font-weight:600; color:#e2e8f0;">Upload Brain MRI</div>')
                        input_img = gr.Image(type="numpy", label=None, show_label=False, height=300)
                        with gr.Row():
                            predict_btn = gr.Button("Analyze MRI", elem_classes="action-btn", scale=3)
                            clear_btn = gr.Button("Clear", elem_classes="clear-btn", scale=1)
                        gr.Examples(
                            examples=EXAMPLE_IMAGES,
                            inputs=input_img,
                            label="Quick Examples — Click to Load",
                            examples_per_page=4,
                        )

                    with gr.Column(scale=1, min_width=250, elem_classes="card-panel"):
                        gr.HTML('<div style="background:#1e293b; padding:10px 16px; font-size:14px; font-weight:600; color:#e2e8f0;">Uploaded Image</div>')
                        uploaded_preview = gr.Image(label=None, show_label=False, interactive=False, height=300)

                with gr.Row():
                    with gr.Column(scale=1, min_width=250, elem_classes="card-panel"):
                        gr.HTML('<div style="background:#1e293b; padding:10px 16px; font-size:14px; font-weight:600; color:#e2e8f0;">Segmentation (AI Output)</div>')
                        seg_output = gr.Image(label=None, show_label=False, interactive=False, height=300)
                        gr.HTML("""
                        <div style="display:flex; gap:16px; padding:8px 16px; justify-content:center;">
                            <span style="display:flex; align-items:center; gap:4px; font-size:12px; color:#94a3b8;">
                                <span style="width:12px; height:12px; background:#dc2626; border-radius:2px; display:inline-block;"></span> Tumor
                            </span>
                            <span style="display:flex; align-items:center; gap:4px; font-size:12px; color:#94a3b8;">
                                <span style="width:12px; height:12px; background:#1e293b; border:1px solid #475569; border-radius:2px; display:inline-block;"></span> Background
                            </span>
                        </div>
                        """)

                    with gr.Column(scale=1, min_width=250, elem_classes="card-panel"):
                        gr.HTML('<div style="background:#1e293b; padding:10px 16px; font-size:14px; font-weight:600; color:#e2e8f0;">Explainability (Grad-CAM)</div>')
                        gradcam_output = gr.Image(label=None, show_label=False, interactive=False, height=300)
                        gr.HTML("""
                        <div style="display:flex; align-items:center; justify-content:center; gap:4px; padding:8px 16px;">
                            <span style="font-size:11px; color:#94a3b8;">Low</span>
                            <div style="width:120px; height:10px; border-radius:5px; background:linear-gradient(90deg, #3b82f6, #22d3ee, #22c55e, #eab308, #ef4444);"></div>
                            <span style="font-size:11px; color:#94a3b8;">High</span>
                        </div>
                        """)

            with gr.Column(scale=3, min_width=300):
                empty_diag, empty_conf, empty_break, empty_prop, empty_exp = get_empty_states()
                diagnosis_output   = gr.HTML(value=empty_diag, elem_classes="result-card")
                confidence_output  = gr.HTML(value=empty_conf, elem_classes="result-card")
                breakdown_output   = gr.HTML(value=empty_break, elem_classes="result-card")
                properties_output  = gr.HTML(value=empty_prop, elem_classes="result-card")
                explanation_output = gr.HTML(value=empty_exp, elem_classes="result-card")

        gr.HTML("""
        <div id="disclaimer-box">
            <div style="font-size:13px; color:#94a3b8; line-height:1.6;">
                <strong style="color:#e2e8f0;">Note:</strong> This system is for research and educational purposes only.
                Always consult a qualified healthcare professional for medical diagnosis.
                This tool is designed to assist radiologists and clinicians - it does not replace professional medical advice.
            </div>
        </div>
        <div style="text-align:center; padding:16px 0 8px 0; color:#475569; font-size:12px;">
            2025 BrainTumorXAI | Explainable AI for Brain Tumor Detection and Segmentation<br>
            <span style="color:#334155;">Built with TensorFlow, Keras & Gradio</span>
        </div>
        """)

        predict_btn.click(
            fn=predict_and_format,
            inputs=[input_img],
            outputs=[
                uploaded_preview,
                seg_output,
                gradcam_output,
                diagnosis_output,
                confidence_output,
                breakdown_output,
                properties_output,
                explanation_output
            ]
        )

        clear_btn.click(
            fn=clear_outputs,
            inputs=[],
            outputs=[
                input_img,
                uploaded_preview,
                seg_output,
                gradcam_output,
                diagnosis_output,
                confidence_output,
                breakdown_output,
                properties_output,
                explanation_output
            ]
        )
        
    return demo
