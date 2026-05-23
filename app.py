import gradio as gr
from model_core import predict_tumor_logic
from ui_components import get_custom_css, build_html_outputs, get_empty_states

def predict_and_format(img):
    # Wrapper for handling image input and formatting HTML
    data = predict_tumor_logic(img)
    if not data["is_valid"]:
        return None, None, None, f"<div style='padding:16px; color:#ef4444;'>{data['error']}</div>", "", "", "", ""
    return build_html_outputs(data)

def clear_outputs():
    # Helper to clear UI inputs and outputs
    empty_diag, empty_conf, empty_break, empty_prop, empty_exp = get_empty_states()
    return None, None, None, None, empty_diag, empty_conf, empty_break, empty_prop, empty_exp

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

if __name__ == "__main__":
    demo.launch(share=True, css=get_custom_css(), theme=gr.themes.Base())
