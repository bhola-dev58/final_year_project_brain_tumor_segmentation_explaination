def get_custom_css():
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

def get_empty_states():
    # Helper to return placeholder layout for result cards before any image is uploaded
    empty_diag = f"""
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">AI Diagnosis</div>
        <div>Awaiting Scan Input...</div>
    </div>
    """
    empty_conf = f"""
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Model Confidence</div>
        <div>--</div>
    </div>
    """
    empty_break = f"""
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Prediction Breakdown</div>
        <div>Awaiting Analysis...</div>
    </div>
    """
    empty_prop = f"""
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">Scan Properties</div>
        <div>N/A</div>
    </div>
    """
    empty_exp = f"""
    <div style="color:#64748b; font-size:14px; text-align:center; padding:16px;">
        <div style="margin-bottom:8px; font-weight:600; color:#94a3b8;">AI Explanation</div>
        <div>Awaiting Analysis...</div>
    </div>
    """
    return empty_diag, empty_conf, empty_break, empty_prop, empty_exp

def build_html_outputs(data):
    if not data["is_valid"]:
        return None, None, None, f"<div style='color:red; padding:16px;'>{data['error']}</div>", "", "", "", ""
    
    is_tumor = data["is_tumor"]
    status_text = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"
    status_color = "#ef4444" if is_tumor else "#22c55e"

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
