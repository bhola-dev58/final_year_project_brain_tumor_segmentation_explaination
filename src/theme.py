import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

def get_clinical_theme() -> Base:
    """
    Creates a bespoke modern clinical studio theme for BrainTumorXAI.
    Combines deep charcoal tones, vibrant orange accents, and crisp typography.
    """
    theme = gr.themes.Base(
        primary_hue=colors.orange,
        secondary_hue=colors.stone,
        neutral_hue=colors.zinc,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_lg,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Plus Jakarta Sans", weights=(400, 500, 600, 700, 800)),
            "system-ui",
            "-apple-system",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("JetBrains Mono", weights=(400, 600)),
            "monospace",
        ),
    )
    
    theme.set(
        # Page & Container
        body_background_fill="#080B0F",
        body_background_fill_dark="#080B0F",
        body_text_color="#F2F2F2",
        body_text_color_dark="#F2F2F2",
        background_fill_primary="#0D1116",
        background_fill_primary_dark="#0D1116",
        background_fill_secondary="#11161C",
        background_fill_secondary_dark="#11161C",
        
        # Block / Card Styles
        block_background_fill="#11161C",
        block_background_fill_dark="#11161C",
        block_border_color="#252C35",
        block_border_color_dark="#252C35",
        block_border_width="1px",
        block_radius="14px",
        block_shadow="0 4px 20px rgba(0, 0, 0, 0.4)",
        
        # Inputs & Forms
        input_background_fill="#0D1116",
        input_background_fill_dark="#0D1116",
        input_border_color="#252C35",
        input_border_color_dark="#252C35",
        input_border_color_focus="#FF7A00",
        input_border_color_focus_dark="#FF7A00",
        input_shadow="none",
        input_radius="10px",
        
        # Buttons
        button_primary_background_fill="#FF7A00",
        button_primary_background_fill_dark="#FF7A00",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_border_color="transparent",
        button_primary_border_color_dark="transparent",
        button_secondary_background_fill="#11161C",
        button_secondary_background_fill_dark="#11161C",
        button_secondary_text_color="#F2F2F2",
        button_secondary_text_color_dark="#F2F2F2",
        button_secondary_border_color="#252C35",
        button_secondary_border_color_dark="#252C35",
        button_transform_hover="translateY(-1px)",
        button_transform_active="translateY(0px)",
        
        # Navigation & Tabs
        checkbox_label_background_fill_selected="#FF7A00",
        checkbox_label_background_fill_selected_dark="#FF7A00",
    )
    
    return theme
