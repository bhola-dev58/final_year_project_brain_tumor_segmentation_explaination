import os
import gradio as gr
from src.dashboard import create_app, get_custom_css
from src.theme import get_clinical_theme

demo = create_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=get_custom_css(),
        theme=get_clinical_theme()
    )
