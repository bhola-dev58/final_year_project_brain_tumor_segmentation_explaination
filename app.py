import gradio as gr
from src.dashboard import create_app, get_custom_css

demo = create_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=get_custom_css(),
        theme=gr.themes.Base()
    )
