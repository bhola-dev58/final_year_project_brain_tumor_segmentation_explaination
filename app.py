import gradio as gr
from src.dashboard import create_app, get_custom_css

demo = create_app()

if __name__ == "__main__":
    demo.launch(
        share=True,
        css=get_custom_css(),
        theme=gr.themes.Base()
    )
