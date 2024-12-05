import gradio as gr

from lahso.ui.model_implementation_tabs import model_implementation_tabs
from lahso.ui.result_comparison_tabs import results_comparison_tabs
from lahso.ui.training_agent_tabs import training_agent_tabs


def main():
    print("LAHSO UI starting up...")

    training_agent = training_agent_tabs()
    model_implementation = model_implementation_tabs()
    results_comparison = results_comparison_tabs()

    # Main Gradio Interface
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        with gr.Tab("Training An Agent"):
            training_agent.render()

        with gr.Tab("Model Implementation"):
            model_implementation.render()

        with gr.Tab("Results Comparison"):
            results_comparison.render()

    # Launch the app
    app.launch()
