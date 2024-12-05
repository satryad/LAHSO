import gradio as gr


# Nested Tabs for Results Comparison
def results_comparison_tabs():
    with gr.Blocks() as tabs:
        with gr.Tab("Dataset Input"):
            gr.Markdown("## Output Files")
            with gr.Row():
                with gr.Column():
                    _file1_input = gr.File(
                        show_label=False,
                        file_types=[".csv"],
                        type="filepath",
                        height=100,
                    )
                with gr.Column():
                    _file1_label_input = gr.Textbox(
                        label="Label In Plot",
                        value="Greedy Policy",
                    )
            with gr.Row():
                with gr.Column():
                    _file2_input = gr.File(
                        show_label=False,
                        file_types=[".csv"],
                        type="filepath",
                        height=100,
                    )
                with gr.Column():
                    _file2_label_input = gr.Textbox(
                        label="Label In Plot",
                        value="Always Wait",
                    )

        with gr.Tab("Policy Comparison"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Comparison For Each Cost Parameter")
                    with gr.Row():
                        gr.Image(interactive=False, show_label=False)
                        gr.Image(interactive=False, show_label=False)
                    with gr.Row():
                        gr.Image(interactive=False, show_label=False)
                        gr.Image(interactive=False, show_label=False)
                with gr.Column():
                    gr.Markdown("## Total Cost For Each Simulation Episode")
                    gr.Image(interactive=False, show_label=False)

    return tabs
