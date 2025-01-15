"""
The subtabs for the 'Result Comparison' tab. The simpler part of this UI, with subtabs
'Dataset Input' and 'Policy Comparison'. Does some basic datavisualisation based on
logic in bar_chart_plot.py
"""
import gradio as gr
import pandas as pd

from lahso.bar_chart_plot import comparison


def check_inputs(file1, file1_label, file2, file2_label):
    """
    Input Validation.
    """
    if file1 is None or file1_label is None or file2 is None or file2_label is None:
        return gr.Tab(interactive=False)
    return gr.Tab(interactive=True)


def compare_results(file1, file1_label, file2, file2_label):
    """
    Create the bar plots.
    """
    file1_label = f"{file1_label} performs better"
    file2_label = f"{file2_label} performs better"
    df_comparison = comparison(file1, file2, file1_label, file2_label)
    color_map = {}
    color_map[file1_label] = "blue"
    color_map[file2_label] = "orange"
    return (
        gr.BarPlot(value=df_comparison, color_map=color_map),
        gr.BarPlot(value=df_comparison, color_map=color_map),
        gr.BarPlot(value=df_comparison, color_map=color_map),
        gr.BarPlot(value=df_comparison, color_map=color_map),
        gr.BarPlot(value=df_comparison, color_map=color_map),
    )


# Nested Tabs for Results Comparison
def results_comparison_tabs():
    with gr.Blocks() as tabs:
        with gr.Tab("Dataset Input"):
            gr.Markdown("## Output Files")
            with gr.Row():
                with gr.Column():
                    file1_input = gr.File(
                        show_label=False,
                        file_types=[".csv"],
                        type="filepath",
                        height=100,
                    )
                with gr.Column():
                    file1_label_input = gr.Textbox(
                        label="Label In Plot",
                        value="Always Wait",
                    )
            with gr.Row():
                with gr.Column():
                    file2_input = gr.File(
                        show_label=False,
                        file_types=[".csv"],
                        type="filepath",
                        height=100,
                    )
                with gr.Column():
                    file2_label_input = gr.Textbox(
                        label="Label In Plot",
                        value="Greedy Policy",
                    )

        with gr.Tab("Policy Comparison", interactive=False) as comparison_tab:
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("## Comparison For Each Cost Parameter")
                    with gr.Row():
                        storage_cost_plot = gr.BarPlot(
                            pd.DataFrame(),
                            x="Episode",
                            y="Total Storage Cost Delta",
                            color="Total Storage Cost Delta Sign",
                            color_title="",
                        )
                        delay_penalty_plot = gr.BarPlot(
                            pd.DataFrame(),
                            x="Episode",
                            y="Total Delay Penalty Delta",
                            color="Total Delay Penalty Delta Sign",
                            color_title="",
                        )
                    with gr.Row():
                        handling_cost_plot = gr.BarPlot(
                            pd.DataFrame(),
                            x="Episode",
                            y="Total Handling Cost Delta",
                            color="Total Handling Cost Delta Sign",
                            color_title="",
                        )
                        travel_cost_plot = gr.BarPlot(
                            pd.DataFrame(),
                            x="Episode",
                            y="Total Travel Cost Delta",
                            color="Total Travel Cost Delta Sign",
                            color_title="",
                        )
                with gr.Column():
                    gr.Markdown("## Total Cost For Each Simulation Episode")
                    total_cost_plot = gr.BarPlot(
                        pd.DataFrame(),
                        x="Episode",
                        y="Total Cost Delta",
                        color="Total Cost Delta Sign",
                        color_title="",
                    )

        gr.on(
            triggers=[
                file1_input.change,
                file1_label_input.change,
                file2_input.change,
                file2_label_input.change,
            ],
            fn=check_inputs,
            inputs=[file1_input, file1_label_input, file2_input, file2_label_input],
            outputs=[comparison_tab],
        )

        comparison_tab.select(
            fn=compare_results,
            inputs=[file1_input, file1_label_input, file2_input, file2_label_input],
            outputs=[
                storage_cost_plot,
                delay_penalty_plot,
                handling_cost_plot,
                travel_cost_plot,
                total_cost_plot,
            ],
        )

    return tabs
