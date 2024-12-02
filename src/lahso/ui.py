import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd


# Dummy functions for components
def training_agent():
    time.sleep(5)
    return pd.DataFrame(
        np.array(
            [
                [0, 470000, -127500],
                [50000, 462500, -117500],
                [100000, 457500, -112500],
                [150000, 456500, -113500],
                [500000, 455000, -107500],
            ]
        ),
        columns=["Training Episode", "Average Total Cost", "Average Total Reward"],
    )


def execute_simulation():
    time.sleep(5)
    return pd.DataFrame(
        np.array(
            [
                ["C1", 465000],
                ["C2", 450000],
                ["C3", 455000],
                ["C16", 550000],
                ["C20", 438000],
            ]
        ),
        columns=["Sample Case", "Total Costs"],
    )


def dataset_input_results_comparison(dataset):
    return f"Results comparison using {dataset}"


def policy_comparison_results_comparison(policy_data):
    return f"Comparing policies: {policy_data}"


def load_data(file):
    return file


def dataset_input_next():
    return (
        gr.Label(label="Dataset Processing", visible=True),
        gr.Button(interactive=False),
        gr.Tab(interactive=False),
    )


def compute_with_dataset_input(
    intermodal_network,
    fixed_schedule_service,
    truck_service,
    demand,
    mode_related_costs,
    storage_cost,
    delay_penalty,
    undelivered_penalty,
    generate_possible_paths,
    provide_k_best,
):
    time.sleep(2)
    return (
        gr.Label(value="Done. Continue to Simulation Settings."),
        gr.Button("Resubmit", interactive=True),
        gr.Tab(interactive=True),
    )


def simulation_settings_next():
    return gr.Tab(interactive=False), gr.Button(interactive=False)


def render_dataset_input_tab():
    with gr.Tab("Dataset Input") as dataset_input_tab, gr.Row():
        with gr.Column():
            gr.Markdown("## Network & Demand")
            intermodal_network_input = gr.File(
                label="Intermodal Network",
                file_types=[".csv"],
                height=100,
                value=str(Path("Datasets/Network.csv").absolute()),
            )
            fixed_schedule_service_input = gr.File(
                label="Fixed Schedule Service",
                file_types=[".csv"],
                height=100,
                value=str(Path("Datasets/Fixed Vehicle Schedule.csv").absolute()),
            )
            truck_service_input = gr.File(
                label="Truck Service",
                file_types=[".csv"],
                height=100,
                value=str(Path("Datasets/Truck Schedule.csv").absolute()),
            )
            demand_input = gr.File(
                label="Demand",
                file_types=[".csv"],
                height=100,
                value=str(
                    Path("Datasets/shipment_requests_200_3w_default.csv").absolute()
                ),
            )
        with gr.Column():
            gr.Markdown("## Costs")
            mode_related_costs_input = gr.File(
                label="Mode Related Costs",
                file_types=[".csv"],
                height=100,
                value=str(Path("Datasets/Mode Costs.csv").absolute()),
            )
            storage_cost_input = gr.Number(
                label="Storage Cost", info="Euro/Container/Hour"
            )
            delay_penalty_input = gr.Number(
                label="Delay Penalty", info="Euro/Container/Hour"
            )
            undelivered_penalty_input = gr.Number(
                label="Undelivered Penalty", info="Euro/Container"
            )
            generate_possible_paths_tickbox = gr.Checkbox(
                label="Generate Possible Paths", value=True
            )
            provide_k_best_tickbox = gr.Checkbox(
                label="Provide K-Best Solution", value=True
            )
            dataset_input_next_button = gr.Button(value="Next Step")
            dataset_input_processing_status = gr.Label(visible=False)

    dataset_inputs = [
        intermodal_network_input,
        fixed_schedule_service_input,
        truck_service_input,
        demand_input,
        mode_related_costs_input,
        storage_cost_input,
        delay_penalty_input,
        undelivered_penalty_input,
        generate_possible_paths_tickbox,
        provide_k_best_tickbox,
    ]

    return (
        dataset_input_tab,
        dataset_inputs,
        dataset_input_next_button,
        dataset_input_processing_status,
    )


# Nested Tabs for Training an Agent
def training_agent_tabs():
    def check_simulation_settings(
        service_disruptions,
        demand_disruptions,
        learning_rate,
        exploratory_rate,
        q_table,
        no_of_simulations,
        simulation_durations,
        extract_q_table,
        last_q_table,
        last_total_cost,
        last_reward,
    ):
        time.sleep(1)
        return gr.Tab(interactive=True), gr.Button("Resubmit", interactive=True)

    def update_plots():
        data = training_agent()
        cost_min = data["Average Total Cost"].min()
        cost_max = data["Average Total Cost"].max()
        cost_ten_percent = (cost_max - cost_min) / 10.0
        reward_min = data["Average Total Reward"].min()
        reward_max = data["Average Total Reward"].max()
        reward_ten_percent = (reward_max - reward_min) / 10.0
        # update_timer = gr.Timer(10) # 10 second intervals
        # update_timer.tick(
        #     update_plots, outputs=[cost_plot, reward_plot]
        # )
        return gr.LinePlot(
            data, y_lim=[cost_min - cost_ten_percent, cost_max + cost_ten_percent]
        ), gr.LinePlot(
            data,
            y_lim=[reward_min - reward_ten_percent, reward_max + reward_ten_percent],
        )

    with gr.Blocks() as tabs:
        (
            dataset_input_tab,
            dataset_inputs,
            dataset_input_next_button,
            dataset_input_processing_status,
        ) = render_dataset_input_tab()

        with (
            gr.Tab("Simulation Settings", interactive=False) as simulation_settings_tab,
            gr.Row(),
        ):
            with gr.Column():
                gr.Markdown("## Disruption Settings")
                service_disruptions_input = gr.File(
                    label="Service Disruptions (optional)", file_types=[".csv"]
                )
                demand_disruptions_input = gr.File(
                    label="Demand Disruptions (optional)", file_types=[".csv"]
                )
                gr.Markdown("## Learning Agent Settings")
                learning_rate_input = gr.Number(
                    label="Learning Rate (alpha)", value=0.5
                )
                exploratory_rate_input = gr.Number(
                    label="Exploratory Rate (epsilon)", value=0.95
                )
                q_table_input = gr.File(
                    label="Q-Table name (for exporting Q-Table)",
                    file_types=[".pkl"],
                    type="filepath",
                )
            with gr.Column():
                gr.Markdown("## Simulation Settings")
                no_of_simulations_input = gr.Number(
                    label="Number of Simulations", info="Times"
                )
                simulation_durations_input = gr.Number(
                    label="Simulation Durations", info="Minutes"
                )
                extract_q_table_input = gr.Number(
                    label="Extract Q-Table regularly (optional: 0 = off)",
                    info="Episodes",
                )
                last_q_table_input = gr.File(
                    label="Last Q-Table", file_types=[".pkl"], type="filepath"
                )
                last_total_cost_input = gr.File(
                    label="Last Total Cost", file_types=[".pkl"], type="filepath"
                )
                last_reward_input = gr.File(
                    label="Last Reward", file_types=[".pkl"], type="filepath"
                )
                simulation_settings_next_button = gr.Button(value="Next Step")

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=[],
            outputs=[
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
            ],
        ).then(
            compute_with_dataset_input,
            dataset_inputs,
            [
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
            ],
        )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            learning_rate_input,
            exploratory_rate_input,
            q_table_input,
            no_of_simulations_input,
            simulation_durations_input,
            extract_q_table_input,
            last_q_table_input,
            last_total_cost_input,
            last_reward_input,
        ]

        with gr.Tab("Training", interactive=False) as training_tab:
            rolling_average_input = gr.Number(
                label="Rolling Average", info="Episodes", value="2500", render=False
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Average Total Cost fFor 50000 Episodes")
                    cost_plot = gr.LinePlot(
                        pd.DataFrame(), x="Training Episode", y="Average Total Cost"
                    )
                with gr.Column():
                    gr.Markdown("## Average Reward For 50000 Episodes")
                    reward_plot = gr.LinePlot(
                        pd.DataFrame(), x="Training Episode", y="Average Total Reward"
                    )
            with gr.Row(), gr.Column():
                rolling_average_input.render()

        training_tab.select(update_plots, inputs=[], outputs=[cost_plot, reward_plot])

        simulation_settings_next_button.click(
            simulation_settings_next,
            inputs=[],
            outputs=[training_tab, simulation_settings_next_button],
        ).then(
            check_simulation_settings,
            simulation_settings,
            [training_tab, simulation_settings_next_button],
            show_progress="minimal",
            scroll_to_output=True,
        )

    return tabs


# Nested Tabs for Model Implementation
def model_implementation_tabs():
    def check_simulation_settings(
        service_disruptions,
        demand_disruptions,
        policy,
        learning_agent,
        no_of_simulations,
        simulation_durations_per_episode,
    ):
        time.sleep(1)
        return (
            gr.Label(value="Done. Continue to Execute Simulation."),
            gr.Button("Resubmit", interactive=True),
            gr.Tab(interactive=True),
        )

    with gr.Blocks() as tabs:
        (
            dataset_input_tab,
            dataset_inputs,
            dataset_input_next_button,
            dataset_input_processing_status,
        ) = render_dataset_input_tab()

        with (
            gr.Tab("Simulation Settings", interactive=False) as simulation_settings_tab,
            gr.Row(),
        ):
            with gr.Column():
                gr.Markdown("## Disruption Settings")
                service_disruptions_input = gr.File(
                    label="Service Disruptions (optional)", file_types=[".csv"]
                )
                demand_disruptions_input = gr.File(
                    label="Demand Disruptions (optional)", file_types=[".csv"]
                )
                gr.Markdown("## Learning Agent Settings")
                policy_input = gr.Dropdown(label="Policy", choices=["Greedy"])
                learning_agent_input = gr.File(
                    label="Learning Agent", file_types=[".pkl"]
                )
            with gr.Column():
                gr.Markdown("## Simulation Settings")
                no_of_simulations_input = gr.Number(
                    label="Number of Simulations", info="Times", value="20"
                )
                simulation_durations_per_episode_input = gr.Number(
                    label="Simulation Durations/Episode",
                    info="Minutes",
                    value="50400",
                )
                simulation_settings_next_button = gr.Button(value="Next Step")
                simulation_settings_processing_status = gr.Label(visible=False)

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=[],
            outputs=[
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
            ],
        ).then(
            compute_with_dataset_input,
            dataset_inputs,
            [
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
            ],
        )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            policy_input,
            learning_agent_input,
            no_of_simulations_input,
            simulation_durations_per_episode_input,
        ]

        with gr.Tab("Execute Simulation", interactive=False) as execute_simulation_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Flow Distribution")
                    # gr.Image(value="")
                with gr.Column():
                    gr.Markdown("## Total Cost For Each Simulation Episode")
                    execute_simulation_barplot = gr.BarPlot(
                        pd.DataFrame(), x="Sample Case", y="Total Costs"
                    )

        simulation_settings_next_button.click(
            simulation_settings_next,
            inputs=[],
            outputs=[
                simulation_settings_processing_status,
                simulation_settings_next_button,
                execute_simulation_tab,
            ],
        ).then(
            check_simulation_settings,
            simulation_settings,
            [
                simulation_settings_processing_status,
                simulation_settings_next_button,
                execute_simulation_tab,
            ],
        )

        def populate_barplot():
            data = execute_simulation()
            cost_min = data["Total Costs"].min()
            cost_max = data["Total Costs"].max()
            cost_ten_percent = (cost_max - cost_min) / 10.0
            return gr.BarPlot(
                data, y_lim=[cost_min - cost_ten_percent, cost_max + cost_ten_percent]
            )

        execute_simulation_tab.select(
            populate_barplot, inputs=[], outputs=[execute_simulation_barplot]
        )

    return tabs


# Nested Tabs for Results Comparison
def results_comparison_tabs():
    with gr.Blocks() as tabs:
        with gr.Tab("Dataset Input"):
            dataset_input_results_comparison_input = gr.Textbox(label="Dataset Input")
            dataset_input_results_comparison_output = gr.Textbox(label="Output")
            dataset_input_results_comparison_input.change(
                dataset_input_results_comparison,
                inputs=dataset_input_results_comparison_input,
                outputs=dataset_input_results_comparison_output,
            )

        with gr.Tab("Policy Comparison"):
            policy_comparison_input = gr.Textbox(label="Policy Data")
            policy_comparison_output = gr.Textbox(label="Output")
            policy_comparison_input.change(
                policy_comparison_results_comparison,
                inputs=policy_comparison_input,
                outputs=policy_comparison_output,
            )

    return tabs


def main():
    print("LAHSO UI starting up...")

    training_agent = training_agent_tabs()
    model_implementation = model_implementation_tabs()
    results_comparison = results_comparison_tabs()

    # Main Gradio Interface
    with gr.Blocks() as app:
        # File input above the tabs
        load_data_from_export_input = gr.File(label="Load Data from Export")
        data_from_export_file = gr.State()
        load_data_from_export_input.change(
            load_data,
            inputs=load_data_from_export_input,
            outputs=[data_from_export_file],
        )

        with gr.Tab("Training an Agent"):
            training_agent.render()

        with gr.Tab("Model Implementation"):
            model_implementation.render()

        with gr.Tab("Results Comparison"):
            results_comparison.render()

    # Launch the app
    app.launch()
