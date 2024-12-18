import math
from pathlib import Path

import gradio as gr
import pandas as pd

from lahso.config import Config
from lahso.model_implementation import model_implementation
from lahso.model_input import ModelInput
from lahso.ui.dataset_input_tab import (
    compute_with_dataset_input,
    dataset_input_next,
    render_dataset_input_tab,
)
from lahso.ui.execution_status import ExecutionStatus


# Nested Tabs for Model Implementation
def model_implementation_tabs():
    with gr.Blocks() as tabs:
        config = gr.State(Config())
        model_input = gr.State()
        (
            dataset_input_tab,
            dataset_inputs,
            dataset_input_next_button,
            dataset_input_processing_status,
            compute_k_best_checkbox,
            demand_input,
        ) = render_dataset_input_tab()

        with gr.Tab(
            "Simulation Settings", interactive=False
        ) as simulation_settings_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Disruption Settings")
                    service_disruptions_input = gr.File(
                        label="Service Disruptions",
                        file_types=[".csv"],
                        height=100,
                        value=str(
                            Path(
                                "Datasets/Disruption_Profiles/No_Service_Disruption_Profile.csv"
                            ).absolute()
                        ),
                    )
                    demand_disruptions_input = gr.File(
                        label="Demand Disruptions",
                        file_types=[".csv"],
                        height=100,
                        value=str(
                            Path(
                                "Datasets/Disruption_Profiles/No_Request_Disruption_Profile.csv"
                            ).absolute()
                        ),
                    )
                    gr.Markdown("## Learning Agent Settings")
                    p = Path("q_table/default_q_table_output.pkl")
                    q_table_input = gr.File(
                        label="Q-Table",
                        file_types=[".pkl"],
                        height=100,
                        value=str(p) if p.exists() else None,
                    )
                    policy_input = gr.Dropdown(
                        label="Policy",
                        choices=[
                            ("Greedy", "gp"),
                            ("Always Wait", "aw"),
                            ("Always Reassign", "ar"),
                        ],
                    )
                with gr.Column():
                    gr.Markdown("## Simulation Settings")
                    no_of_simulations_input = gr.Number(
                        label="Number of Simulations", value=20, precision=0
                    )
                    simulation_durations_per_episode_input = gr.Number(
                        label="Simulation Durations/Episode",
                        info="in days",
                        value=35,
                        precision=0,
                    )
                    simulation_settings_next_button = gr.Button(value="Next Step")
                    simulation_settings_processing_status = gr.Textbox(
                        label="Checking Settings", visible=False
                    )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            q_table_input,
            policy_input,
            no_of_simulations_input,
            simulation_durations_per_episode_input,
            config,
        ]

        with gr.Tab("Execute Simulation", interactive=False) as execute_simulation_tab:
            with gr.Row():
                # with gr.Column():
                #     gr.Markdown("## Flow Distribution")
                #     flow_distribution_image = gr.Image(
                #         interactive=False, show_label=False
                #     )
                with gr.Column():
                    gr.Markdown(
                        lambda number_of_simulation: f"## Total Cost Per Simulation Episode Over \
                                        {number_of_simulation} Episodes",
                        inputs=[no_of_simulations_input],
                    )
                    execute_simulation_barplot = gr.BarPlot(
                        value=pd.DataFrame(),
                        x="Episode",
                        y="Total Cost",
                        height=400,
                    )
                    simulation_execution_status = gr.State(ExecutionStatus.NOT_STARTED)
                    simulation_execution_generator = gr.State()

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=dataset_inputs,
            outputs=[
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
                execute_simulation_tab,
                simulation_execution_status,
            ],
        ).then(
            compute_with_dataset_input,
            dataset_inputs,
            [
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
                config,
                compute_k_best_checkbox,
                demand_input,
            ],
        )

        def simulation_settings_next():
            return gr.Textbox(visible=True), gr.Tab(interactive=False)

        def check_simulation_settings(
            service_disruptions,
            demand_disruptions,
            q_table,
            policy,
            no_of_simulations,
            simulation_durations_per_episode,
            config,
        ):
            if service_disruptions is None:
                msg = "No Service Disruptions file selected"
                raise gr.Error(msg)
            if demand_disruptions is None:
                msg = "No Demand Disruptions file selected"
                raise gr.Error(msg)
            if no_of_simulations <= 0:
                msg = "Number of Simulations should be positive"
                raise gr.Error(msg)
            if simulation_durations_per_episode < 0:
                msg = "Simulation Durations should be non-negative"
                raise gr.Error(msg)

            config = Config(
                s_disruption_path = Path(service_disruptions),
                d_disruption_path = Path(demand_disruptions),
                number_of_simulation = no_of_simulations,
                # Note that simulation_duration in the config is in minutes, not days
                simulation_duration = simulation_durations_per_episode * 1440,
                start_from_0 = True,
                q_table_path = Path(q_table),
                policy_name = policy,
                extract_shipment_output = True,
            )

            print(config)

            return (
                gr.Textbox(value="Done. Continue to Execute Simulation."),
                gr.Button("Resubmit"),
                gr.Tab(interactive=True),
                config,
                ModelInput(config),
                ExecutionStatus.NOT_STARTED,
            )

        simulation_settings_next_button.click(
            simulation_settings_next,
            inputs=[],
            outputs=[
                simulation_settings_processing_status,
                execute_simulation_tab,
            ],
        ).then(
            check_simulation_settings,
            simulation_settings,
            [
                simulation_settings_processing_status,
                simulation_settings_next_button,
                execute_simulation_tab,
                config,
                model_input,
                simulation_execution_status,
            ],
        )

        def populate_visualisations(
            config,
            model_input,
            status,
            gen,
        ):
            if status is ExecutionStatus.NOT_STARTED:
                gr.Info("Starting Simulation.", duration=3)
                gen = model_implementation(config, model_input)
            for result in gen:
                if status is ExecutionStatus.PAUSED:
                    yield (
                        # gr.skip(),
                        gr.skip(),
                        ExecutionStatus.EXECUTING,
                        gen,
                    )
                if result is None:
                    yield (
                        # gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gen,
                    )
                else:
                    data = result[["Episode", "Total Cost"]]
                    width = 1 + int(math.log10(config.number_of_simulation))
                    # noinspection PyTypeChecker
                    data["Episode"] = data["Episode"].map(
                        lambda x: f"E{str(x).zfill(width)}"  # noqa: B023
                    )
                    cost_min = float(data["Total Cost"].min())
                    cost_max = float(data["Total Cost"].max())
                    cost_ten_percent = (cost_max - cost_min) / 10.0
                    yield (
                        # gr.Image(),
                        gr.BarPlot(
                            value=data,
                            y_lim=[
                                cost_min - cost_ten_percent,
                                cost_max + cost_ten_percent,
                            ],
                        ),
                        gr.skip(),
                        gen,
                    )
            yield (
                # gr.skip(),
                gr.skip(),
                ExecutionStatus.FINISHED,
                gen,
            )

        execute_simulation_tab_select_event = execute_simulation_tab.select(
            fn=populate_visualisations,
            inputs=[
                config,
                model_input,
                simulation_execution_status,
                simulation_execution_generator,
            ],
            outputs=[
                # flow_distribution_image,
                execute_simulation_barplot,
                simulation_execution_status,
                simulation_execution_generator,
            ],
            show_progress="minimal",
        )

        def simulation_finished(status):
            if status is ExecutionStatus.FINISHED:
                gr.Info("Simulation Finished", duration=10)

        execute_simulation_tab_select_event.success(
            simulation_finished,
            inputs=[simulation_execution_status],
        )

        def cancel_simulation_execution(status):
            if status is ExecutionStatus.EXECUTING:
                gr.Info(
                    "Simulation Paused. Return to Execute Simulation to resume.",
                    duration=3,
                )
                return ExecutionStatus.PAUSED
            return status

        gr.on([simulation_settings_tab.select, dataset_input_tab.select],
            fn=cancel_simulation_execution,
            inputs=[simulation_execution_status],
            outputs=[simulation_execution_status],
            cancels=[execute_simulation_tab_select_event],
        )

    return tabs
