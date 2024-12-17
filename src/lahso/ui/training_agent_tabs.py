import math
from pathlib import Path

import gradio as gr
import pandas as pd

from lahso.config import Config
from lahso.model_input import ModelInput
from lahso.model_train import model_train
from lahso.ui.dataset_input_tab import (
    compute_with_dataset_input,
    dataset_input_next,
    render_dataset_input_tab,
)
from lahso.ui.execution_status import ExecutionStatus

averages = [x * pow(10, y) for y in range(1, 10) for x in range(1, 10)]


# Nested Tabs for Training an Agent
def training_agent_tabs():
    with gr.Blocks() as tabs:
        config = gr.State(Config())
        model_input = gr.State()
        (
            dataset_input_tab,
            dataset_inputs,
            dataset_input_next_button,
            dataset_input_processing_status,
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
                    learning_rate_input = gr.Number(
                        label="Learning Rate",
                        info="(alpha)",
                        value=0.5,
                        precision=5,
                    )
                    exploratory_rate_input = gr.Number(
                        label="Exploratory Rate",
                        info="(epsilon)",
                        value=0.95,
                        precision=5,
                    )
                with gr.Column():
                    gr.Markdown("## Simulation Settings")
                    no_of_simulations_input = gr.Number(
                        label="Number of Simulations",
                        value=50000,
                        precision=0,
                    )
                    simulation_durations_input = gr.Number(
                        label="Simulation Durations",
                        info="in days",
                        value=42,
                        precision=0,
                    )
                    continue_from_prev_training_input = gr.Checkbox(
                        label="Continue from previous training?",
                        value=False,
                    )
                    last_q_table_input = gr.File(
                        label="Last Q-Table",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        visible=False,
                        value=str(
                            Path("q_table/default_q_table_output.pkl").absolute()
                        ),
                    )
                    last_total_cost_input = gr.File(
                        label="Last Total Cost",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        visible=False,
                        value=str(Path("training/default_total_cost_output.pkl").absolute()),
                    )
                    last_reward_input = gr.File(
                        label="Last Reward",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        visible=False,
                        value=str(
                            Path("training/default_total_reward_output.pkl").absolute()
                        ),
                    )
                    continue_from_prev_training_input.change(
                        lambda checked: (
                            gr.File(visible=checked),
                            gr.File(visible=checked),
                            gr.File(visible=checked),
                        ),
                        inputs=[continue_from_prev_training_input],
                        outputs=[
                            last_q_table_input,
                            last_total_cost_input,
                            last_reward_input,
                        ],
                    )
                    simulation_settings_next_button = gr.Button(value="Next Step")
                    simulation_settings_processing_status = gr.Textbox(
                        label="Checking Settings", visible=False
                    )

        with gr.Tab("Training", interactive=False) as training_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        lambda n: f"## Average Total \
                            Cost For {n} Episodes",
                        inputs=[no_of_simulations_input],
                    )
                    cost_plot = gr.LinePlot(
                        pd.DataFrame(),
                        x="Episode",
                        x_title="Training Episode",
                        y="Total Cost",
                        y_title="Total Cost",
                        x_bin=5,
                        y_aggregate="mean",
                    )
                with gr.Column():
                    gr.Markdown(
                        lambda n: f"## Average \
                            Reward For {n} Episodes",
                        inputs=[no_of_simulations_input],
                    )
                    reward_plot = gr.LinePlot(
                        pd.DataFrame(),
                        x="Episode",
                        x_title="Training Episode",
                        y="Total Reward",
                        y_title="Total Reward",
                        x_bin=5,
                        y_aggregate="mean",
                    )
            training_execution_status = gr.State(ExecutionStatus.NOT_STARTED)
            training_execution_generator = gr.State()
            # TODO: debugging tool, remove later
            gr.Textbox(value=lambda x: f"{x}", inputs=[training_execution_status])

        def update_plots(
            config,
            model_input,
            status,
            gen,
        ):
            if status is ExecutionStatus.NOT_STARTED:
                gr.Info("Starting Simulation.", duration=3)
                gen = model_train(config, model_input)
                yield (
                    gr.skip(),
                    gr.skip(),
                    ExecutionStatus.EXECUTING,
                    gen,
                )
            for result in gen:
                if result is None:
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gen,
                    )
                else:
                    data = result[["Episode", "Total Cost", "Total Reward"]]
                    rolling_average = (
                        0
                        if data["Episode"].max() <= 100
                        else math.pow(
                            10, math.floor(math.log10(data["Episode"].max())) - 1
                        )
                    )
                    cost_min = data["Total Cost"].min()
                    cost_max = data["Total Cost"].max()
                    cost_ten_percent = (cost_max - cost_min) / 10.0
                    reward_min = data["Total Reward"].min()
                    reward_max = data["Total Reward"].max()
                    reward_ten_percent = (reward_max - reward_min) / 10.0
                    yield (
                        gr.LinePlot(
                            data,
                            x_bin=rolling_average,
                            y_lim=[
                                cost_min - cost_ten_percent,
                                cost_max + cost_ten_percent,
                            ],
                            y_title=f"Average Total Cost per \
                                {int(rolling_average)} Episodes"
                            if rolling_average > 0
                            else "Total Cost",
                        ),
                        gr.LinePlot(
                            data,
                            x_bin=rolling_average,
                            y_lim=[
                                reward_min - reward_ten_percent,
                                reward_max + reward_ten_percent,
                            ],
                            y_title=f"Average Total Reward per \
                                {int(rolling_average)} Episodes"
                            if rolling_average > 0
                            else "Total Reward",
                        ),
                        gr.skip(),
                        gen,
                    )
            yield (
                gr.skip(),
                gr.skip(),
                ExecutionStatus.FINISHED,
                gen,
            )

        training_tab_select_event = training_tab.select(
            fn=update_plots,
            inputs=[
                config,
                model_input,
                training_execution_status,
                training_execution_generator,
            ],
            outputs=[
                cost_plot,
                reward_plot,
                training_execution_status,
                training_execution_generator,
            ],
            show_progress="minimal",
        )

        def training_finished(status):
            if status is ExecutionStatus.FINISHED:
                gr.Info("Training Finished", duration=10)

        training_tab_select_event.success(
            training_finished,
            inputs=[training_execution_status],
        )

        def cancel_training(status):
            if status is ExecutionStatus.EXECUTING:
                gr.Info(
                    "Training Paused. Return to Training to resume.",
                    duration=3,
                )
                return ExecutionStatus.PAUSED
            return status

        simulation_settings_tab.select(
            fn=cancel_training,
            inputs=[training_execution_status],
            outputs=[training_execution_status],
            cancels=[training_tab_select_event],
        )

        dataset_input_tab.select(
            fn=cancel_training,
            inputs=[training_execution_status],
            outputs=[training_execution_status],
            cancels=[training_tab_select_event],
        )

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=dataset_inputs,
            outputs=[
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
                training_tab,
                training_execution_status,
            ],
        ).then(
            compute_with_dataset_input,
            dataset_inputs,
            [
                dataset_input_processing_status,
                dataset_input_next_button,
                simulation_settings_tab,
                config,
            ],
        )

        def simulation_settings_next():
            return gr.Textbox(visible=True), gr.Tab(interactive=False)

        simulation_settings_next_button_click = simulation_settings_next_button.click(
            simulation_settings_next,
            inputs=[],
            outputs=[simulation_settings_processing_status, training_tab],
        )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            learning_rate_input,
            exploratory_rate_input,
            no_of_simulations_input,
            simulation_durations_input,
            continue_from_prev_training_input,
            last_q_table_input,
            last_total_cost_input,
            last_reward_input,
            config,
        ]

        def training_check_simulation_settings(
            service_disruptions,
            demand_disruptions,
            learning_rate,
            exploratory_rate,
            no_of_simulations,
            simulation_durations,
            continue_from_prev_training,
            last_q_table,
            last_total_cost,
            last_reward,
            config,
        ):
            if service_disruptions is None:
                msg = "No Service Disruptions file selected"
                raise gr.Error(msg)
            if demand_disruptions is None:
                msg = "No Demand Disruptions file selected"
                raise gr.Error(msg)
            if learning_rate <= 0:
                msg = "Learning Rate should be positive"
                raise gr.Error(msg)
            if exploratory_rate <= 0:
                msg = "Exploratory Rate should be positive"
                raise gr.Error(msg)
            if no_of_simulations <= 0:
                msg = "Number of Simulations should be positive"
                raise gr.Error(msg)
            if simulation_durations < 0:
                msg = "Simulation Durations should be non-negative"
                raise gr.Error(msg)
            if last_q_table is None:
                msg = "No Q-Table file selected"
                raise gr.Error(msg)
            if last_total_cost is None:
                msg = "No Last Total Cost file selected"
                raise gr.Error(msg)
            if last_reward is None:
                msg = "No Last Reward file selected"
                raise gr.Error(msg)

            config.service_disruptions = Path(service_disruptions)
            config.demand_disruptions = Path(demand_disruptions)
            config.alpha = learning_rate
            config.epsilon = exploratory_rate
            config.number_of_simulation = no_of_simulations
            # Note that simulation_duration in the config is in minutes, not days
            config.simulation_duration = simulation_durations * 1440
            config.extract_q_table = 1
            config.start_from_0 = not continue_from_prev_training
            if continue_from_prev_training:
                config.q_table_path = Path(last_q_table)
                config.tc_path = Path(last_total_cost)
                config.tr_path = Path(last_reward)
            else:
                config.q_table_path = Path("q_table/default_q_table_output.pkl")
                config.tc_path = Path("training/default_total_cost_output.pkl")
                config.tr_path = Path("training/default_total_reward_output.pkl")

            print(config)

            return (
                gr.Textbox(value="Done. Continue to Training."),
                gr.Tab(interactive=True),
                gr.Button("Resubmit", interactive=True),
                config,
                ModelInput(config),
                ExecutionStatus.NOT_STARTED,
            )

        simulation_settings_next_button_click.then(
            training_check_simulation_settings,
            inputs=simulation_settings,
            outputs=[
                simulation_settings_processing_status,
                training_tab,
                simulation_settings_next_button,
                config,
                model_input,
                training_execution_status,
            ],
            show_progress="minimal",
            scroll_to_output=True,
        )

    return tabs
