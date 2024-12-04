import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import time
from lahso.service_to_path import service_to_path
from lahso.kbest import kbest
from lahso.config import Config
from lahso.model_input import ModelInput
from lahso.model_implementation import model_implementation
from lahso.model_train import model_train


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
    # dataframe available in the Python code
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
        columns=["Sample Case", "Total Cost"],
    )


def dataset_input_results_comparison(dataset):
    return f"Results comparison using {dataset}"


def policy_comparison_results_comparison(policy_data):
    return f"Comparing policies: {policy_data}"


def load_data(file):
    return file


def dataset_input_next(
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
    if intermodal_network is None:
        raise gr.Error("No Intermodal Network file selected")
    if fixed_schedule_service is None:
        raise gr.Error("No Fixed Schedule Service file selected")
    if truck_service is None:
        raise gr.Error("No Truck Service file selected")
    if truck_service is None:
        raise gr.Error("No Demand file selected")
    if mode_related_costs is None:
        raise gr.Error("No Mode Related Costs file selected")
    if storage_cost < 0:
        raise gr.Error("Storage Cost should be non-negative")
    if delay_penalty < 0:
        raise gr.Error("Delay Penalty should be non-negative")
    if undelivered_penalty < 0:
        raise gr.Error("Undelivered Penalty should be non-negative")
    return (
        gr.Label(label="Processing Status", visible=True),
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
    config = Config(
        network_path=Path(intermodal_network),
        fixed_service_schedule_path=Path(fixed_schedule_service),
        truck_schedule_path=Path(truck_service),
        demand_default_path=Path(demand),
        mode_costs_path=Path(mode_related_costs),
        storage_cost=int(storage_cost),
        delay_penalty=int(delay_penalty),
        undelivered_penalty=int(undelivered_penalty),
    )
    if generate_possible_paths:
        service_to_path(config)
    if provide_k_best:
        kbest(config)
    return (
        gr.Label(value="Done. Continue to Simulation Settings."),
        gr.Button("Resubmit", interactive=True),
        gr.Tab(interactive=True),
        config,
    )


def render_dataset_input_tab():
    with gr.Tab("Dataset Input") as dataset_input_tab:
        with gr.Row():
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
                    label="Storage Cost",
                    info="in Euro/Container/Hour",
                    value=1,
                    precision=0,
                )
                delay_penalty_input = gr.Number(
                    label="Delay Penalty",
                    info="in Euro/Container/Hour",
                    value=1,
                    precision=0,
                )
                undelivered_penalty_input = gr.Number(
                    label="Undelivered Penalty",
                    info="in Euro/Container",
                    value=100,
                    precision=0,
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
    def simulation_settings_next():
        return gr.Label(visible=True), gr.Tab(interactive=False)

    def check_simulation_settings(
        service_disruptions,
        demand_disruptions,
        learning_rate,
        exploratory_rate,
        no_of_simulations,
        simulation_durations,
        extract_q_table,
        continue_from_prev_training,
        last_q_table,
        last_total_cost,
        last_reward,
        config,
    ):
        if service_disruptions is None:
            raise gr.Error("No Service Disruptions file selected")
        if demand_disruptions is None:
            raise gr.Error("No Demand Disruptions file selected")
        if learning_rate <= 0:
            raise gr.Error("Learning Rate should be positive")
        if exploratory_rate <= 0:
            raise gr.Error("Exploratory Rate should be positive")
        if no_of_simulations <= 0:
            raise gr.Error("Number of Simulations should be positive")
        if simulation_durations < 0:
            raise gr.Error("Simulation Durations should be non-negative")
        if extract_q_table < 0:
            raise gr.Error("Extract Q Table should be positive")
        if last_q_table is None:
            raise gr.Error("No Q-Table file selected")
        if last_total_cost is None:
            raise gr.Error("No Last Total Cost file selected")
        if last_reward is None:
            raise gr.Error("No Last Reward file selected")

        config.service_disruptions = Path(service_disruptions)
        config.demand_disruptions = Path(demand_disruptions)
        config.alpha = learning_rate
        config.epsilon = exploratory_rate
        config.number_of_simulation = no_of_simulations
        config.simulation_duration = simulation_durations
        config.extract_q_table = extract_q_table
        config.start_from_0 = not continue_from_prev_training
        config.q_table_path = Path(last_q_table)
        config.tc_path = Path(last_total_cost)
        config.tr_path = Path(last_reward)

        return (
            gr.Label(value="Done. Continue to Training."),
            gr.Tab(interactive=True),
            gr.Button("Resubmit", interactive=True),
            config,
            ModelInput(config),
        )

    def update_plots():
        data = training_agent()
        cost_min = data["Average Total Cost"].min()
        cost_max = data["Average Total Cost"].max()
        cost_ten_percent = (cost_max - cost_min) / 10.0
        reward_min = data["Average Total Reward"].min()
        reward_max = data["Average Total Reward"].max()
        reward_ten_percent = (reward_max - reward_min) / 10.0
        return gr.LinePlot(
            data, y_lim=[cost_min - cost_ten_percent, cost_max + cost_ten_percent]
        ), gr.LinePlot(
            data,
            y_lim=[reward_min - reward_ten_percent, reward_max + reward_ten_percent],
        )

    with gr.Blocks() as tabs:
        config = gr.State()
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
                    extract_q_table_input = gr.Number(
                        label="Extract Q-Table regularly",
                        info="in episodes (optional: 0 = off)",
                        value=5000,
                        precision=0,
                    )
                    continue_from_prev_training_input = gr.Checkbox(
                        label="Continue from previous training?"
                    )
                    last_q_table_input = gr.File(
                        label="Last Q-Table",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        value=str(
                            Path("q_table/q_table_200_50000_eps_test.pkl").absolute()
                        ),
                    )
                    last_total_cost_input = gr.File(
                        label="Last Total Cost",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        value=str(Path("training/total_cost_200_test.pkl").absolute()),
                    )
                    last_reward_input = gr.File(
                        label="Last Reward",
                        file_types=[".pkl"],
                        type="filepath",
                        height=100,
                        value=str(
                            Path("training/total_reward_200_test.pkl").absolute()
                        ),
                    )
                    simulation_settings_next_button = gr.Button(value="Next Step")
                    simulation_settings_processing_status = gr.Label(
                        label="Checking Settings", visible=False
                    )

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=dataset_inputs,
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
                config,
            ],
        )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            learning_rate_input,
            exploratory_rate_input,
            no_of_simulations_input,
            simulation_durations_input,
            extract_q_table_input,
            continue_from_prev_training_input,
            last_q_table_input,
            last_total_cost_input,
            last_reward_input,
            config,
        ]

        with gr.Tab("Training", interactive=False) as training_tab:
            rolling_average_input = gr.Number(
                label="Rolling Average", info="Episodes", value="2500", render=False
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Average Total Cost For 50000 Episodes")
                    cost_plot = gr.LinePlot(
                        pd.DataFrame(), x="Training Episode", y="Average Total Cost"
                    )
                with gr.Column():
                    gr.Markdown("## Average Reward For 50000 Episodes")
                    reward_plot = gr.LinePlot(
                        pd.DataFrame(), x="Training Episode", y="Average Total Reward"
                    )
            with gr.Row():
                with gr.Column():
                    rolling_average_input.render()

        training_tab.select(update_plots, inputs=[], outputs=[cost_plot, reward_plot])

        simulation_settings_next_button.click(
            simulation_settings_next,
            inputs=[],
            outputs=[simulation_settings_processing_status, training_tab],
        ).then(
            check_simulation_settings,
            inputs=simulation_settings,
            outputs=[
                simulation_settings_processing_status,
                training_tab,
                simulation_settings_next_button,
                config,
                model_input,
            ],
            show_progress="minimal",
            scroll_to_output=True,
        )

    return tabs


# Nested Tabs for Model Implementation
def model_implementation_tabs():
    def simulation_settings_next():
        return gr.Label(visible=True), gr.Tab(interactive=False)

    def check_simulation_settings(
        service_disruptions,
        demand_disruptions,
        policy,
        no_of_simulations,
        simulation_durations_per_episode,
        config,
    ):
        if service_disruptions is None:
            raise gr.Error("No Service Disruptions file selected")
        if demand_disruptions is None:
            raise gr.Error("No Demand Disruptions file selected")
        if no_of_simulations <= 0:
            raise gr.Error("Number of Simulations should be positive")
        if simulation_durations_per_episode < 0:
            raise gr.Error("Simulation Durations should be non-negative")

        config.service_disruptions = Path(service_disruptions)
        config.demand_disruptions = Path(demand_disruptions)
        config.policy_name = policy
        config.number_of_simulation = no_of_simulations
        config.simulation_duration = simulation_durations_per_episode

        return (
            gr.Label(value="Done. Continue to Execute Simulation."),
            gr.Button("Resubmit"),
            gr.Tab(interactive=True),
            config,
            ModelInput(config),
        )

    with gr.Blocks() as tabs:
        config = gr.State()
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
                    simulation_settings_processing_status = gr.Label(
                        label="Checking Settings", visible=False
                    )

        dataset_input_next_button.click(
            dataset_input_next,
            inputs=dataset_inputs,
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
                config,
            ],
        )

        simulation_settings = [
            service_disruptions_input,
            demand_disruptions_input,
            policy_input,
            no_of_simulations_input,
            simulation_durations_per_episode_input,
            config,
        ]

        with gr.Tab("Execute Simulation", interactive=False) as execute_simulation_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Flow Distribution")
                    flow_distribution_image = gr.Image(
                        interactive=False, show_label=False
                    )
                with gr.Column():
                    gr.Markdown("## Total Cost For Each Simulation Episode")
                    execute_simulation_barplot = gr.BarPlot(
                        pd.DataFrame(), x="Sample Case", y="Total Cost"
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
            ],
        )

        def populate_visualisations(config, model_input):
            data = model_implementation(config, model_input)
            print(data)
            cost_min = int(data["Total Cost"].min())
            cost_max = int(data["Total Cost"].max())
            cost_ten_percent = (cost_max - cost_min) / 10.0
            return [
                gr.Image(),
                gr.BarPlot(
                    data,
                    y_lim=[cost_min - cost_ten_percent, cost_max + cost_ten_percent],
                ),
            ]

        execute_simulation_tab.select(
            populate_visualisations,
            inputs=[config, model_input],
            outputs=[flow_distribution_image, execute_simulation_barplot],
        )

    return tabs


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
                        value="Greedy Policy",
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
