from pathlib import Path

import gradio as gr

from lahso.config import Config
from lahso.kbest import kbest
from lahso.service_to_path import service_to_path
from lahso.ui.execution_status import ExecutionStatus


def dataset_input_next(
    intermodal_network,
    fixed_schedule_service,
    truck_service,
    _demand,
    mode_related_costs,
    storage_cost,
    delay_penalty,
    undelivered_penalty,
    _generate_possible_paths,
    _provide_k_best,
):
    if intermodal_network is None:
        msg = "No Intermodal Network file selected"
        raise gr.Error(msg)
    if fixed_schedule_service is None:
        msg = "No Fixed Schedule Service file selected"
        raise gr.Error(msg)
    if truck_service is None:
        msg = "No Truck Service file selected"
        raise gr.Error(msg)
    if truck_service is None:
        msg = "No Demand file selected"
        raise gr.Error(msg)
    if mode_related_costs is None:
        msg = "No Mode Related Costs file selected"
        raise gr.Error(msg)
    if storage_cost < 0:
        msg = "Storage Cost should be non-negative"
        raise gr.Error(msg)
    if delay_penalty < 0:
        msg = "Delay Penalty should be non-negative"
        raise gr.Error(msg)
    if undelivered_penalty < 0:
        msg = "Undelivered Penalty should be non-negative"
        raise gr.Error(msg)
    return (
        gr.Textbox(label="Processing Status", visible=True),
        gr.Button(interactive=False),
        gr.Tab(interactive=False),
        gr.Tab(interactive=False),
        ExecutionStatus.NOT_STARTED,
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
        gr.Textbox(value="Done. Continue to Simulation Settings."),
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
                dataset_input_processing_status = gr.Textbox(visible=False)

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
