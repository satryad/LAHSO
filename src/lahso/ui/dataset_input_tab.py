from pathlib import Path

import gradio as gr

from lahso.config import Config
from lahso.kbest import kbest
from lahso.service_to_path import service_to_path
from lahso.ui.execution_status import ExecutionStatus


def dataset_input_next(
    intermodal_network,
    network_barge,
    network_train,
    network_truck,
    fixed_schedule_service,
    truck_service,
    _demand,
    mode_related_costs,
    storage_cost,
    delay_penalty,
    undelivered_penalty,
    _compute_k_best,
):
    if intermodal_network is None:
        msg = "No Intermodal Network file selected"
        raise gr.Error(msg)
    if network_barge is None:
        msg = "No Barge Network file selected"
        raise gr.Error(msg)
    if network_train is None:
        msg = "No Train Network file selected"
        raise gr.Error(msg)
    if network_truck is None:
        msg = "No Truck Network file selected"
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
    network_barge,
    network_train,
    network_truck,
    fixed_schedule_service,
    truck_service,
    demand,
    mode_related_costs,
    storage_cost,
    delay_penalty,
    undelivered_penalty,
    compute_k_best,
):
    config = Config(
        print_event_enabled=False,
        network_path=Path(intermodal_network),
        network_barge_path=Path(network_barge),
        network_train_path=Path(network_train),
        network_truck_path=Path(network_truck),
        fixed_service_schedule_path=Path(fixed_schedule_service),
        truck_schedule_path=Path(truck_service),
        demand_default_path=Path(demand),
        demand_type="kbest" if compute_k_best else "default",
        mode_costs_path=Path(mode_related_costs),
        storage_cost=int(storage_cost),
        delay_penalty=int(delay_penalty),
        undelivered_penalty=int(undelivered_penalty),
    )
    config.demand_kbest_path = config.demand_default_path.with_stem(
        f"{config.demand_default_path.stem}_kbest"
    )
    service_to_path(config)
    if compute_k_best:
        kbest(config)
    return (
        gr.Textbox(value="Done. Continue to Simulation Settings."),
        gr.Button("Resubmit", interactive=True),
        gr.Tab(interactive=True),
        config,
        gr.Checkbox(value=False),
        gr.File(value=str(config.demand_kbest_path)) if compute_k_best else gr.skip()
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
                    value="Datasets/Network.csv",
                )
                network_barge_input = gr.File(
                    label="Barge Network",
                    file_types=[".csv"],
                    height=100,
                    value=str(Path("Datasets/Network_Barge.csv").absolute()),
                )
                network_train_input = gr.File(
                    label="Train Network",
                    file_types=[".csv"],
                    height=100,
                    value=str(Path("Datasets/Network_Train.csv").absolute()),
                )
                network_truck_input = gr.File(
                    label="Truck Network",
                    file_types=[".csv"],
                    height=100,
                    value=str(Path("Datasets/Network_Truck.csv").absolute()),
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
                compute_k_best_checkbox = gr.Checkbox(
                    label="Compute K-Best Solution", value=True
                )
                dataset_input_next_button = gr.Button(value="Next Step")
                dataset_input_processing_status = gr.Textbox(visible=False)

    dataset_inputs = [
        intermodal_network_input,
        network_barge_input,
        network_train_input,
        network_truck_input,
        fixed_schedule_service_input,
        truck_service_input,
        demand_input,
        mode_related_costs_input,
        storage_cost_input,
        delay_penalty_input,
        undelivered_penalty_input,
        compute_k_best_checkbox,
    ]

    return (
        dataset_input_tab,
        dataset_inputs,
        dataset_input_next_button,
        dataset_input_processing_status,
        compute_k_best_checkbox,
        demand_input,
    )
