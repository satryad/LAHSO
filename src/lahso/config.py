from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Simulation Settings
    number_of_simulation: int = 25000
    simulation_duration: int = 6 * 7 * 1440
    planning_interval: int = 7 * 1440
    random_seed: bool = True  # False = random seed is the same for each simulation
    # Only applies if random_seed is False to set a certain random seed value
    random_seed_value: int = 0
    print_event_enabled: bool = False  # Print event logs
    print_output: bool = True
    extract_shipment_output: bool = False  # Extract a shipment logs in sv file
    start_from_0: bool = False  # False = continue training from the last saved model
    training: bool = True
    apply_s_disruption: bool = True
    apply_d_disruption: bool = False
    # disruption set (def, S1, S2, S3, S4, S5) according to the last 2 character in the
    #  disruption file name
    sd: str = "Def"
    demand_type: str = "kbest"  # 'kbest', 'planned', or 'default
    solution_pool: int = (
        10  # number of itineraries to generate in the k-best (if necessary)
    )

    # paths
    data_path: Path = Path("Datasets")
    disruption_path: Path = Path("Datasets/Disruption_Profiles")

    # Input file names (fn)
    ## Service Network
    network_fn: str = "Network.csv"
    network_barge_fn: str = "Network_Barge.csv"
    network_train_fn: str = "Network_Train.csv"
    network_truck_fn: str = "Network_Truck.csv"

    ## Service Schedule
    fixed_service_schedule_fn: str = "Fixed Vehicle Schedule.csv"
    truck_schedule_fn: str = "Truck Schedule.csv"

    ## Demand
    request_fn: str = "shipment_requests_200_3w"

    ## Disruptions
    if apply_s_disruption:
        s_disruption_fn: str = f"Service_Disruption_Profile_{sd}.csv"
    else:
        s_disruption_fn: str = "No_Service_Disruption_Profile.csv"

    if apply_d_disruption:
        d_disruption_fn: str = "Request_Disruption_Profile.csv"
    else:
        d_disruption_fn: str = "No_Request_Disruption_Profile.csv"

    ## Possible paths, used for path based optimization
    possible_paths_fn: str = "Possible_Paths.csv"

    ## Costs
    mode_costs_fn: str = "Mode Costs.csv"

    # Ouput Names
    training_output: str = "Training_Output_v2.csv"
    tc_name: str = "total_cost_200_v2.pkl"
    tr_name: str = "total_reward_200_v2.pkl"
    q_name: str = "q_table_200_50000"
    smoothing: int = 300  # for training chart

    # Training Path
    training_path: Path = Path(f"training/{training_output}")
    tc_path: Path = Path(f"training/{tc_name}")
    tr_path: Path = Path(f"training/{tr_name}")

    # Dataset for path user
    # path = ....

    # Implementation Settings
    """
    Policy options:
    gp = greedy policy
    aw = always wait policy
    ar = always reassign policy

    For Training
    eg = epsilon greedy policy
    """
    policy_name: str = "eg"
    q_table_path: Path = Path(f"q_table/{q_name}_eps_v2.pkl")
    output_fn: str = f"{policy_name}_{sd}_{number_of_simulation}.csv"
    output_path: Path = Path(f"csv_output/{output_fn}")


    # Cost Parameters (Manually input)
    storage_cost: int = 1  # EUR/TEU/hour
    delay_penalty: int = 1  # EUR/TEU/hour
    penalty_per_unfulfilled_demand: int = 150000  # For optimization module
    undelivered_penalty: int = 100  # For RL

    # Time Parameter
    handling_time: int = 1  # minute/TEU (for simplification)
    # 2.5 hours from the release time to departure time from terminal
    truck_waiting_time: int = 150
    # Loading only start 1.5 hour before scheduled departure time
    loading_time_window: int = 90
    # Consider a time window from 3.5 hour before scheduled departure to start
    #  applying disruptions
    start_operation: int = 210

    # Learning Agent Parameters
    epsilon: float = 0.05  # for epsilon greedy (training)
    alpha: float = 0.5  # learning rate
    gamma: float = 0.9  # reward discount factor
