from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    # Simulation Settings
    number_of_simulation = 1
    simulation_duration = 6 * 7 * 1440
    planning_interval = 7 * 1440
    random_seed = True  # False = random seed is the same for each simulation
    random_seed_value = (
        0  # Only applies if random_seed is False to set a certain random seed value
    )
    print_event_enabled = True  # Print event logs
    print_output = True
    extract_shipment_output = False  # Extract a shipment logs in sv file
    start_from_0 = True  # False = continue training from the last saved model
    training = False
    apply_s_disruption = True
    apply_d_disruption = False
    sd = "Def"  # disruption set (def, S1, S2, S3, S4, S5) according to the last 2 character in the disruption file name
    demand_type = "kbest"  # 'kbest', 'planned', or 'default
    solution_pool = 10  # number of itineraries to generate in the k-best (if necessary)

    # paths
    data_path = Path("Datasets")
    disruption_path = Path("Datasets/Disruption_Profiles")
    output_path = Path("Output")

    # Input file names (fn)
    ## Service Network
    network_fn = "Network.csv"
    network_barge_fn = "Network_Barge.csv"
    network_train_fn = "Network_Train.csv"
    network_truck_fn = "Network_Truck.csv"

    ## Service Schedule
    fixed_service_schedule_fn = "Fixed Vehicle Schedule.csv"
    truck_schedule_fn = "Truck Schedule.csv"

    ## Demand
    request_fn = "shipment_requests_200_3w"

    ## Disruptions
    if apply_s_disruption:
        s_disruption_fn = f"Service_Disruption_Profile_{sd}.csv"
    else:
        s_disruption_fn = "No_Service_Disruption_Profile.csv"

    if apply_d_disruption:
        d_disruption_fn = "Request_Disruption_Profile.csv"
    else:
        d_disruption_fn = "No_Request_Disruption_Profile.csv"

    ## Possible paths, used for path based optimization
    possible_paths_fn = "Possible_Paths.csv"

    ## Costs
    mode_costs_fn = "Mode Costs.csv"

    # Ouput Names
    tc_name = "total_cost_200_test.pkl"
    tr_name = "total_reward_200_test.pkl"
    q_name = "q_table_200_50000"
    smoothing = 400  # for training chart

    # Training Path
    tc_path = Path(f"training/{tc_name}")
    tr_path = Path(f"training/{tr_name}")

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
    policy_name = "eg"
    q_table_path = Path(f"q_table/{q_name}_eps_test.pkl")
    output_path = Path(f"csv_output/{policy_name}_{sd}_{number_of_simulation}.csv")

    # Cost Parameters (Manually input)
    storage_cost = 1  # EUR/TEU/hour
    delay_penalty = 1  # EUR/TEU/hour
    penalty_per_unfulfilled_demand = 150000  # For optimization module
    undelivered_penalty = 100  # For RL

    # Time Parameter
    handling_time = 1  # minute/TEU (for simplification)
    truck_waiting_time = (
        150  # 2.5 hours from the release time to departure time from terminal
    )
    # Loading only start 1.5 hour before scheduled departure time
    loading_time_window = 90
    # Consider a time window from 3.5 hour before scheduled departure to start applying disruptions
    start_operation = 210

    # Learning Agent Parameters
    epsilon = 0.05  # for epsilon greedy (training)
    alpha = 0.5  # learning rate
    gamma = 0.9  # reward discount factor
