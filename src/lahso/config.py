from dataclasses import dataclass, field
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
    demand_type: str = "kbest"  # 'kbest', 'planned', or 'default'
    # number of itineraries to generate in the k-best (if necessary)
    solution_pool: int = 10
    # Extract Q-Table every this many episodes, turn off with value 0
    extract_q_table: int = 5000

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
    s_disruption_fn: str = field(init=False)
    d_disruption_fn: str = field(init=False)

    ## Possible paths, used for path based optimization
    possible_paths_fn: str = "Possible_Paths.csv"

    ## Costs
    mode_costs_fn: str = "Mode Costs.csv"

    # Ouput Names
    training_output: str = "Training_Output_v2.csv"
    tc_name: str = "total_cost_200_v2.pkl"
    tr_name: str = "total_reward_200_v2.pkl"
    q_name: str = "q_table_200_50000_eps_test.pkl"
    smoothing: int = 300  # for training chart

    # Training Path
    tc_path: Path = field(init=False)
    tr_path: Path = field(init=False)

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
    q_table_path: Path | None = None

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

    network_path: Path | None = None
    network_barge_path: Path | None = None
    network_train_path: Path | None = None
    network_truck_path: Path | None = None

    possible_paths_path: Path | None = None

    fixed_service_schedule_path: Path | None = None
    truck_schedule_path: Path | None = None
    mode_costs_path: Path | None = None

    demand_default_path: Path | None = None
    demand_planned_path: Path | None = None
    demand_kbest_path: Path | None = None

    s_disruption_path: Path | None = None
    d_disruption_path: Path | None = None

    output_path: Path | None = None

    def __post_init__(self):
        self.s_disruption_fn = (
            f"Service_Disruption_Profile_{self.sd}.csv"
            if self.apply_s_disruption
            else "No_Service_Disruption_Profile.csv"
        )
        self.d_disruption_fn = (
            "Request_Disruption_Profile.csv"
            if self.apply_d_disruption
            else "No_Request_Disruption_Profile.csv"
        )
        self.tc_path = Path(f"training/{self.tc_name}")
        self.tr_path = Path(f"training/{self.tr_name}")
        
        if self.q_table_path is None:
            self.q_table_path = Path(f"q_table/{self.q_name}")

        if self.network_path is None:
            self.network_path = self.data_path / self.network_fn
        if self.network_barge_path is None:
            self.network_barge_path = self.data_path / self.network_barge_fn
        if self.network_train_path is None:
            self.network_train_path = self.data_path / self.network_train_fn
        if self.network_truck_path is None:
            self.network_truck_path = self.data_path / self.network_truck_fn

        if self.possible_paths_path is None:
            self.possible_paths_path = self.data_path / self.possible_paths_fn

        if self.fixed_service_schedule_path is None:
            self.fixed_service_schedule_path = (
                self.data_path / self.fixed_service_schedule_fn
            )
        if self.truck_schedule_path is None:
            self.truck_schedule_path = self.data_path / self.truck_schedule_fn
        if self.mode_costs_path is None:
            self.mode_costs_path = self.data_path / self.mode_costs_fn

        if self.demand_default_path is None:
            self.demand_default_path = self.data_path / f"{self.request_fn}_default.csv"
        if self.demand_planned_path is None:
            self.demand_planned_path = self.data_path / f"{self.request_fn}_planned.csv"
        if self.demand_kbest_path is None:
            self.demand_kbest_path = self.data_path / f"{self.request_fn}_kbest.csv"

        if self.s_disruption_path is None:
            self.s_disruption_path = self.disruption_path / self.s_disruption_fn
        if self.d_disruption_path is None:
            self.d_disruption_path = self.disruption_path / self.d_disruption_fn

        if self.output_path is None:
            self.output_path = Path("csv_output") / f"{self.policy_name}_{self.number_of_simulation}.csv"
