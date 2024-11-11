# Simulation Settings
number_of_simulation = 50000
simulation_duration = 6*7*1440
planning_interval = 7*1440
random_seed = True # False = random seed is the same for each simulation
random_seed_value = 0 #Only applies if random_seed is False to set a certain random seed value
print_event_enabled = False # Print event logs
print_output = True
extract_shipment_output = False # Extract a shipment logs in sv file
start_from_0 = False # False = continue training from the last saved model
training = True
apply_s_disruption = True
apply_d_disruption = False
sd = 'Def' #disruption set (def, S1, S2, S3, S4, S5) according to the last 2 character in the disruption file name
demand_type = 'kbest' # 'kbest', 'planned', or 'default
solution_pool = 10 # number of itineraries to generate in the k-best (if necessary)

# paths
data_path = "Datasets"
disruption_path = "Datasets\\Disruption_Profiles"
output_path = "Output"

# Input file names (fn)
## Service Network
network_fn = 'Network.csv'
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
tc_name = 'total_cost_200_new.pkl'
tr_name = 'total_reward_200_new.pkl'
q_name = 'q_table_200_50000_eps_new.pkl'
smoothing = 400  # for training chart

# Training Path
tc_path = f'training\\{tc_name}'
tr_path = f'training\\{tr_name}'

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
policy_name = 'gp'
q_table_path = f'q_table\\{q_name}'
output_path = f'csv_output\\{policy_name}_{sd}_{number_of_simulation}.csv'

