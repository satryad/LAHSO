import pandas as pd
import numpy as np
from collections import defaultdict
import os
import pickle

# import from other python files
from helper_functions import split_to_sublists, get_first_sublist, remove_first_sublist
from config import *

# Helper functions
def get_distance(row):

    if 'Barge' in row['Service_ID']:
        return network_barge_ref.at[row['Origin'], row['Destination']]
    elif 'Train' in row['Service_ID']:
        return network_train_ref.at[row['Origin'], row['Destination']]
    return network_truck_ref.at[row['Origin'], row['Destination']]


# Datasets input
# Network dataset --> distance between terminals for each mode type
network = pd.read_csv(f"{data_path}\\{network_fn}")
network_barge = pd.read_csv(f"{data_path}\\{network_barge_fn}")
network_train = pd.read_csv(f"{data_path}\\{network_train_fn}")
network_truck = pd.read_csv(f"{data_path}\\{network_truck_fn}")

network_ref = network.set_index(['N'])
network_barge_ref = network_barge.set_index(['N'])
network_train_ref = network_train.set_index(['N'])
network_truck_ref = network_truck.set_index(['N'])

# Create network dictionary and node list
network_dict = {i + 1: terminal for i, terminal in enumerate(network["N"])}
reverse_dict = {terminal: id for id, terminal in network_dict.items()}
node_list = network["N"].tolist()

# Service schedule datasets for
df_fixed_schedule = pd.read_csv(f"{data_path}\\{fixed_service_schedule_fn}")
df_fixed_schedule = df_fixed_schedule.drop(columns=['Travel Cost'])
df_fixed_schedule = df_fixed_schedule.drop(columns=['Mode'])
df_truck_schedule = pd.read_csv(f"{data_path}\\{truck_schedule_fn}")
df_truck_schedule = df_truck_schedule.drop(columns=['Travel Cost'])

# Demand dataset
if demand_type == 'kbest':
    request = pd.read_csv(f"Datasets\\{request_fn}_kbest2.csv")
    request['Solution_List'] = request['Solution_List'].apply(lambda s: [] if s == '0' else split_to_sublists(s))
elif demand_type == 'planned':
    request = pd.read_csv(f"Datasets\\{request_fn}_planned.csv")
    request['Solution_List'] = request['Solution_List'].apply(lambda s: [] if s == '0' else split_to_sublists(s))
else:
    request = pd.read_csv(f"Datasets\\{request_fn}_default.csv")
    request['Solution_List'] = request['Solution_List'].apply(lambda s: [] if s == 0 else split_to_sublists(s))
    request['Origin'] = request['Origin'].map(network_dict)  # Convert terminal ID to terminal name
    request['Destination'] = request['Destination'].map(network_dict)

request['Mode'] = request['Solution_List'].apply(lambda s: [] if s == [] else get_first_sublist(s))
request['Solution_List'] = request['Solution_List'].apply(lambda s: [] if s == [] else remove_first_sublist(s))
request = request[['Demand_ID', 'Origin', 'Destination', 'Release Time', 'Due Time', 'Volume', 'Mode', 'Solution_List',
                   'Announce Time']]

# Disruption dataset
## Service disruption
s_disruption_profile = pd.read_csv(f"{disruption_path}\{s_disruption_fn}")

## Demand disruption
d_disruption_profile = pd.read_csv(f"{disruption_path}\{d_disruption_fn}")

# For optimization module
possible_paths_ref = pd.read_csv(f"{data_path}\\{possible_paths_fn}")

# Cost parameters
mode_costs = pd.read_csv(f"{data_path}\\{mode_costs_fn}")

barge_travel_cost1 = mode_costs['Barge'][0]  # EUR/TEU/hour
barge_travel_cost2 = mode_costs['Barge'][1]  # EUR/TEU/km
barge_handling_cost = mode_costs['Barge'][2]  # EUR/TEU

train_travel_cost1 = mode_costs['Train'][0]  # EUR/TEU/hour
train_travel_cost2 = mode_costs['Train'][1]  # EUR/TEU/km
train_handling_cost = mode_costs['Train'][2]  # EUR/TEU

truck_travel_cost1 = mode_costs['Truck'][0]  # EUR/TEU/hour
truck_travel_cost2 = mode_costs['Truck'][1]  # EUR/TEU/km
truck_handling_cost = mode_costs['Truck'][2]  # EUR/TEU

storage_cost = 1  # EUR/TEU/hour
delay_penalty = 1  # EUR/TEU/hour

penalty_per_unfulfilled_demand = 150000  # For optimization module

undelivered_penalty = 100  # For RL

# Time Parameter
handling_time = 1  # minute/TEU (for simplification)
truck_waiting_time = 150  # 2.5 hours from the release time to departure time from terminal
loading_time_window = 90  # Loading only start 1.5 hour before scheduled departure time
start_operation = 210  # Consider a time window from 3.5 hour before scheduled departure to start applying disruptions

# Learning Agent Parameters
epsilon = 0.05 # for epsilon greedy (training)
alpha = 0.5 # learning rate
gamma = 0.9 # reward discount factor

# ------------------ Data Preprocessing ------------------ #

df_fixed_schedule[['Service_ID', 'Origin', 'Destination']] = df_fixed_schedule[['Service_ID', 'Origin', 'Destination']].astype("string")
df_truck_schedule[['Service_ID', 'Origin', 'Destination']] = df_truck_schedule[['Service_ID', 'Origin', 'Destination']].astype("string") # Added for truck schedule

# Crate a list for reference, and only consider the service disruption. Profile 6 is used as dummy
d_list = s_disruption_profile['Profile'].tolist() + ['Profile6']

# Add a new column 'Distance' to the fixed_vehicle_schedule
df_fixed_schedule['Distance'] = df_fixed_schedule.apply(get_distance, axis=1)
df_truck_schedule['Distance'] = df_truck_schedule.apply(get_distance, axis=1) # Added for truck schedule

# Modifiy the mode schedule for simulation
fixed_list = df_fixed_schedule['Service_ID'].unique().tolist()
truck_list = df_truck_schedule['Service_ID'].unique().tolist()

mode_list = fixed_list + truck_list # List of all modes/servie lines
mode_ID = {mode_list[i]: i + 1 for i in range(len(mode_list))}

# Convert the service schedule dataset to lists and insert the cost params
fixed_schedule = df_fixed_schedule.values.tolist()
for i in range(len(fixed_schedule)):
    if 'Barge' in fixed_schedule[i][0]:
        fixed_schedule[i].append((barge_travel_cost1, barge_travel_cost2, barge_handling_cost))
    else:
        fixed_schedule[i].append((train_travel_cost1, train_travel_cost2, train_handling_cost))

fixed_schedule = [[list[0]] + [(list[1], list[2], list[3]*60)] + list[6:] for list in fixed_schedule] #Create tuple for schedule
fixed_schedule_dict = {fixed_schedule[i][0]: (fixed_schedule[i][0:]) for i in range(len(fixed_schedule))}

truck_schedule = df_truck_schedule.values.tolist()
truck_cost = (truck_travel_cost1, truck_travel_cost2, truck_handling_cost)
for i in range(len(truck_schedule)):
    truck_schedule[i].append(truck_cost)
truck_schedule = [[list[0]] + [(list[1], list[2], list[3]*60)] + list[6:] for list in truck_schedule] #Create tuple for schedule
truck_schedule_dict = {truck_schedule[i][0]: (truck_schedule[i][0:]) for i in range(len(truck_schedule))}

# Modify the request dataset to lists for simulation
request_ids = request['Demand_ID'].tolist()
request_list = request.values.tolist()

# Convert disruption profiles to list
s_disruption_profile = s_disruption_profile[['Profile', 'Impact Type', 'LB Duration', 'UB Duration', 'LB Capacity', 'UB Capacity', 'Location', 'Lambda']]
s_disruption_profile['Location'] = s_disruption_profile['Location'].apply(lambda s: s.split(', '))
s_disruption_profile = s_disruption_profile.values.tolist()
d_disruption_profile = d_disruption_profile[['Profile', 'Impact Type', 'LB Time', 'UB Time', 'LB Volume', 'UB Volume', 'Lambda']]
d_disruption_profile = d_disruption_profile.values.tolist()

possible_paths_ref['Transshipment Terminal(s)'].fillna('0', inplace=True)
possible_paths_ref['Loading_time'] = handling_time

# Convert category to index for RL input
locations = node_list + mode_list
loc_to_index = {location: idx for idx, location in enumerate(locations)}

destinations = node_list
dest_to_index = {destination: idx for idx, destination in enumerate(destinations)}

d_profiles = ['no disruption'] + d_list
d_profile_to_index = {d_profile: idx for idx, d_profile in enumerate(d_profiles)}

# Loading the Q-table
n_actions = 1 + len(mode_list) # action in terminal state (0) and assigning to a service line
np_actions = 2 # wait or reassign

if os.path.exists(q_table_path):
    with open(f'{q_table_path}', 'rb') as f:
        Q = pickle.load(f)
        print(f'{q_table_path} is loaded')

else:
    Q = defaultdict(lambda: np.zeros(n_actions))
    print(f'New Q-table is created')

# Load the last output of the training
if not start_from_0:
    with open(f'{tc_path}', 'rb') as f:
        total_cost_plot_read = pickle.load(f)

    with open(f'{tr_path}', 'rb') as f:
        total_reward_plot_read = pickle.load(f)
    print(len(total_cost_plot_read))
