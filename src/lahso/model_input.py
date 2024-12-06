import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# import from other python files
from lahso.helper_functions import (
    get_first_sublist,
    remove_first_sublist,
    split_to_sublists,
)


class ModelInput:
    def get_distance(self, row):
        if "Barge" in row["Service_ID"]:
            return self.network_barge_ref.at[row["Origin"], row["Destination"]]
        if "Train" in row["Service_ID"]:
            return self.network_train_ref.at[row["Origin"], row["Destination"]]
        return self.network_truck_ref.at[row["Origin"], row["Destination"]]

    def __init__(self, config):
        # Datasets input
        # Network dataset --> distance between terminals for each mode type
        network = pd.read_csv(config.data_path / config.network_fn)
        network_barge = pd.read_csv(config.data_path / config.network_barge_fn)
        network_train = pd.read_csv(config.data_path / config.network_train_fn)
        network_truck = pd.read_csv(config.data_path / config.network_truck_fn)

        network.set_index(["N"])
        self.network_barge_ref = network_barge.set_index(["N"])
        self.network_train_ref = network_train.set_index(["N"])
        self.network_truck_ref = network_truck.set_index(["N"])

        # Create network dictionary and node list
        network_dict = {i + 1: terminal for i, terminal in enumerate(network["N"])}
        {terminal: id for id, terminal in network_dict.items()}
        self.node_list = network["N"].tolist()

        # Service schedule datasets for
        df_fixed_schedule = pd.read_csv(
            config.data_path / config.fixed_service_schedule_fn
        )
        df_fixed_schedule = df_fixed_schedule.drop(columns=["Travel Cost"])
        df_fixed_schedule = df_fixed_schedule.drop(columns=["Mode"])
        df_truck_schedule = pd.read_csv(config.data_path / config.truck_schedule_fn)
        df_truck_schedule = df_truck_schedule.drop(columns=["Travel Cost"])

        # Demand dataset
        if config.demand_type == "kbest":
            self.request = pd.read_csv(
                config.data_path / f"{config.request_fn}_kbest.csv"
            )
            self.request["Solution_List"] = self.request["Solution_List"].apply(
                lambda s: [] if s == "0" else split_to_sublists(s)
            )
        elif config.demand_type == "planned":
            self.request = pd.read_csv(
                config.data_path / f"{config.request_fn}_planned.csv"
            )
            self.request["Solution_List"] = self.request["Solution_List"].apply(
                lambda s: [] if s == "0" else split_to_sublists(s)
            )
        else:
            self.request = pd.read_csv(
                config.data_path / f"{config.request_fn}_default.csv"
            )
            self.request["Solution_List"] = self.request["Solution_List"].apply(
                lambda s: [] if s == 0 else split_to_sublists(s)
            )

        self.request["Mode"] = self.request["Solution_List"].apply(
            lambda s: [] if s == [] else get_first_sublist(s)
        )
        self.request["Solution_List"] = self.request["Solution_List"].apply(
            lambda s: [] if s == [] else remove_first_sublist(s)
        )
        self.request = self.request[
            [
                "Demand_ID",
                "Origin",
                "Destination",
                "Release Time",
                "Due Time",
                "Volume",
                "Mode",
                "Solution_List",
                "Announce Time",
            ]
        ]

        # penalty_per_unfulfilled_demand,

        # Disruption dataset
        ## Service disruption
        self.s_disruption_profile = pd.read_csv(
            config.disruption_path / config.s_disruption_fn
        )

        ## Demand disruption
        self.d_disruption_profile = pd.read_csv(
            config.disruption_path / config.d_disruption_fn
        )

        # For optimization module
        self.possible_paths_ref = pd.read_csv(
            config.data_path / config.possible_paths_fn
        )

        # Cost parameters
        mode_costs = pd.read_csv(config.data_path / config.mode_costs_fn)

        barge_travel_cost1 = mode_costs["Barge"][0]  # EUR/TEU/hour
        barge_travel_cost2 = mode_costs["Barge"][1]  # EUR/TEU/km
        self.barge_handling_cost = mode_costs["Barge"][2]  # EUR/TEU

        train_travel_cost1 = mode_costs["Train"][0]  # EUR/TEU/hour
        train_travel_cost2 = mode_costs["Train"][1]  # EUR/TEU/km
        self.train_handling_cost = mode_costs["Train"][2]  # EUR/TEU

        truck_travel_cost1 = mode_costs["Truck"][0]  # EUR/TEU/hour
        truck_travel_cost2 = mode_costs["Truck"][1]  # EUR/TEU/km
        self.truck_handling_cost = mode_costs["Truck"][2]  # EUR/TEU

        # ------------------ Data Preprocessing ------------------ #

        df_fixed_schedule[["Service_ID", "Origin", "Destination"]] = df_fixed_schedule[
            ["Service_ID", "Origin", "Destination"]
        ].astype("string")
        df_truck_schedule[["Service_ID", "Origin", "Destination"]] = df_truck_schedule[
            ["Service_ID", "Origin", "Destination"]
        ].astype("string")  # Added for truck schedule

        # Crate a list for reference, and only consider the service disruption.
        #  Profile 6 is used as dummy
        d_list = [*self.s_disruption_profile["Profile"].tolist(), "Profile6"]

        # Add a new column 'Distance' to the fixed_vehicle_schedule
        df_fixed_schedule["Distance"] = df_fixed_schedule.apply(
            self.get_distance, axis=1
        )
        df_truck_schedule["Distance"] = df_truck_schedule.apply(
            self.get_distance, axis=1
        )  # Added for truck schedule

        # Modifiy the mode schedule for simulation
        self.fixed_list = df_fixed_schedule["Service_ID"].unique().tolist()
        self.truck_list = df_truck_schedule["Service_ID"].unique().tolist()

        self.mode_list = (
            self.fixed_list + self.truck_list
        )  # List of all modes/servie lines
        self.mode_ID = {self.mode_list[i]: i + 1 for i in range(len(self.mode_list))}

        # Convert the service schedule dataset to lists and insert the cost params
        fixed_schedule = df_fixed_schedule.values.tolist()
        for i in range(len(fixed_schedule)):
            if "Barge" in fixed_schedule[i][0]:
                fixed_schedule[i].append(
                    (barge_travel_cost1, barge_travel_cost2, self.barge_handling_cost)
                )
            else:
                fixed_schedule[i].append(
                    (train_travel_cost1, train_travel_cost2, self.train_handling_cost)
                )

        fixed_schedule = [
            [list[0]] + [(list[1], list[2], list[3] * 60)] + list[6:]
            for list in fixed_schedule
        ]  # Create tuple for schedule
        self.fixed_schedule_dict = {
            fixed_schedule[i][0]: (fixed_schedule[i][0:])
            for i in range(len(fixed_schedule))
        }

        truck_schedule = df_truck_schedule.values.tolist()
        truck_cost = (truck_travel_cost1, truck_travel_cost2, self.truck_handling_cost)
        for i in range(len(truck_schedule)):
            truck_schedule[i].append(truck_cost)
        truck_schedule = [
            [list[0]] + [(list[1], list[2], list[3] * 60)] + list[6:]
            for list in truck_schedule
        ]  # Create tuple for schedule
        self.truck_schedule_dict = {
            truck_schedule[i][0]: (truck_schedule[i][0:])
            for i in range(len(truck_schedule))
        }

        # Modify the self.request dataset to lists for simulation
        self.request_ids = self.request["Demand_ID"].tolist()
        self.request_list = self.request.values.tolist()

        # Convert disruption profiles to list
        self.s_disruption_profile = self.s_disruption_profile[
            [
                "Profile",
                "Impact Type",
                "LB Duration",
                "UB Duration",
                "LB Capacity",
                "UB Capacity",
                "Location",
                "Lambda",
            ]
        ]
        self.s_disruption_profile["Location"] = self.s_disruption_profile[
            "Location"
        ].apply(lambda s: s.split(", "))
        self.s_disruption_profile = self.s_disruption_profile.values.tolist()
        self.d_disruption_profile = self.d_disruption_profile[
            [
                "Profile",
                "Impact Type",
                "LB Time",
                "UB Time",
                "LB Volume",
                "UB Volume",
                "Lambda",
            ]
        ]
        self.d_disruption_profile = self.d_disruption_profile.values.tolist()

        self.possible_paths_ref.fillna({"Transshipment Terminal(s)": "0"}, inplace=True)
        self.possible_paths_ref["Loading_time"] = config.handling_time

        # Convert category to index for RL input
        locations = self.node_list + self.mode_list
        self.loc_to_index = {location: idx for idx, location in enumerate(locations)}

        destinations = self.node_list
        self.dest_to_index = {
            destination: idx for idx, destination in enumerate(destinations)
        }

        d_profiles = ["no disruption", *d_list]
        self.d_profile_to_index = {
            d_profile: idx for idx, d_profile in enumerate(d_profiles)
        }

        # Loading the Q-table
        n_actions = 1 + len(
            self.mode_list
        )  # action in terminal state (0) and assigning to a service line
        self.np_actions = 2  # wait or reassign

        if os.path.exists(config.q_table_path):
            with open(config.q_table_path, "rb") as f:
                self.Q = pickle.load(f)
                print(f"{config.q_table_path} is loaded")

        else:
            self.Q = defaultdict(lambda: np.zeros(n_actions))
            print("New Q-table is created")

        # Load the last output of the training
        # if not config.start_from_0:
        #     with open(config.tc_path, "rb") as f:
        #         self.total_cost_plot_read = pickle.load(f)

        #     with open(config.tr_path, "rb") as f:
        #         self.total_reward_plot_read = pickle.load(f)
        #     print(len(self.total_cost_plot_read))

        # Load the last output of the training (dataframe)
        if not config.start_from_0:
            with open(config.training_path, "rb") as f:
                self.df_training_output_read = pd.read_csv(f)

            print(len(self.df_training_output_read))