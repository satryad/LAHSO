import pandas as pd

from lahso.config import (
    data_path,
    fixed_service_schedule_fn,
    possible_paths_fn,
    truck_schedule_fn,
)
from lahso.model_input import (
    barge_handling_cost,
    loading_time_window,
    storage_cost,
    train_handling_cost,
    truck_handling_cost,
)
from lahso.service_to_path_helper import *

# Load the CSV file into a DataFrame
data = pd.read_csv(f"{data_path}\\{fixed_service_schedule_fn}")
truck_df = pd.read_csv(f"{data_path}\\{truck_schedule_fn}")

# Constants
LOADING_TIME = loading_time_window / 60  # Loading time in hours
TRANSSHIPMENT_TIME = 2 * LOADING_TIME  # Loading + Unloading time
TRANSSHIPMENT_COST_BARGE = barge_handling_cost  # Cost per transshipment
TRANSSHIPMENT_COST_TRAIN = train_handling_cost  # Cost per transshipment
TRANSSHIPMENT_COST_TRUCK = truck_handling_cost  # Cost per transshipment
STORAGE_COST_PER_HOUR = storage_cost  # Storage cost per hour of waiting

services_list = data.to_dict("records")

# Example usage
origin = "Delta"
destination = "Venlo"
paths = find_paths_recursive(
    origin, destination, services_list, LOADING_TIME, TRANSSHIPMENT_TIME
)
calculated_paths = calculate_costs_and_emissions(
    paths,
    TRANSSHIPMENT_COST_BARGE,
    TRANSSHIPMENT_COST_TRAIN,
    TRANSSHIPMENT_COST_TRUCK,
    TRANSSHIPMENT_TIME,
    STORAGE_COST_PER_HOUR,
)

# Assuming 'calculated_paths' contains the list of paths with calculated costs and emissions
# Apply the filter to the calculated paths
filtered_paths = remove_origin_revisiting_paths(calculated_paths)

# Get all unique origins and destinations
origins = data["Origin"].unique()
destinations = data["Destination"].unique()

# Container for all filtered paths
all_filtered_paths = []

# Iterate over all unique pairs of origin and destination
for origin in origins:
    for destination in destinations:
        if origin != destination:
            # Calculate all paths for the current origin-destination pair
            paths = find_paths_recursive(
                origin,
                destination,
                data.to_dict("records"),
                LOADING_TIME,
                TRANSSHIPMENT_TIME,
            )
            calculated_paths = calculate_costs_and_emissions(
                paths,
                TRANSSHIPMENT_COST_BARGE,
                TRANSSHIPMENT_COST_TRAIN,
                TRANSSHIPMENT_COST_TRUCK,
                TRANSSHIPMENT_TIME,
                STORAGE_COST_PER_HOUR,
            )

            # Filter out the paths revisiting the origin
            filtered_paths = remove_origin_revisiting_paths(calculated_paths)

            # Add the filtered paths to the container
            all_filtered_paths.extend(filtered_paths)

# Prepare the data for CSV output
data_for_csv = [
    {
        "origin": path["path"][0][0],
        "destination": path["path"][-1][-1],
        "path": " -> ".join(f"{step[0]} to {step[1]}" for step in path["path"]),
        "service_ids": ", ".join(
            str(service["Service_ID"]) for service in path["services"]
        ),
        "first_service_mode": path["services"][0]["Mode"],
        "last_service_mode": path["services"][-1]["Mode"],
        "first_service": path["services"][0]["Service_ID"],
        "first_service_departure": path["services"][0][
            "Departure"
        ],  # First service departure time
        "service_capacities": ", ".join(
            str(service["Capacity"]) for service in path["services"]
        ),
        "service_capacity": min((service["Capacity"]) for service in path["services"]),
        "total_transport_cost": path["total_transport_cost"],
        # 'total_emission': path['total_emission'],
        "transshipment_cost": path["transshipment_cost"],
        "transshipment_time": path["transshipment_times"],
        "storage_time": path["storage_cost"] / STORAGE_COST_PER_HOUR,
        "storage_cost": path["storage_cost"],
        "total_cost": path["total_cost"],
    }
    for path in all_filtered_paths
]

# Create a DataFrame from the filtered data
df_all_filtered_paths = pd.DataFrame(data_for_csv)

# Process the dataset
new_rows = []
df = df_all_filtered_paths
truck_deaprture_window = 2.5
truck_loading_time = 1.5

for index, row in df.iterrows():
    services = row["service_ids"].split(", ")
    if row["transshipment_time"] == 1:
        # Replace the second service with "Truck"
        new_row = row.copy()
        first_service_id = services[0]
        second_service_id = services[1]
        second_mode = second_service_id[:-2]
        updated_transshipment_cost = correct_transshipment_costs(
            row["transshipment_cost"],
            second_mode,
            TRANSSHIPMENT_COST_BARGE,
            TRANSSHIPMENT_COST_TRAIN,
            TRANSSHIPMENT_COST_TRUCK,
            True,
        )
        first_service_arrival = get_service_arrival(data, first_service_id)
        truck_travel_time = get_truck_travel_time(
            truck_df, row["origin"], row["destination"]
        )

        if first_service_arrival is not None and truck_travel_time is not None:
            truck_ID = get_truck_ID(
                truck_df,
                get_service_destination(data, first_service_id),
                row["destination"],
            )
            new_row["service_ids"] = first_service_id + f", {truck_ID}"
            new_row["first_service_departure"] = row["first_service_departure"]
            truck_departure = first_service_arrival + truck_deaprture_window
            truck_arrival = truck_departure + truck_travel_time
            new_row["last_service_arrival"] = truck_arrival
            new_row["storage_time"] = truck_deaprture_window - truck_loading_time
            new_row["storage_cost"] = (
                truck_deaprture_window - truck_loading_time
            ) * STORAGE_COST_PER_HOUR
            new_row["transshipment_cost"] = updated_transshipment_cost
            new_row["total_transport_cost"] = get_service_cost(
                data, first_service_id
            ) + get_truck_travel_cost(
                truck_df,
                get_service_destination(data, first_service_id),
                row["destination"],
            )
            new_row["total_cost"] = (
                new_row["total_transport_cost"]
                + new_row["storage_cost"]
                + row["transshipment_cost"]
            )
            new_row["service_capacity"] = get_service_capacity(data, first_service_id)
            new_row["last_service_mode"] = "Truck"
            new_rows.append(new_row)

    elif row["transshipment_time"] == 2:
        # Replace the second service with "Truck"
        new_row = row.copy()
        first_service_id = services[0]
        second_service_id = services[1]
        third_service_id = services[2]
        second_mode = second_service_id[:-2]
        updated_transshipment_cost = correct_transshipment_costs(
            row["transshipment_cost"],
            second_mode,
            TRANSSHIPMENT_COST_BARGE,
            TRANSSHIPMENT_COST_TRAIN,
            TRANSSHIPMENT_COST_TRUCK,
            False,
        )
        first_service_arrival = get_service_arrival(data, first_service_id)
        truck_travel_time = get_truck_travel_time(
            truck_df,
            get_service_destination(data, first_service_id),
            get_service_origin(data, third_service_id),
        )

        if first_service_arrival is not None and truck_travel_time is not None:
            truck_ID = get_truck_ID(
                truck_df,
                get_service_destination(data, first_service_id),
                get_service_origin(data, third_service_id),
            )
            new_row["service_ids"] = (
                first_service_id + f", {truck_ID}, " + third_service_id
            )
            new_row["first_service_departure"] = row["first_service_departure"]
            truck_departure = (
                first_service_arrival + truck_deaprture_window
            )  # Truck departure one hour after first service arrival
            truck_arrival = truck_departure + truck_travel_time
            third_service_departure = get_service_departure(
                data, third_service_id
            )  # Third service departure
            third_service_arrival = get_service_arrival(data, third_service_id)
            new_row["last_service_arrival"] = third_service_arrival
            new_row["storage_time"] = (
                truck_deaprture_window
                - truck_loading_time
                + third_service_departure
                - truck_arrival
                - LOADING_TIME
            )
            new_row["storage_cost"] = new_row["storage_time"] * STORAGE_COST_PER_HOUR
            new_row["transshipment_cost"] = updated_transshipment_cost
            new_row["total_transport_cost"] = (
                get_service_cost(data, first_service_id)
                + get_service_cost(data, third_service_id)
                + get_truck_travel_cost(
                    truck_df,
                    get_service_destination(data, first_service_id),
                    get_service_origin(data, third_service_id),
                )
            )
            new_row["total_cost"] = (
                new_row["total_transport_cost"]
                + new_row["storage_cost"]
                + row["transshipment_cost"]
            )
            new_row["service_capacity"] = min(
                get_service_capacity(data, first_service_id),
                get_service_capacity(data, third_service_id),
            )
            new_rows.append(new_row)

        # Replace the third service with "Truck"
        new_row = row.copy()
        third_mode = third_service_id[:-2]
        updated_transshipment_cost = correct_transshipment_costs(
            row["transshipment_cost"],
            third_mode,
            TRANSSHIPMENT_COST_BARGE,
            TRANSSHIPMENT_COST_TRAIN,
            TRANSSHIPMENT_COST_TRUCK,
            True,
        )
        second_service_arrival = get_service_arrival(data, second_service_id)
        truck_travel_time = get_truck_travel_time(
            truck_df,
            get_service_origin(data, third_service_id),
            get_service_destination(data, third_service_id),
        )
        if second_service_arrival is not None and truck_travel_time is not None:
            truck_ID = get_truck_ID(
                truck_df,
                get_service_origin(data, third_service_id),
                get_service_destination(data, third_service_id),
            )
            new_row["service_ids"] = (
                first_service_id + ", " + second_service_id + f", {truck_ID}"
            )
            new_row["first_service_departure"] = row["first_service_departure"]
            truck_departure = second_service_arrival + truck_deaprture_window
            truck_arrival = truck_departure + truck_travel_time
            new_row["last_service_arrival"] = truck_arrival
            new_row["storage_time"] = (
                get_service_departure(data, second_service_id)
                - get_service_arrival(data, first_service_id)
                - LOADING_TIME
            ) + (truck_deaprture_window - truck_loading_time)
            new_row["storage_cost"] = new_row["storage_time"] * STORAGE_COST_PER_HOUR
            new_row["transshipment_cost"] = updated_transshipment_cost
            new_row["total_transport_cost"] = (
                get_service_cost(data, first_service_id)
                + get_service_cost(data, second_service_id)
                + get_truck_travel_cost(
                    truck_df,
                    get_service_origin(data, third_service_id),
                    get_service_destination(data, third_service_id),
                )
            )
            new_row["total_cost"] = (
                new_row["total_transport_cost"]
                + new_row["storage_cost"]
                + row["transshipment_cost"]
            )
            new_row["service_capacity"] = min(
                get_service_capacity(data, first_service_id),
                get_service_capacity(data, second_service_id),
            )
            new_row["last_service_mode"] = "Truck"
            new_rows.append(new_row)

# Add the new rows to the dataframe
new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
new_df["Transshipment Terminal(s)"] = new_df["path"].apply(extract_intermediary_points)
new_df["Transshipment Terminal(s)"] = new_df["Transshipment Terminal(s)"].apply(
    lambda s: "0" if s == "" else str(s)
)

# Add loading and unloading costs
add_loading_unloading_costs(
    new_df, TRANSSHIPMENT_COST_BARGE, TRANSSHIPMENT_COST_TRAIN, TRANSSHIPMENT_COST_TRUCK
)

new_df.to_csv(f"{data_path}\\{possible_paths_fn}_test")
