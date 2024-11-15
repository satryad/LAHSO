def find_paths_recursive(
    origin,
    destination,
    services,
    loading_time,
    transshipment_time,
    current_path=[],
    current_services=[],
):
    """
    Recursively find all paths from origin to destination from a list of service dictionaries.
    """
    if origin == destination:
        return [{"path": current_path, "services": current_services}]

    paths = []
    possible_services = [service for service in services if service["Origin"] == origin]

    for service in possible_services:
        # Check if this is the first service or if the service is temporally feasible
        if (
            not current_services
            or service["Departure"]
            >= current_services[-1]["Arrival"] + loading_time + transshipment_time
        ):
            next_paths = find_paths_recursive(
                service["Destination"],
                destination,
                services,
                loading_time,
                transshipment_time,
                current_path + [(origin, service["Destination"])],
                current_services + [service],
            )
            paths.extend(next_paths)

    return paths


def calculate_costs_and_emissions(
    paths, tc_barge, tc_train, tc_truck, transshipment_time, storage_cost_per_hour
):
    """
    Calculate total cost and emissions for each path.
    """
    for path in paths:
        total_transport_cost = sum(
            service["Travel Cost"] for service in path["services"]
        )
        transshipment_cost = 0
        n = 0
        if len(path["services"]) > 1:
            for service in path["services"]:
                mode = service["Mode"]
                if mode == "Barge":
                    hc = tc_barge
                elif mode == "Train":
                    hc = tc_train
                elif mode == "Truck":
                    hc = tc_truck

                if n > 0:
                    transshipment_cost += 2 * hc
                else:
                    transshipment_cost += hc
                n += 1
            transshipment_cost -= hc  # remove the unloading in the destination terminal

        storage_cost = 0
        for i in range(1, len(path["services"])):
            waiting_time = (
                path["services"][i]["Departure"]
                - path["services"][i - 1]["Arrival"]
                - transshipment_time
            )
            storage_cost += max(0, waiting_time) * storage_cost_per_hour

        transshipment_times = len(path["services"]) - 1

        total_cost = total_transport_cost + transshipment_cost + storage_cost
        path.update(
            {
                "total_transport_cost": total_transport_cost,
                "transshipment_cost": transshipment_cost,
                "storage_cost": storage_cost,
                "total_cost": total_cost,
                "transshipment_times": transshipment_times,
            }
        )

    return paths


def remove_origin_revisiting_paths(paths):
    """
    Removes paths that revisit the origin after the first node in the path sequence.
    """
    filtered_paths = []
    for path in paths:
        # Get a list of all stops after the first one
        stops_after_first = [step[1] for step in path["path"][1:]]

        # Find the origin of the current path (first node)
        path_origin = path["path"][0][0]

        # If the origin is not revisited, add the path to the filtered list
        if path_origin not in stops_after_first:
            filtered_paths.append(path)

    return filtered_paths


# Pre-processing to add trucks
# Function to get truck travel time
def get_truck_travel_time(truck_df, origin, destination):
    row = truck_df[
        (truck_df["Origin"] == origin) & (truck_df["Destination"] == destination)
    ]
    if not row.empty:
        return row["Travel Time"].values[0]
    return None


# Function to get truck travel cost
def get_truck_travel_cost(truck_df, origin, destination):
    row = truck_df[
        (truck_df["Origin"] == origin) & (truck_df["Destination"] == destination)
    ]
    if not row.empty:
        return row["Travel Cost"].values[0]
    return None


# Function to get service departure time
def get_service_departure(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Departure"].values[0]
    return None


# Function to get service arrival time
def get_service_arrival(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Arrival"].values[0]
    return None


# Function to get service origin
def get_service_origin(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Origin"].values[0]
    return None


# Function to get service destination
def get_service_destination(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Destination"].values[0]
    return None


# Function to get service arrival time
def get_service_cost(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Travel Cost"].values[0]
    return None


def get_service_capacity(data, service_id):
    row = data[data["Service_ID"] == service_id]
    if not row.empty:
        return row["Capacity"].values[0]
    return None


def get_truck_ID(truck_df, origin, destination):
    row = truck_df[
        (truck_df["Origin"] == origin) & (truck_df["Destination"] == destination)
    ]
    if not row.empty:
        return row["Service_ID"].values[0]
    return None  # Optional: return None or handle the case where no match is found


def extract_intermediary_points(path):
    parts = path.split(" -> ")
    intermediaries = [part.split(" to ")[1] for part in parts[:-1]]
    return ",".join(intermediaries)


# # Function to correct transshipment costs
def correct_transshipment_costs(
    prev_trans_cost, prev_mode, barge_hc, train_hc, truck_hc, last_mode
):
    # Define the cost for each mode
    costs = {"Barge": barge_hc, "Train": train_hc}

    if last_mode:
        new_trans_cost = prev_trans_cost - costs.get(prev_mode, 0) + truck_hc
    else:
        new_trans_cost = prev_trans_cost - 2 * costs.get(prev_mode, 0) + 2 * truck_hc
    return new_trans_cost


# Function to add loading and unloading costs
def add_loading_unloading_costs(df, barge_hc, train_hc, truck_hc):
    # Define the cost for each mode
    costs = {"Barge": barge_hc, "Train": train_hc, "Truck": truck_hc}

    # Add new columns based on the first and last service mode
    df["Loading_cost_at_origin"] = df["first_service_mode"].apply(
        lambda mode: costs.get(mode, 0)
    )
    df["Unloading_cost_at_destination"] = df["last_service_mode"].apply(
        lambda mode: costs.get(mode, 0)
    )

    return df
