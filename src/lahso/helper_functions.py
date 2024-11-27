import numpy as np

# from lahso.config import print_event_enabled


# Functions for input data pre-processing
def split_to_sublists(input_string):
    sub_lists = input_string.split(";")
    sub_lists = [sub_list.strip().split(", ") for sub_list in sub_lists]
    return sub_lists


def get_first_sublist(sub_lists):
    first_in_list = sub_lists[0]
    sub_lists.pop(0)
    return first_in_list


def remove_first_sublist(sub_lists):
    return sub_lists[1:]


def state_to_vector(state_s, state_to_index):
    vector = [0] * len(state_to_index)
    vector[state_to_index[state_s]] = 1
    return vector


# Function to generate simulation logs
def print_event(config, *args, **kwargs):
    if config.print_event_enabled:
        print(*args, **kwargs)


# Function to convert time to minutes
def time_format(minutes):
    return f"{(minutes % 1440) // 60:02d}:{(minutes % 1440) % 60:02d}"


# Function to represent the clock in the simulation
def clock(env, tick, simulation):
    while True:
        current_day = env.now // 1440 + 1
        print_event(" ")
        print_event(f"current day: {current_day}, simulation: {simulation + 1}")
        # print_event(f"List of pending shipment: {set(request['ID']) - set(delivered_shipments)}")
        yield env.timeout(tick)


# Function to identify truck lines
def identify_truck_line(mode_name):
    name = ""
    for letter in mode_name:
        if letter != ".":
            name += letter
        else:
            break
    return name


# Function to update path capacity
def update_service_capacity(df, service_name, new_capacity):
    # Define a function to update capacities within a single row
    def update_row_capacities(
        service_ids, service_capacities, service_name, new_capacity
    ):
        service_ids_list = service_ids.split(", ")
        service_capacities_list = service_capacities.split(", ")

        # Update the capacity for the matching service
        for i, service in enumerate(service_ids_list):
            if service == service_name:
                service_capacities_list[i] = str(new_capacity)

        # Join the updated capacities back into a string
        updated_capacities = ", ".join(service_capacities_list)
        return updated_capacities

    # Apply the function to each row in the dataframe
    df["service_capacities"] = df.apply(
        lambda row: update_row_capacities(
            row["service_ids"], row["service_capacities"], service_name, new_capacity
        ),
        axis=1,
    )
    return df


# Function to # extract the unique od pairs
def unique_origin_destination_pairs(data):
    unique_pairs = data[["Origin", "Destination"]].drop_duplicates().values.tolist()
    return unique_pairs


# Function to sample within the range
def sample_duration_in_range(mean, std_dev, min_val, max_val):
    while True:
        sample = np.random.normal(mean, std_dev)
        if min_val <= sample <= max_val:
            return int(round(sample))
