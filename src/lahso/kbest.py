import pandas as pd

from lahso.config import (
    data_path,
    network_fn,
    possible_paths_fn,
    request_fn,
    solution_pool,
)
from lahso.model_input import (
    delay_penalty,
    handling_time,
    penalty_per_unfulfilled_demand,
    storage_cost,
)
from lahso.optimization_module import *

# Read datasets
services = pd.read_csv(f"{data_path}\\{possible_paths_fn}", index_col=None)
original_services = services.copy()
demand = pd.read_csv(f"{data_path}\\{request_fn}_default.csv", index_col=None)
network = pd.read_csv(f"{data_path}\\{network_fn}")

# Data pre-processing
network_dict = {i + 1: terminal for i, terminal in enumerate(network["N"])}
reverse_dict = {terminal: id for id, terminal in network_dict.items()}
demand["Origin"] = demand["Origin"].map(network_dict)
demand["Destination"] = demand["Destination"].map(network_dict)
demand.rename(columns={"Announce Time": "Actual Announce Time"}, inplace=True)
demand["Fulfilled"] = False
loading_time = handling_time / 60
services["Loading Time"] = loading_time
original_services["Loading Time"] = loading_time

# Initialize logging
log_columns = [
    "Time_Step",
    "Service_ID",
    "Demand_ID",
    "Containers_Moved",
    "Remaining_Capacity",
]
log_df = pd.DataFrame(columns=log_columns)  # DataFrame to log optimization details
log_step = pd.DataFrame(columns=log_columns)

# Main script to run the optimization periodically and save results
all_log_entries = []
service_counter = 0
unmatched_demand = pd.DataFrame()
log_step = []
for time in demand["Actual Announce Time"].unique():
    log_step = []
    new_demand = demand[demand["Actual Announce Time"] == time]
    temp_demand = pd.concat([new_demand, unmatched_demand])
    temp_demand = temp_demand.reset_index(drop=True)
    temp_demand["Announce Time"] = temp_demand.index
    min_time = temp_demand["Announce Time"].min()
    max_time = temp_demand["Announce Time"].max()
    services = service_update(original_services, loading_time, service_counter)
    for time_step in range(min_time, max_time + 1):
        # Fetch the demands for the current hour
        current_demands = temp_demand[temp_demand["Announce Time"] == time_step]
        if not current_demands.empty:
            try:
                # Run the optimization for the current time step
                best_solution, all_solutions, services = optimization_model(
                    current_demands,
                    services,
                    storage_cost,
                    delay_penalty,
                    penalty_per_unfulfilled_demand,
                    solution_pool,
                    time_step,
                )
                log_step.extend(best_solution)
                for solution in all_solutions:
                    all_log_entries.extend(solution)
            except Exception as e:
                print(f"Optimization failed for time step {time_step}: {e}")
                continue
    # Save unmatched demands for the next iteration
    temp_ids = temp_demand["Demand_ID"]
    log_df = pd.DataFrame(all_log_entries)
    service_counter += 1

print(
    "Optimization completed for all time steps. Results logged to 'optimization_log.csv'."
)

# Postprocessing the ouput for the simulation model input
df_combined = demand.merge(
    log_df[["Demand_ID", "Service_ID", "Solution_Number", "Service_week"]],
    on="Demand_ID",
    how="left",
).fillna(0)
df_combined.Mode = df_combined.Service_ID
df_combined = df_combined[
    [
        "Demand_ID",
        "Origin",
        "Destination",
        "Release Time",
        "Due Time",
        "Volume",
        "Mode",
        "Service_week",
        "Actual Announce Time",
        "Solution_Number",
    ]
]
grouped = (
    df_combined.groupby("Demand_ID")
    .agg(
        {
            "Origin": "first",
            "Destination": "first",
            "Release Time": "first",
            "Due Time": "first",
            "Service_week": "first",
            "Volume": "first",
            "Actual Announce Time": "first",
            "Mode": lambda x: "; ".join(
                f"{item}" for item in list(x)
            ),  # Join the solutions with ', ' and wrap each in quotes
        }
    )
    .reset_index()
)

# Rename the aggregated column
grouped = grouped.rename(
    columns={"Mode": "Solution_List", "Actual Announce Time": "Announce Time"}
)
grouped.to_csv(rf"{data_path}\{request_fn}_kbest_test.csv")
