from gurobipy import GRB, Model


def fetch_hourly_demand(hour, demand_data):
    return demand_data[demand_data["Announce Time"] == hour]


def update_service_capacities(df, service_id, assigned_volume):
    # Find the row with the given service_id
    service_row = df[df["Service_id"] == service_id]

    # Get the service_ids for the given service_id
    service_ids = service_row["service_ids"].values[0].split(", ")

    # Update the capacities for the given service_id
    for index in service_row.index:
        row_service_ids = df.at[index, "service_ids"].split(", ")
        capacities = list(map(int, df.at[index, "service_capacities"].split(", ")))
        updated_capacities = [
            cap - assigned_volume
            for cap, sid in zip(capacities, row_service_ids, strict=False)
        ]
        df.at[index, "service_capacities"] = ", ".join(map(str, updated_capacities))
        df.at[index, "service_capacity"] = (
            min(updated_capacities) if updated_capacities else 0
        )

    # Get the vehicles associated with the given service_id
    vehicles = service_ids

    # Update the capacities for all rows containing the vehicles
    for vehicle in vehicles:
        vehicle_str = str(vehicle)
        vehicle_rows = df[
            df["service_ids"].str.contains(vehicle_str)
            & (df["Service_id"] != service_id)
        ]
        for index in vehicle_rows.index:
            row_service_ids = df.at[index, "service_ids"].split(", ")
            capacities = list(map(int, df.at[index, "service_capacities"].split(", ")))
            updated_capacities = [
                cap - assigned_volume if vehicle_str == sid else cap
                for cap, sid in zip(capacities, row_service_ids, strict=False)
            ]
            df.at[index, "service_capacities"] = ", ".join(map(str, updated_capacities))
            df.at[index, "service_capacity"] = (
                min(updated_capacities) if updated_capacities else 0
            )

    return df


# Function for rolling horizon to generate K-Best solutions
def service_update(service_original, loading_time, week):
    df = service_original.copy()  # load original path file
    df["first_service_departure"] = df["first_service_departure"] + (24 * 7 * week)
    df["last_service_arrival"] = df["last_service_arrival"] + (24 * 7 * week)
    df["Loading_time"] = loading_time
    df["Week"] = week
    df["Service_id"] = df.index

    return df


def optimization_model(
    demands,
    services,
    storage_cost,
    delay_penalty,
    penalty_per_unfulfilled_demand,
    solution_number,
    time_step,
):
    m = Model("synchromodal_freight_transportation")
    m.setParam("Threads", 0)
    m.setParam("MIPGap", 0.1)
    m.setParam("OutputFlag", 0)
    m.setParam("LogFile", "")
    m.setParam(
        GRB.Param.PoolSolutions, solution_number
    )  # Set the number of solutions to store in the solution pool
    m.setParam(
        GRB.Param.PoolSearchMode, 2
    )  # Set the search mode to focus on finding multiple solutions

    # Define decision variables
    x = m.addVars(demands.index, services.index, vtype=GRB.BINARY, name="x")
    y = m.addVars(demands.index, vtype=GRB.BINARY, name="y")

    # Auxiliary variables for storage and delay time calculations
    storage_hours = m.addVars(demands.index, services.index, lb=0, name="storage_hours")
    delay_hours = m.addVars(demands.index, services.index, lb=0, name="delay_hours")

    m.setObjective(
        sum(
            x[d, s]
            * demands.loc[d, "Volume"]
            * (
                services.loc[s, "total_cost"]
                + storage_hours[d, s] * storage_cost
                + delay_hours[d, s] * delay_penalty
                + services.loc[s, "Loading_cost_at_origin"]
                + services.loc[s, "Unloading_cost_at_destination"]
            )
            for d in demands.index
            for s in services.index
        )
        + penalty_per_unfulfilled_demand * sum(1 - y[d] for d in demands.index),
        GRB.MINIMIZE,
    )

    for d in demands.index:
        demand_row = demands.loc[d]
        # Adjusted fulfillment constraint to allow for demands to be unfulfilled
        m.addConstr(
            sum(x[d, s] for s in services.index) == y[d], f"DemandFulfillment_{d}"
        )
        for s in services.index:
            service_row = services.loc[s]
            # Only proceed if origin and destination match;
            #  otherwise, force x[d, s] to 0
            if (
                service_row["origin"] != demand_row["Origin"]
                or service_row["destination"] != demand_row["Destination"]
            ):
                m.addConstr(x[d, s] == 0, f"match_origin_destination_{d}_{s}")

    for d in demands.index:
        for s in services.index:
            # Constraint to calculate storage time if the demand Release time is before
            #  the first service departure
            m.addConstr(
                storage_hours[d, s]
                == max(
                    0,
                    services.loc[s, "first_service_departure"]
                    - demands.loc[d, "Release Time"]
                    - services.loc[s, "Loading Time"] * demands.loc[d, "Volume"],
                )
                * x[d, s],
                name=f"storage_time_{d}_{s}",
            )

            # Constraint to calculate delay time if the service's last arrival is after
            #  the demand's due time
            m.addConstr(
                delay_hours[d, s]
                == max(
                    0,
                    services.loc[s, "last_service_arrival"]
                    - demands.loc[d, "Due Time"],
                )
                * x[d, s],
                name=f"delay_time_{d}_{s}",
            )

    # Constraint: Do not exceed service capacities
    for s in services.index:
        m.addConstr(
            sum(x[d, s] * demands.loc[d, "Volume"] for d in demands.index)
            <= services.loc[s, "service_capacity"],
            f"ServiceCapacity_{s}",
        )

    for d in demands.index:
        for s in services.index:
            # then the departure time of the service must be at least
            #  min_transshipment_time after the demand's release time.
            m.addConstr(
                x[d, s]
                * (
                    services.loc[s, "first_service_departure"]
                    - demands.loc[d, "Release Time"]
                )
                >= services.loc[s, "Loading Time"] * demands.loc[d, "Volume"] * x[d, s],
                f"TransshipmentTime_{d}_{s}",
            )

    for d in demands.index:
        for s in services.index:
            # Add a constraint that ensures the service departure time is after the
            #  demand release time
            m.addConstr(
                x[d, s] * services.loc[s, "first_service_departure"]
                >= x[d, s] * demands.loc[d, "Release Time"],
                name=f"departure_after_release_{d}_{s}",
            )

    try:
        m.optimize()
        m.update()
        if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            best_solution = []
            all_solutions = []
            best_obj_val = float("inf")
            best_k = -1

            for k in range(min(10, m.SolCount)):
                m.setParam(GRB.Param.SolutionNumber, k)
                obj_val = m.PoolObjVal
                if obj_val < best_obj_val:
                    best_obj_val = obj_val
                    best_k = k

            # Set the model to the best solution
            m.setParam(GRB.Param.SolutionNumber, best_k)
            best_solution_vars = m.getAttr("Xn", x)

            # Process the best solution
            for d in demands.index:
                for s in services.index:
                    if (
                        best_solution_vars[d, s] > 0.5
                    ):  # If demand d is matched with service s
                        assigned_demand = demands.loc[d, "Volume"]
                        services = update_service_capacities(
                            services, s, assigned_demand
                        )
                        best_solution.append(
                            {
                                "Time_Step": time_step,
                                "Service_ID": services.loc[s, "service_ids"],
                                "Service_Capacity": services.loc[s, "service_capacity"],
                                "Service_first_departure": services.loc[
                                    s, "first_service_departure"
                                ],
                                "Service_last_arrival": services.loc[
                                    s, "last_service_arrival"
                                ],
                                "Demand_ID": demands.loc[d, "Demand_ID"],
                                "Containers_Moved": demands.loc[d, "Volume"],
                                "Demand_Announce_time": demands.loc[d, "Announce Time"],
                                "Demand_Release_time": demands.loc[d, "Release Time"],
                                "Demand_Due_time": demands.loc[d, "Due Time"],
                                "service_week": services.loc[s, "Week"],
                                "Storage_hours": storage_hours[d, s].X,
                                "delay_hours": delay_hours[d, s].X,
                                "Number of transshipments": services.loc[
                                    s, "transshipment_time"
                                ],
                                "transportation_cost": best_solution_vars[d, s]
                                * demands.loc[d, "Volume"]
                                * services.loc[s, "total_cost"],
                                "real_objective": best_solution_vars[d, s]
                                * demands.loc[d, "Volume"]
                                * (
                                    services.loc[s, "total_cost"]
                                    + storage_hours[d, s].X * storage_cost
                                    + delay_hours[d, s].X * delay_penalty
                                ),
                                "Potential_Services": s,
                                "Potential_Requests": d,
                                "Solution_Number": best_k
                                + 1,  # Add the solution number
                                "Objective_Value": obj_val,  # Add the objective value
                            }
                        )

            # Process all solutions for logging
            for k in range(min(10, m.SolCount)):
                m.setParam(GRB.Param.SolutionNumber, k)
                obj_val = m.PoolObjVal
                solution_vars = m.getAttr("Xn", x)
                log_entries = []
                for d in demands.index:
                    for s in services.index:
                        if (
                            solution_vars[d, s] > 0.5
                        ):  # If demand d is matched with service s
                            log_entries.append(
                                {
                                    "Time_Step": time_step,
                                    "Service_ID": services.loc[s, "service_ids"],
                                    "Service_week": services.loc[s, "Week"],
                                    "Service_Capacity": services.loc[
                                        s, "service_capacity"
                                    ],
                                    "Service_first_departure": services.loc[
                                        s, "first_service_departure"
                                    ],
                                    "Service_last_arrival": services.loc[
                                        s, "last_service_arrival"
                                    ],
                                    "Demand_ID": demands.loc[d, "Demand_ID"],
                                    "Containers_Moved": demands.loc[d, "Volume"],
                                    "Demand_Announce_time": demands.loc[
                                        d, "Announce Time"
                                    ],
                                    "Demand_Release_time": demands.loc[
                                        d, "Release Time"
                                    ],
                                    "Demand_Due_time": demands.loc[d, "Due Time"],
                                    "Storage_hours": storage_hours[d, s].X,
                                    "delay_hours": delay_hours[d, s].X,
                                    "Number of transshipments": services.loc[
                                        s, "transshipment_time"
                                    ],
                                    "transportation_cost": solution_vars[d, s]
                                    * demands.loc[d, "Volume"]
                                    * services.loc[s, "total_cost"],
                                    "real_objective": solution_vars[d, s]
                                    * demands.loc[d, "Volume"]
                                    * (
                                        services.loc[s, "total_cost"]
                                        + storage_hours[d, s].X * storage_cost
                                        + delay_hours[d, s].X * delay_penalty
                                    ),
                                    "Potential_Services": s,
                                    "Potential_Requests": d,
                                    # Add the solution number
                                    "Solution_Number": k + 1,
                                    # Add the objective value
                                    "Objective_Value": obj_val,
                                }
                            )
                all_solutions.append(log_entries)

            return best_solution, all_solutions, services
        print(f"Model for time step {time_step} is infeasible or unbounded.")
        return [], [], services

    except Exception as e:
        print(f"Exception during optimization at time step {time_step}: {e}")
        return [], [], services


# For planning during simulation
def run_optimization(
    demands, services, storage_cost, delay_penalty, penalty_per_unfulfilled_demand
):
    services["Service_id"] = services.index
    demands["Announce Time"] = demands.index
    num_hours = len(demands)
    all_log_entries = []
    for time_step in range(num_hours + 1):
        # Fetch the demands for the current hour
        current_demands = demands[demands["Announce Time"] == time_step]
        # services = services[services['service_capacity'] >= 0]

        if not current_demands.empty:
            # try:
            # Run the optimization for the current time step
            best_solution, all_solutions, services = optimization_model(
                current_demands,
                services,
                storage_cost,
                delay_penalty,
                penalty_per_unfulfilled_demand,
                1,
                1,
            )

            # Append the results of the current hour to the master log
            all_log_entries.extend(best_solution)
            # except Exception as e:
            #     print(f"Optimization failed for time step {time_step}: {e}")
            #     continue

    return all_log_entries
