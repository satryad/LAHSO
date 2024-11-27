import sys

import numpy as np
import pandas as pd
import simpy as sim

from lahso.extract_output import shipment_logs
from lahso.global_variables import *
from lahso.policy_function_improved import make_epsilon_greedy_policy

# import from other python files
from lahso.simulation_module_improved import *

sim_start_time = time.time()  # To measure the runtime
env = sim.Environment()

# Redirect print statements to a file
if print_event_enabled:
    sys.stdout = open(r"csv_output\simulation_logs.txt", "w")

if random_seed:
    print_event("Random seed is enabled")
else:
    print_event(f"Random seed is disabled. Seed value: {random_seed_value}")

print(f"""
Implementation for policy: {policy_name}
Disruption set: {sd}
Episodes: {number_of_simulation}
      """)


# Initiate plot if the training starts from scratch
if start_from_0:
    total_cost_plot = []
    total_reward_plot = []
    last_episode = 0
# In case of continuing training from previous paused training
else:
    total_cost_plot = total_cost_plot_read
    total_reward_plot = total_reward_plot_read
    last_episode = len(total_cost_plot_read)

# Create policy function for RL
policy = make_epsilon_greedy_policy(Q, epsilon, np_actions, mode_ID, policy_name)

# ----- Run the Simulation ----- #
for simulation in range(number_of_simulation):
    eps_start_time = time.time()
    global_vars.reset()

    # Set global variables that need simpy
    global_vars.s_disruption_event = {
        node: env.event() for node in (node_list + mode_list)
    }
    global_vars.actual_carried_shipments = {mode: 0 for mode in mode_list}
    global_vars.actual_itinerary = {req_id: [] for req_id in request_ids}
    global_vars.wait_actions = {req_id: 0 for req_id in request_ids}
    global_vars.reassign_actions = {req_id: 0 for req_id in request_ids}
    global_vars.late_dict = {mode: [0, 0] for mode in mode_list}
    global_vars.current_episode = last_episode + simulation

    current_episode = last_episode + simulation

    # try:
    print(f"Simulation number: {current_episode + 1} starts")
    env = sim.Environment()
    env.process(clock(env, 1440, simulation))

    # Restore possible paths departure time
    possible_paths = possible_paths_ref.copy()

    # Initiate transport modes
    mode_schedule_dict = {}
    for mode in fixed_list:
        name, schedule, capacity, speed, distance, costs = fixed_schedule_dict[mode]
        mode_schedule_dict[mode] = Mode(
            env, name, schedule, capacity, speed, distance, costs
        )
        env.process(mode_schedule_dict[mode].operate())

    # Initiate shipment requests
    shipment_dict = {}
    for req in request_list:
        shipment = Shipment(env, req)
        shipment_dict[req[0]] = shipment

    # initiate rl agent
    rl_module = ReinforcementLearning(
        env, shipment_dict, mode_schedule_dict, Q, gamma, alpha, policy
    )

    # Initiate matching module
    planning = MatchingModule(
        env, mode_schedule_dict, shipment_dict, rl_module, planning_interval
    )
    env.process(planning.planning())

    # Initiate service disruptions
    s_disruption = ServiceDisruption(env, mode_schedule_dict, s_disruption_profile)
    env.process(s_disruption.produce())

    # Initiate demand disruptions
    d_disruption = DemandDisruption(env, shipment_dict, d_disruption_profile)
    env.process(d_disruption.produce())

    # Initiate affected shipment checker
    env.process(affected_request_detection(env, shipment_dict, s_disruption, planning))

    # Cost updating for undelivered shipments
    env.process(
        update_undelivered_shipments(
            env, shipment_dict, simulation_duration, undelivered_penalty
        )
    )

    # Random seed is set according to the simulation order
    if random_seed:
        seed = current_episode
    else:
        seed = random_seed_value
    np.random.seed(seed)

    # Run the simulation until the simulation duration
    env.run(until=simulation_duration)

    # Calculate number of each action for each shipment
    total_wait_action = 0
    total_reassign_action = 0
    for rq in request_ids:
        total_wait_action += global_vars.wait_actions[rq]
        total_reassign_action += global_vars.reassign_actions[rq]

    # Store values for observation throughout multiple simulations
    total_storage_cost_plot.append(global_vars.total_storage_cost)
    total_travel_cost_plot.append(global_vars.total_travel_cost)
    total_handling_cost_plot.append(global_vars.total_handling_cost)
    total_shipment_delay_plot.append(global_vars.total_delay_penalty)
    total_cost_plot.append(global_vars.total_cost)
    total_reward_plot.append(global_vars.total_reward)
    total_late_plot.append(global_vars.total_late_departure)
    total_number_late_plot.append(global_vars.nr_late_departure)
    total_rl_triggers.append(global_vars.rl_triggers)
    total_assigned_rl.append(len(global_vars.rl_assignment))
    total_wait_plot.append(total_wait_action)
    total_reassign_plot.append(total_reassign_action)
    total_shipment = len(request_list)
    total_delivered = len(global_vars.delivered_shipments)
    u_req = total_shipment - total_delivered
    undelivered_requests.append(u_req)
    x.append(simulation + 1)
    print(" ")
    print(f"Simulation number {current_episode + 1} ends")

    if print_output:
        print("\nService disruptions: ", global_vars.s_disruption_triggers, " times")
        print("\nOptimization module is triggred: ", global_vars.om_triggers, " times")
        print(
            "\nReinforcement learning is triggred: ",
            global_vars.rl_triggers,
            " times",
        )
        print("\nTotal late departure: ", global_vars.total_late_departure, " minutes")
        print(
            "Total number of late departure: ",
            global_vars.nr_late_departure,
            " times",
        )
        print(
            "Average late departure: ",
            global_vars.total_late_departure / global_vars.nr_late_departure,
            " minutes",
        )
        print("\nTOTAL COSTS")
        print("----------------------------------------")
        print(f"Total storage cost: {global_vars.total_storage_cost:.2f} EUR")
        print(f"Total handling cost: {global_vars.total_handling_cost:.2f} EUR")
        print(f"Total travel cost: {global_vars.total_travel_cost:.2f} EUR")
        print(f"Total delay penalty: {global_vars.total_delay_penalty:.2f} EUR")
        print(f"Total cost: {global_vars.total_cost:.2f} EUR")
        print("----------------------------------------")

        average_storage_time = np.mean(global_vars.storage_time_list)
        print("\nPERFORMANCE SUMMARY")
        print("----------------------------------------")
        print(f"Average storage time: {average_storage_time / 60:.2f} hours/shipment")
        print(f"Total storage time: {global_vars.total_storage_time / 60:.2f} hours")
        print(
            f"Total delay time: {global_vars.total_shipment_delay // 60:02d} hour(s) {global_vars.total_shipment_delay % 60:02d} minute(s)"
        )
        print("----------------------------------------\n")
        total_shipment = len(request_list)
        total_delivered = len(global_vars.delivered_shipments)

        print(
            f"{total_delivered} shipment are delivered from total {total_shipment} requests"
        )
        undelivered = set(request["Demand_ID"]) - set(global_vars.delivered_shipments)
        print(f"List of undelivered shipment: {undelivered}")

        # Export the episode total cost list
        current_episode = last_episode + simulation
        if extract_shipment_output:
            shipment_logs(
                shipment_dict,
                global_vars.actual_itinerary,
                global_vars.actual_carried_shipments,
                global_vars.shipment_to_rl,
                global_vars.wait_actions,
                global_vars.reassign_actions,
                global_vars.late_dict,
                current_episode,
                sd,
                policy_name,
                number_of_simulation,
            )
    eps_end_time = time.time()  # To measure the runtime
    eps_time = eps_end_time - eps_start_time
    print(f"Episode runtime: {eps_time} seconds")
    # except Exception as e:
    #     print(f"Error in simulation number {simulation + 1}: {e}")

sim_end_time = time.time()  # To measure the runtime
sim_time = sim_end_time - sim_start_time
print(f"Simulation time: {sim_time} seconds")

# Create output dataframe
output = pd.DataFrame(
    {
        "Episode": x,
        "Total Storage Cost": total_storage_cost_plot,
        "Total Travel Cost": total_travel_cost_plot,
        "Total Handling Cost": total_handling_cost_plot,
        "Total Delay Penalty": total_shipment_delay_plot,
        "Total Cost": total_cost_plot,
        "Total Cost": total_cost_plot,
        "Total Reward": total_reward_plot,
        "Total Late Departure": total_late_plot,
        "Number of Late Departure": total_number_late_plot,
        "RL Triggers": total_rl_triggers,
        "Shipment to RL": total_assigned_rl,
        "Undelivered Requests": undelivered_requests,
        "Wait Actions": total_wait_plot,
        "Reassign Actions": total_reassign_plot,
    }
)
if extract_shipment_output:
    output.to_csv(f"{output_path}", index=False)

# Close the redirected stdout (important for file integrity)
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Simulation is finished")
