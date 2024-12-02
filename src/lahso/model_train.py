import numpy as np
import simpy as sim

from lahso.config import Config
from lahso.model_input import ModelInput
from lahso.global_variables import *
from lahso.policy_function import make_epsilon_greedy_policy

# import from other python files
from lahso.simulation_module import *


def model_train(config, model_input):
    global_vars = GlobalVars(model_input.possible_paths_ref)
    sim_start_time = time.time()  # To measure the runtime
    env = sim.Environment()

    if config.random_seed:
        print_event(config.print_event_enabled, "Random seed is enabled")
    else:
        print_event(
            config.print_event_enabled,
            f"Random seed is disabled. Seed value: {config.random_seed_value}",
        )

    # Initiate plot if the training starts from scratch
    if config.start_from_0:
        total_cost_plot = []
        total_reward_plot = []
        last_episode = 0
    # In case of continuing training from previous paused training
    else:
        total_cost_plot = model_input.total_cost_plot_read
        total_reward_plot = model_input.total_reward_plot_read
        last_episode = len(model_input.total_cost_plot_read)

    # Create policy function for RL
    policy = make_epsilon_greedy_policy(
        config,
        model_input.Q,
        config.epsilon,
        model_input.np_actions,
        model_input.mode_ID,
        config.policy_name,
    )

    # ----- Run the Simulation ----- #
    for simulation in range(config.number_of_simulation):
        eps_start_time = time.time()
        global_vars.reset(model_input.possible_paths_ref)

        # Set global variables that need simpy
        global_vars.s_disruption_event = {
            node: env.event()
            for node in (model_input.node_list + model_input.mode_list)
        }
        global_vars.actual_carried_shipments = {
            mode: 0 for mode in model_input.mode_list
        }
        global_vars.actual_itinerary = {
            req_id: [] for req_id in model_input.request_ids
        }
        global_vars.wait_actions = {req_id: 0 for req_id in model_input.request_ids}
        global_vars.reassign_actions = {req_id: 0 for req_id in model_input.request_ids}
        global_vars.late_dict = {mode: [0, 0] for mode in model_input.mode_list}
        global_vars.current_episode = last_episode + simulation

        current_episode = last_episode + simulation

        try:
            print(f"Simulation number: {current_episode + 1} starts")
            env = sim.Environment()
            env.process(clock(config.print_event_enabled, env, 1440, simulation))

            # Restore possible paths departure time
            possible_paths = model_input.possible_paths_ref.copy()

            # Initiate transport modes
            mode_schedule_dict = {}
            for mode in model_input.fixed_list:
                name, schedule, capacity, speed, distance, costs = (
                    model_input.fixed_schedule_dict[mode]
                )
                mode_schedule_dict[mode] = Mode(
                    env,
                    name,
                    schedule,
                    capacity,
                    speed,
                    distance,
                    costs,
                    config,
                    model_input,
                    global_vars,
                )
                env.process(mode_schedule_dict[mode].operate())

            # Initiate shipment requests
            shipment_dict = {}
            for req in model_input.request_list:
                shipment = Shipment(env, req, config, model_input, global_vars)
                shipment_dict[req[0]] = shipment

            # initiate rl agent
            rl_module = ReinforcementLearning(
                env,
                shipment_dict,
                mode_schedule_dict,
                model_input.Q,
                config.gamma,
                config.alpha,
                policy,
                config,
                model_input,
                global_vars,
            )

            # Initiate matching module
            planning = MatchingModule(
                env,
                mode_schedule_dict,
                shipment_dict,
                rl_module,
                config.planning_interval,
                config,
                model_input,
                global_vars,
            )
            env.process(planning.planning())

            # Initiate service disruptions
            s_disruption = ServiceDisruption(
                env,
                mode_schedule_dict,
                model_input.s_disruption_profile,
                config,
                model_input,
                global_vars,
            )
            env.process(s_disruption.produce())

            # Initiate demand disruptions
            d_disruption = DemandDisruption(
                env,
                shipment_dict,
                model_input.d_disruption_profile,
                config,
                global_vars,
            )
            env.process(d_disruption.produce())

            # Initiate affected shipment checker
            env.process(
                affected_request_detection(
                    env,
                    shipment_dict,
                    s_disruption,
                    planning,
                    config,
                    model_input,
                    global_vars,
                )
            )

            # Cost updating for undelivered shipments
            env.process(
                update_undelivered_shipments(
                    env,
                    shipment_dict,
                    config.simulation_duration,
                    config.undelivered_penalty,
                    config,
                    global_vars,
                )
            )

            # Random seed is set according to the simulation order
            seed = current_episode if config.random_seed else config.random_seed_value
            np.random.seed(seed)

            # Run the simulation until the simulation duration
            env.run(until=config.simulation_duration)

            # Calculate number of each action for each shipment
            total_wait_action = 0
            total_reassign_action = 0
            for rq in model_input.request_ids:
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
            total_shipment = len(model_input.request_list)
            total_delivered = len(global_vars.delivered_shipments)
            u_req = total_shipment - total_delivered
            undelivered_requests.append(u_req)
            x.append(simulation + 1)
            print(" ")
            print(f"Simulation number {current_episode + 1} ends")

            if config.print_output:
                print(
                    "\nService disruptions: ",
                    global_vars.s_disruption_triggers,
                    " times",
                )
                print(
                    "\nOptimization module is triggred: ",
                    global_vars.om_triggers,
                    " times",
                )
                print(
                    "\nReinforcement learning is triggred: ",
                    global_vars.rl_triggers,
                    " times",
                )
                print(
                    "\nTotal late departure: ",
                    global_vars.total_late_departure,
                    " minutes",
                )
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
                print(
                    f"Average storage time: {average_storage_time / 60:.2f} hours/shipment"
                )
                print(
                    f"Total storage time: {global_vars.total_storage_time / 60:.2f} hours"
                )
                print(
                    f"Total delay time: {global_vars.total_shipment_delay // 60:02d} hour(s) {global_vars.total_shipment_delay % 60:02d} minute(s)"
                )
                print("----------------------------------------\n")
                total_shipment = len(model_input.request_list)
                total_delivered = len(global_vars.delivered_shipments)

                print(
                    f"{total_delivered} shipment are delivered from total {total_shipment} requests"
                )
                undelivered = set(model_input.request["Demand_ID"]) - set(
                    global_vars.delivered_shipments
                )
                print(f"List of undelivered shipment: {undelivered}")

                # Export the episode total cost list
                current_episode = last_episode + simulation
                with open(config.tc_path, "wb") as f:
                    pickle.dump(total_cost_plot, f)
                    print(f"Total cost per episode is exported as {config.tc_name}")
                with open(config.tr_path, "wb") as f:
                    pickle.dump(total_reward_plot, f)
                    print(f"Total reward per episode is exported as {config.tr_name}")
                with open(config.q_table_path, "wb") as f:
                    pickle.dump(dict(model_input.Q), f)
                    print(f"Total q_table is exported as {config.q_name}")
                if current_episode % 5000 == 0:
                    print_event(
                        config.print_event_enabled,
                        f"Q-table is saved as {config.q_name}_{current_episode}_eps.pkl",
                    )
                    with open(
                        f"q_table/{config.q_name}_{current_episode}_eps.pkl", "wb"
                    ) as f:
                        pickle.dump(dict(model_input.Q), f)
            eps_end_time = time.time()  # To measure the runtime
            eps_time = eps_end_time - eps_start_time
            print(f"Episode runtime: {eps_time} seconds")
        except Exception as e:
            print(f"Error in simulation number {simulation + 1}: {e}")

    sim_end_time = time.time()  # To measure the runtime
    sim_time = sim_end_time - sim_start_time
    print(f"Simulation time: {sim_time} seconds")


def main():
    config = Config()
    model_input = ModelInput(config)
    model_train(config, model_input)
