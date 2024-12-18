import sys
import time

import numpy as np
import simpy as sim

from lahso.config import Config
from lahso.extract_output import shipment_logs
from lahso.global_variables import AggregateStatistics, SimulationVars
from lahso.helper_functions import clock, print_event
from lahso.model_input import ModelInput
from lahso.policy_function import make_epsilon_greedy_policy
from lahso.simulation_module import (
    DemandDisruption,
    MatchingModule,
    Mode,
    ReinforcementLearning,
    ServiceDisruption,
    Shipment,
    affected_request_detection,
    update_undelivered_shipments,
)


def model_implementation(config, model_input):
    statistics = AggregateStatistics()

    simulation_vars = SimulationVars(model_input.possible_paths_ref)
    sim_start_time = time.time()  # To measure the runtime
    env = sim.Environment()

    # Redirect print statements to a file
    if config.print_event_enabled:
        # Ignoring lint complaint about opening file in a context (with statement)
        sys.stdout = open(r"csv_output\simulation_logs.txt", "w")  # noqa: SIM115

    if config.random_seed:
        print_event(config.print_event_enabled, "Random seed is enabled")
    else:
        print_event(
            config.print_event_enabled,
            f"Random seed is disabled. Seed value: {config.random_seed_value}",
        )

    print(f"""
    Implementation for policy: {config.policy_name}
    Disruption set: {config.sd}
    Episodes: {config.number_of_simulation}
        """)

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
        simulation_vars.reset(model_input.possible_paths_ref)

        # Set global variables that need simpy
        simulation_vars.s_disruption_event = {
            node: env.event()
            for node in (model_input.node_list + model_input.mode_list)
        }
        simulation_vars.actual_carried_shipments = {
            mode: 0 for mode in model_input.mode_list
        }
        simulation_vars.actual_itinerary = {
            req_id: [] for req_id in model_input.request_ids
        }
        simulation_vars.wait_actions = {req_id: 0 for req_id in model_input.request_ids}
        simulation_vars.reassign_actions = {
            req_id: 0 for req_id in model_input.request_ids
        }
        simulation_vars.late_dict = {mode: [0, 0] for mode in model_input.mode_list}
        simulation_vars.current_episode = last_episode + simulation

        current_episode = last_episode + simulation

        try:
            print_event(config, f"Simulation number: {current_episode + 1} starts")
            env = sim.Environment()
            env.process(clock(config.print_event_enabled, env, 1440, simulation))

            # Restore possible paths departure time
            model_input.possible_paths_ref.copy()

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
                    simulation_vars,
                )
                env.process(mode_schedule_dict[mode].operate())

            # Initiate shipment requests
            shipment_dict = {}
            for req in model_input.request_list:
                shipment = Shipment(env, req, config, model_input, simulation_vars)
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
                simulation_vars,
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
                simulation_vars,
            )
            env.process(planning.planning())

            # Initiate service disruptions
            s_disruption = ServiceDisruption(
                env,
                mode_schedule_dict,
                model_input.s_disruption_profile,
                config,
                model_input,
                simulation_vars,
            )
            env.process(s_disruption.produce())

            # Initiate demand disruptions
            d_disruption = DemandDisruption(
                env,
                shipment_dict,
                model_input.d_disruption_profile,
                config,
                simulation_vars,
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
                    simulation_vars,
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
                    simulation_vars,
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
                total_wait_action += simulation_vars.wait_actions[rq]
                total_reassign_action += simulation_vars.reassign_actions[rq]

            # Store values for observation throughout multiple simulations
            statistics.total_storage_cost_plot.append(
                simulation_vars.total_storage_cost
            )
            statistics.total_travel_cost_plot.append(simulation_vars.total_travel_cost)
            statistics.total_handling_cost_plot.append(
                simulation_vars.total_handling_cost
            )
            statistics.total_shipment_delay_plot.append(
                simulation_vars.total_delay_penalty
            )
            total_cost_plot.append(simulation_vars.total_cost)
            total_reward_plot.append(simulation_vars.total_reward)
            statistics.total_late_plot.append(simulation_vars.total_late_departure)
            statistics.total_number_late_plot.append(simulation_vars.nr_late_departure)
            statistics.total_rl_triggers.append(simulation_vars.rl_triggers)
            statistics.total_assigned_rl.append(len(simulation_vars.rl_assignment))
            statistics.total_wait_plot.append(total_wait_action)
            statistics.total_reassign_plot.append(total_reassign_action)
            total_shipment = len(model_input.request_list)
            total_delivered = len(simulation_vars.delivered_shipments)
            u_req = total_shipment - total_delivered
            statistics.undelivered_requests.append(u_req)
            statistics.x.append(simulation + 1)
            print(" ")
            print(f"Simulation number {current_episode + 1} ends")

            if config.print_output:
                print(
                    "\nService disruptions: ",
                    simulation_vars.s_disruption_triggers,
                    " times",
                )
                print(
                    "\nOptimization module is triggred: ",
                    simulation_vars.om_triggers,
                    " times",
                )
                print(
                    "\nReinforcement learning is triggred: ",
                    simulation_vars.rl_triggers,
                    " times",
                )
                print(
                    "\nTotal late departure: ",
                    simulation_vars.total_late_departure,
                    " minutes",
                )
                print(
                    "Total number of late departure: ",
                    simulation_vars.nr_late_departure,
                    " times",
                )
                average_late_departure = (
                    simulation_vars.total_late_departure
                    / simulation_vars.nr_late_departure
                    if simulation_vars.nr_late_departure
                    else 0
                )
                print("Average late departure: ", average_late_departure, " minutes")
                print("\nTOTAL COSTS")
                print("----------------------------------------")
                print(
                    f"Total storage cost: {simulation_vars.total_storage_cost:.2f} EUR"
                )
                print(
                    f"Total handling cost: {simulation_vars.total_handling_cost:.2f} \
                    EUR"
                )
                print(f"Total travel cost: {simulation_vars.total_travel_cost:.2f} EUR")
                print(
                    f"Total delay penalty: {simulation_vars.total_delay_penalty:.2f} \
                    EUR"
                )
                print(f"Total cost: {simulation_vars.total_cost:.2f} EUR")
                print("----------------------------------------")

                average_storage_time = np.mean(simulation_vars.storage_time_list)
                print("\nPERFORMANCE SUMMARY")
                print("----------------------------------------")
                print(
                    f"Average storage time: {average_storage_time / 60:.2f} \
                    hours/shipment"
                )
                print(
                    f"Total storage time: \
                    {simulation_vars.total_storage_time / 60:.2f} hours"
                )
                print(
                    f"Total delay time: \
                    {simulation_vars.total_shipment_delay // 60:02d} hour(s) \
                    {simulation_vars.total_shipment_delay % 60:02d} minute(s)"
                )
                print("----------------------------------------\n")
                total_shipment = len(model_input.request_list)
                total_delivered = len(simulation_vars.delivered_shipments)

                print(
                    f"{total_delivered} shipment are delivered from total \
                    {total_shipment} requests"
                )
                undelivered = set(model_input.request["Demand_ID"]) - set(
                    simulation_vars.delivered_shipments
                )
                print(f"List of undelivered shipment: {undelivered}")

                # Export the episode total cost list
                current_episode = last_episode + simulation
                if config.extract_shipment_output:
                    shipment_logs(
                        shipment_dict,
                        simulation_vars.actual_itinerary,
                        simulation_vars.actual_carried_shipments,
                        simulation_vars.shipment_to_rl,
                        simulation_vars.wait_actions,
                        simulation_vars.reassign_actions,
                        simulation_vars.late_dict,
                        current_episode,
                        config.sd,
                        config.policy_name,
                        config.number_of_simulation,
                    )
            eps_end_time = time.time()  # To measure the runtime
            eps_time = eps_end_time - eps_start_time
            print(f"Episode runtime: {eps_time} seconds")
            yield statistics.dataframe(total_cost_plot, total_reward_plot)
        except Exception as e:
            print(f"Error in simulation number {simulation + 1}: {e}")
            yield None

    sim_end_time = time.time()  # To measure the runtime
    sim_time = sim_end_time - sim_start_time
    print(f"Simulation time: {sim_time} seconds")

    # Create output dataframe
    output = statistics.dataframe(total_cost_plot, total_reward_plot)
    if config.extract_shipment_output:
        output.to_csv(config.output_path, index=False)

    yield output


def main():
    config = Config()
    model_input = ModelInput(config)
    deque(model_implementation(config, model_input), maxlen=0)
