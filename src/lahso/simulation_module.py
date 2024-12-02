import math
import time

import simpy as sim
import pandas as pd

import lahso.optimization_module as om
from lahso.helper_functions import *
from lahso.model_input import ModelInput
from lahso.policy_function import get_q_value

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None


# Define the mode of transport
class Mode:
    def __init__(
        self,
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
    ):
        self.env = env
        self.name = name
        self.origin, self.destination, self.departure_time = schedule
        self.actual_departure = 0
        self.travel_cost1, self.travel_cost2, self.handling_cost = costs
        self.capacity = capacity
        self.free_capacity = capacity
        self.assigned_shipments = []  # for capacitated
        self.speed = speed
        self.distance = distance
        self.handling_time = (
            config.handling_time
        )  # 1 minute for 1 container loading/unloading
        self.loading_time_window = config.loading_time_window  # Adjustable parameter
        self.loading = 0
        self.unloading = 0
        self.used_capacity = 0
        self.arrival = {
            node: self.env.event() for node in model_input.node_list
        }  # Events to signal when the barge/train arrives at a terminal
        self.handling_events = (
            self.env.event()
        )  # Event to signal when the barge finishes loading/unloading
        self.loading_events = self.env.event()
        self.departing_events = self.env.event()
        self.current_location = self.origin  # Track current location
        self.truck_service = self.env.event()  # Event to signal the truck service
        self.status = "Available"
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled
        self.storage_cost = config.storage_cost

    # Function for vehicle operation
    def operate(self):
        # Initialize the barge at the first location to load containers
        self.handling_events.succeed()

        while True:
            # Truck process starts only after any shipment is assigned
            if "Truck" in self.name:
                yield self.truck_service

            # Simulation starts from 2 hours before the first arrival at the origin to capture disruptions before
            # arrival in the origin terminal
            arrival_time = self.departure_time - self.loading_time_window
            operation_time = max(0, arrival_time - self.env.now - 120)
            yield self.env.timeout(operation_time)
            self.status = "Operating"

            # Simulate first arrival 1,5 hr before the first departure (according to the loading time window)
            if arrival_time > self.env.now:
                yield self.env.timeout(arrival_time - self.env.now)
            if self.name in self.global_vars.disruption_location:
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - {self.name} will arrive late in origin due to disruption at {self.name}",
                )
                yield self.global_vars.s_disruption_event[self.name]
            self.current_location = self.origin
            self.arrival[self.origin].succeed()  # Signal the arrival at the origin
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} is scheduled to depart from {self.origin} at {time_format(self.departure_time)}",
            )

            yield self.env.timeout(1)  # Wait for the used capacity to be updated
            self.arrival[self.origin] = (
                self.env.event()
            )  # reset the event in origin location for next depature

            # prioritizing cargo with the earliest release time
            sorted_shipments = sorted(self.assigned_shipments, key=lambda x: x[1])
            filtered_list = [sublist[0] for sublist in self.assigned_shipments]
            rl_shipments = []
            for shipment in sorted_shipments:
                if (
                    shipment[0] not in self.global_vars.rl_assignment
                ):  # Prioritize undisrupted shipments
                    if self.used_capacity + shipment[2] <= self.capacity:
                        self.used_capacity += shipment[2]
                        self.loading += shipment[2]
                        filtered_list.remove(shipment[0])
                else:
                    rl_shipments.append(shipment)
            for shipment in rl_shipments:
                if self.used_capacity + shipment[2] <= self.capacity:
                    self.used_capacity += shipment[2]
                    self.loading += shipment[2]
                    filtered_list.remove(shipment[0])
            self.assigned_shipments = filtered_list
            self.loading_events.succeed()
            self.loading_events = self.env.event()

            # Simulate the actual loading time according to the assigned volume
            yield self.env.timeout(self.handling_time * self.loading)
            self.assigned_shipments = []  # reset the assigned shipments

            # Check if disrupted after loading
            if (
                self.current_location in self.global_vars.disruption_location
            ):  # if the terminal is disrupted
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - {self.name} departure will be delayed due to disruption at {self.current_location}",
                )
                yield self.global_vars.s_disruption_event[self.current_location]
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} finished loading {self.loading} TEUs at {self.origin}",
            )
            if (
                self.name in self.global_vars.disruption_location
            ):  # if the mode is disrupted
                yield self.global_vars.s_disruption_event[self.name]

            # Wait until departure time
            if self.env.now < self.departure_time:
                yield self.env.timeout(self.departure_time - self.env.now)
                # Check disruption before departing
                if self.current_location in self.global_vars.disruption_location:
                    if self.used_capacity > 0:  # to focus the output on used mode
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} departure will be delayed due to disruption at {self.current_location}",
                        )
                    yield self.global_vars.s_disruption_event[self.current_location]
                self.actual_departure = self.env.now
                # Store late departure data
                if self.actual_departure > self.departure_time:
                    late_departure = self.actual_departure - self.departure_time
                    self.global_vars.total_late_departure += late_departure
                    self.global_vars.nr_late_departure += 1
                    if "Truck" in self.name:
                        name = identify_truck_line(self.name)
                    else:
                        name = self.name
                    self.global_vars.late_logs.append([name, late_departure])
                    self.global_vars.late_dict[name][0] += late_departure
                    self.global_vars.late_dict[name][1] += 1
                if self.used_capacity > 0:
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} departs from {self.origin} carrying {self.used_capacity} TEUs",
                    )
            else:
                self.actual_departure = self.env.now
                if self.used_capacity > 0:
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} late departure from {self.origin} with free capacity {self.capacity - self.used_capacity} TEUs",
                    )
                    # Store late departure data
                    late_departure = self.actual_departure - self.departure_time
                    self.global_vars.total_late_departure += late_departure
                    self.global_vars.nr_late_departure += 1
                    if "Truck" in self.name:
                        name = identify_truck_line(self.name)
                    else:
                        name = self.name
                    self.global_vars.late_logs.append([name, late_departure])
                    self.global_vars.late_dict[name][0] += late_departure
                    self.global_vars.late_dict[name][1] += 1

            # Signaling to trigger process in the shipment class
            self.departing_events.succeed()
            self.departing_events = self.env.event()
            self.handling_events = self.env.event()
            self.loading = 0  # Reset the loading counter
            self.current_location = self.name
            self.status = "En route"

            # Update departure time to the next week in the path dataset
            self.global_vars.possible_paths.loc[
                self.global_vars.possible_paths["first_service"] == self.name,
                "first_service_departure",
            ] += 168
            self.global_vars.possible_paths.loc[
                self.global_vars.possible_paths["first_service"] == self.name,
                "last_service_arrival",
            ] += 168

            # Travel to destination
            if self.name in self.global_vars.disruption_location:  # Disruption check
                yield self.global_vars.s_disruption_event[self.current_location]
            yield self.env.timeout(int(self.distance / self.speed * 60))

            # Signal the change of location for arrival
            self.current_location = self.destination
            if (
                self.current_location in self.global_vars.disruption_location
            ):  # Disruption check
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - {self.name} arriving late at {self.destination} due to disruption at {self.current_location}",
                )
                yield self.global_vars.s_disruption_event[self.current_location]
            self.arrival[self.destination].succeed()
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} arrived at {self.destination}",
            )

            # Simulate container unloading time
            yield self.env.timeout(1)  # Wait for the used capacity to be updated
            yield self.env.timeout(
                self.handling_time * self.unloading
            )  # Simulate unloading time

            # For observation
            if "Truck" in self.name:
                name = identify_truck_line(self.name)
                self.global_vars.actual_carried_shipments[name] = self.unloading
            else:
                self.global_vars.actual_carried_shipments[self.name] += self.unloading
            if self.unloading > 0:
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - {self.name} finished unloading {self.unloading} TEUs at {self.destination}",
                )
            self.unloading = 0  # Reset the unloading counter

            # Signal unloading done
            self.handling_events.succeed()
            self.arrival[self.destination] = (
                self.env.event()
            )  # reset the event in destination location for next leg
            self.status = "Available"
            if "Truck" in self.name:
                self.truck_service = self.env.event()  # finish the truck service
                self.departure_time = 99999  # reset the truck's departure time
                self.global_vars.truck_name_list.remove(self.name)
            else:
                self.departure_time = (
                    self.departure_time + 7 * 1440
                )  # Wait for the next week


# Define shipment
class Shipment:
    def __init__(self, env, request_details, config, model_input, global_vars):
        self.env = env
        (
            self.name,
            self.origin,
            self.destination,
            release_time,
            due_time,
            self.num_containers,
            self.mode,
            self.possible_itineraries,
            announce_time,
        ) = request_details
        self.announce_time = announce_time * 60  # minutes
        self.release_time = release_time * 60  # minutes
        self.due_time = due_time * 60  # minutes
        self.loading = self.env.event()
        self.tot_shipment_storage_cost = 0
        self.tot_shipment_handling_cost = 0
        self.tot_shipment_travel_cost = 0
        self.tot_shipment_delay_penalty = 0
        self.current_location = self.origin
        self.planning = self.env.event()
        self.matching_module = MatchingModule  # to call the matching module
        self.process = self.env.process(self.handled())
        self.status = "Announced"
        self.loading_signal = self.env.event()
        self.assigned_to_rl = False
        self.rl_start_time = 0
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled
        self.delay_penalty = config.delay_penalty
        self.storage_cost = config.storage_cost
        self.truck_waiting_time = config.truck_waiting_time
        # for RL
        self.reward = 0
        self.tot_shipment_reward = 0
        self.reward_event = self.env.event()
        self.state_event = {mode: self.env.event() for mode in (model_input.mode_list)}
        self.action_event = {mode: self.env.event() for mode in (model_input.mode_list)}
        self.missed_service = 0

    def handled(self):
        yield self.env.timeout(self.announce_time)

        # Announce the shipment
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - {self.name} with {self.num_containers} containers requests transport from {self.origin} to {self.destination}",
        )
        self.global_vars.announced_requests.append(self.name)
        self.global_vars.unassigned_requests.append(
            [
                self.name,
                self.origin,
                self.destination,
                self.release_time,
                self.due_time,
                self.num_containers,
                self.mode,
            ]
        )
        self.global_vars.active_requests.append(self.name)

        # Set event to wait for mode assignments, wait until planning period
        while self.status != "Assigned":
            try:
                yield self.planning
                self.status = "Assigned"
            except sim.Interrupt:
                if self.status == "New Release Time":
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} has a new release time ({self.release_time})",
                    )
                elif self.status == "New Volume":
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} has new container volume: {self.num_containers} TEUs",
                    )

        # Remove the shipment from the unassigned requests
        for req in self.global_vars.unassigned_requests:
            if req[0] == self.name:
                self.global_vars.unassigned_requests.remove(req)

        # Wait until release time
        while self.release_time > self.env.now:
            try:
                yield self.env.timeout(self.release_time - self.env.now)

            # Check demand disruption
            except sim.Interrupt:
                if self.status == "New Release Time":
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} has a new release time ({self.release_time})",
                    )
                    if self.release_time > self.mode[0].departure_time:
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} is assigned to {self.mode[0].name} with departure time {self.mode[0].departure_time}",
                        )
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} will miss the service",
                        )
                        for i in range(len(self.mode)):
                            # self.mode[i].status = "Available"
                            self.mode[i] = self.mode[i].name
                        self.release_time = max(self.env.now, self.release_time)
                        self.global_vars.requests_to_replan.append(
                            [
                                self.name,
                                self.origin,
                                self.destination,
                                self.release_time,
                                self.due_time,
                                self.num_containers,
                                self.mode,
                            ]
                        )
                        self.global_vars.disruption_location.append(self.name)
                        self.matching_module.replanning()  # Replan the shipment
                        self.global_vars.disruption_location.remove(self.name)

                elif self.status == "New Volume":
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {self.name} has new container volume: {self.num_containers} TEUs",
                    )
                    if self.num_containers > self.mode[0].free_capacity:
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} cant be assigned to {self.mode[0].name} due to insufficient capacity",
                        )
                        for i in range(len(self.mode)):
                            self.mode[i] = self.mode[i].name
                        self.global_vars.requests_to_replan.append(
                            [
                                self.name,
                                self.origin,
                                self.destination,
                                self.release_time,
                                self.due_time,
                                self.num_containers,
                                self.mode,
                            ]
                        )
                        self.global_vars.disruption_location.append(self.name)
                        self.matching_module.replanning()  # Replan the shipment
                        self.global_vars.disruption_location.remove(self.name)
                    else:
                        self.mode[0].free_capacity -= self.num_containers

        self.global_vars.announced_requests.remove(self.name)

        # Simulate the shipment handling
        while self.current_location != self.destination:
            self.status = "Waiting for arrival"

            # Order truck if the mode is truck
            if "Truck" in self.mode[0].name:
                self.mode[0].truck_service.succeed()  # trigger the truck service
                self.mode[0].departure_time = (
                    self.release_time + self.truck_waiting_time
                )
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} will be transported from {self.current_location} to {self.mode[0].destination} on {self.mode[0].name}",
            )
            while self.status == "Waiting for arrival":
                try:
                    # Wait for the mode to arrive at the shipment's origin
                    yield self.mode[0].arrival[self.current_location]

                    # Capacity availability check
                    self.mode[0].assigned_shipments.append(
                        [self.name, self.release_time, self.num_containers]
                    )
                    yield self.mode[0].loading_events
                    if self.name in self.mode[0].assigned_shipments:
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} wait for the next arrival for mode {self.mode[0].name} due to insufficient capacity",
                        )
                        self.missed_service += 1
                        yield self.mode[0].arrival[self.mode[0].destination]
                    else:
                        self.status = "Ready to load"

                # In case of the shipment is reassigned while waiting
                except sim.Interrupt:
                    if self.current_location != self.mode[0].destination:
                        print_event(
                            self.print_event_enabled,
                            f"{time_format(self.env.now)} - {self.name} is replanned and will be transported from {self.current_location} to {self.mode[0].destination} on {self.mode[0].name}",
                        )
                    if "Truck" in self.mode[0].name:
                        self.mode[
                            0
                        ].truck_service.succeed()  # trigger the truck service
                        self.mode[0].departure_time = (
                            self.env.now + self.truck_waiting_time
                        )

            # Simulate loading containers onto the mode
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} starts loading on {self.mode[0].name}",
            )
            finish_loading = self.env.now

            # Calculate the storage cost
            storage_time = max(
                0, self.env.now - self.release_time
            )  # Calculate the storage time
            shipment_storage_cost = (
                (storage_time / 60) * self.storage_cost * self.num_containers
            )  # Calculate the storage cost
            self.global_vars.total_storage_time += (
                storage_time  # Calculate the total storage time for all shipments
            )
            self.tot_shipment_storage_cost += shipment_storage_cost
            self.global_vars.total_storage_cost += shipment_storage_cost  # Calculate the total storage cost for all shipments

            # Calculate reward for RL (storage)
            if self.assigned_to_rl:
                storage_time_rl = max(0, self.env.now - self.rl_start_time)
                self.reward += (
                    (storage_time_rl / 60) * self.storage_cost * self.num_containers
                ) * -1

            # Calculate the loading cost
            loading_cost = self.num_containers * self.mode[0].handling_cost
            self.tot_shipment_handling_cost += loading_cost
            self.global_vars.total_handling_cost += loading_cost

            # Calculate reward for RL (loading)
            if self.assigned_to_rl:
                self.reward += (self.num_containers * self.mode[0].handling_cost) * -1

            # Simulate travel time from origin to destination
            self.current_location = self.mode[0].name
            self.status = "On board"
            yield self.mode[0].departing_events  # Wait for until mode actually departs

            # Calculate extra storage time while idling before the actual departure
            extra_storage_time = max(0, self.env.now - finish_loading)
            extra_storage_cost = (
                (extra_storage_time / 60) * self.storage_cost * self.num_containers
            )
            self.global_vars.total_storage_time += extra_storage_time
            self.tot_shipment_storage_cost += extra_storage_cost
            self.global_vars.total_storage_cost += extra_storage_cost
            if self.assigned_to_rl:
                self.reward += extra_storage_cost * -1

            # Update mode free capacity for next planning/replanning
            self.mode[0].free_capacity += self.num_containers

            # Update possible itineraries (for k-best solution pool approach)
            updated_itinerary = []
            if self.possible_itineraries:
                for path in self.possible_itineraries:
                    if path[0] == self.mode[0].name:
                        updated_itinerary.append(path[1:])
            self.possible_itineraries = updated_itinerary

            # Wait until the mode arrives at the destination terminal
            yield self.mode[0].arrival[self.mode[0].destination]

            # Calculate travel cost
            travel_cost1 = (
                self.mode[0].travel_cost1
                * (self.env.now - self.mode[0].actual_departure)
                / 60
                * self.num_containers
            )
            travel_cost2 = (
                self.mode[0].travel_cost2 * self.mode[0].distance * self.num_containers
            )
            travel_cost = travel_cost1 + travel_cost2
            self.tot_shipment_travel_cost += travel_cost
            self.global_vars.total_travel_cost += travel_cost

            # Calculate reward for RL (travel)
            if self.assigned_to_rl:
                self.reward += travel_cost * -1  # reward for previous action

            # Simulate unloading containers
            self.mode[0].used_capacity -= self.num_containers
            self.mode[0].unloading += self.num_containers
            yield self.mode[0].handling_events  # Wait for until unloading is done
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} completed unloading at {self.mode[0].destination} on {self.mode[0].name}",
            )

            # Calculate the handling cost
            unloading_cost = self.num_containers * self.mode[0].handling_cost
            self.tot_shipment_handling_cost += unloading_cost
            self.global_vars.total_handling_cost += unloading_cost

            # Calculate reward for RL (unloading)
            if self.assigned_to_rl:
                self.reward += (self.num_containers * self.mode[0].handling_cost) * -1
            self.current_location = self.mode[0].destination
            self.release_time = self.env.now

            # accumulate the reward from this leg
            self.tot_shipment_reward += self.reward

            # Trigger the reward generation if the shipment is assigned to RL
            if self.assigned_to_rl:
                if (
                    self.mode[0] != self.mode[-1]
                ):  # Check if the shipment is on the last mode
                    self.rl_start_time = (
                        self.env.now
                    )  # Update start time for next reward calculation
                    self.reward_event.succeed()

                    # Update state for next action
                    if "Truck" in self.mode[1].name:
                        name = identify_truck_line(self.mode[1].name)
                        self.state_event[name].succeed()
                    else:
                        self.state_event[self.mode[1].name].succeed()

                    # Signal for action completion
                    if "Truck" in self.mode[0].name:
                        name = identify_truck_line(self.mode[0].name)
                        self.action_event[name].succeed()
                    else:
                        self.action_event[self.mode[0].name].succeed()
                    yield self.env.timeout(1)
                    self.reward_event = (
                        self.env.event()
                    )  # reset the reward event for next action

                else:
                    if "Truck" in self.mode[0].name:
                        name = identify_truck_line(self.mode[0].name)
                        self.action_event[name].succeed()
                    else:
                        self.action_event[self.mode[0].name].succeed()

            # If the shipment is assigned to RL during travelling
            if self.name in self.global_vars.rl_assignment and not self.assigned_to_rl:
                self.assigned_to_rl = True  # triggered if the shipment is assigned to RL during travelling
                self.rl_start_time = self.env.now

                if "Truck" in self.mode[1].name:
                    name = identify_truck_line(self.mode[1].name)
                    self.state_event[name].succeed()
                else:
                    self.state_event[self.mode[1].name].succeed()

            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} is available at {self.current_location}",
            )
            self.global_vars.actual_itinerary[self.name].append(
                self.mode[0].name
            )  # for observation
            self.mode.pop(0)  # Remove the completed service from the itinerary

        # Shipment has arrived at the end destination
        self.status = "Delivered"
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - {self.name} has been delivered to {self.destination}",
        )
        self.global_vars.delivered_shipments.append(self.name)
        self.global_vars.active_requests.remove(self.name)

        # Calculate the delay penalty
        if self.env.now > self.due_time:
            delay = self.env.now - self.due_time
            shipment_delay_penalty = (
                (delay / 60) * self.delay_penalty * self.num_containers
            )
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.name} is late for {delay // 60:02d} hour(s) {delay % 60:02d} minute(s)",
            )
            self.global_vars.total_shipment_delay += delay
            self.tot_shipment_delay_penalty += shipment_delay_penalty
            self.global_vars.total_delay_penalty += shipment_delay_penalty

            # Calculate reward for RL (delay)
            if self.assigned_to_rl:
                self.reward += shipment_delay_penalty * -1
                self.reward_event.succeed()
                yield self.env.timeout(1)
                self.global_vars.rl_assignment.remove(self.name)
        else:
            if self.assigned_to_rl:
                self.reward_event.succeed()
                self.global_vars.rl_assignment.remove(self.name)

        # Calculate the total cost for the shipment
        self.global_vars.total_cost += (
            self.tot_shipment_storage_cost
            + self.tot_shipment_handling_cost
            + self.tot_shipment_travel_cost
            + self.tot_shipment_delay_penalty
        )
        self.tot_shipment_reward += self.reward
        self.global_vars.storage_time_list.append(self.tot_shipment_storage_cost)


# Function to check for disrupted requests
def affected_request_detection(
    env, shipment, s_disruption, planning, config, model_input, global_vars
):
    while True:
        # Wait until a service disruption occurs
        yield s_disruption.disruption_signal
        new_disrupted_location = global_vars.disruption_location[-1]
        affected_requests_list = []  # Initiate a list of affected requests for the new disruption
        for act_r in global_vars.active_requests:
            locations = []  # Initiate a list of locations in the shipment's itinerary
            if shipment[act_r].mode:
                if not isinstance(shipment[act_r].mode[0], str):
                    for mode in shipment[act_r].mode:
                        locations.append(mode.name)
                        locations.append(
                            mode.destination
                        )  # Add the assigned mode's destination
            if shipment[act_r].current_location in locations:
                locations.remove(
                    shipment[act_r].current_location
                )  # Remove the current location
            if shipment[act_r].current_location in model_input.mode_list:
                locations.remove(
                    shipment[act_r].mode[0].destination
                )  # Remove the current location destination if a shipment is on a service line
            end_destination = shipment[act_r].destination
            if end_destination in locations:
                locations.remove(end_destination)  # Remove the end destination

            # Check if the disrupted location is in the shipment's itinerary
            if locations:
                if any(location in new_disrupted_location for location in locations):
                    affected_requests_list.append(act_r)
        print_event(
            config.print_event_enabled,
            f"{time_format(env.now)} - Affected requests: {affected_requests_list}",
        )
        global_vars.affected_requests[new_disrupted_location] = affected_requests_list

        # Populate the request to replan with current information
        for af_r in affected_requests_list:
            s = shipment[af_r]
            if s.current_location in model_input.node_list:
                s.origin = s.current_location
            else:
                s.origin = s.mode[0].destination
                # print(f'{s.name} is disrupted while on board')
            for i in range(len(s.mode)):
                s.mode[i].status = "Available"
                s.mode[i].free_capacity += s.num_containers
                s.mode[i] = s.mode[i].name
            global_vars.requests_to_replan.append(
                [
                    s.name,
                    s.origin,
                    s.destination,
                    s.release_time,
                    s.due_time,
                    s.num_containers,
                    s.mode,
                ]
            )

        if global_vars.requests_to_replan:
            planning.replanning()
        s_disruption.disruption_signal = env.event()


# Matching Module
class MatchingModule:
    def __init__(
        self,
        env,
        mode_schedule,
        shipment,
        rl_module,
        interval,
        config,
        model_input,
        global_vars,
    ):
        self.env = env
        self.mode_schedule = mode_schedule
        self.shipment = shipment
        self.disruption_event = env.event()
        self.rl_module = rl_module
        self.planning_interval = interval
        self.mode_list = model_input.mode_list
        self.node_list = model_input.node_list
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled
        self.fixed_list = model_input.fixed_list
        self.handling_time = config.handling_time
        self.storage_cost = config.storage_cost
        self.delay_penalty = config.delay_penalty
        self.penalty_per_unfulfilled_demand = config.penalty_per_unfulfilled_demand
        self.truck_schedule_dict = model_input.truck_schedule_dict
        self.truck_list = model_input.truck_list
        self.config = config
        self.model_input = model_input

    def planning(self):
        yield self.env.timeout(1)
        while True:
            disrupted_location = self.global_vars.disruption_location
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - disruption at {disrupted_location}",
            )
            request_list = self.global_vars.unassigned_requests

            # Identify planned and unplanned requests
            planned_requests = []
            unplanned_requests = []
            for req in request_list:
                if not req[6]:
                    unplanned_requests.append(req)
                else:
                    planned_requests.append(req)

            # ---------------------------------OPTIMIZATION MODULE---------------------------------
            available_paths = self.FilterPath()
            if request_list:
                matching = self.OptimizationModule(unplanned_requests, available_paths)
            # ------------------------------------------------------------------------------------

            # Assign the planned requests to the matcching dictionary
            for req in planned_requests:
                matching[req[0]] = ([], req[6])
            self.ModeAssignment(request_list, matching)

            # Wait for next planning phase
            yield self.env.timeout(self.planning_interval)

    def replanning(self):
        disrupted_location = self.global_vars.disruption_location[-1]
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - Replanning due to disruption at {disrupted_location}",
        )
        request_list = self.global_vars.requests_to_replan
        if (
            disrupted_location in self.node_list or disrupted_location in self.mode_list
        ):  # To skip this process for disruption in request
            for request in request_list:
                if self.shipment[request[0]].current_location in self.node_list:
                    self.shipment[request[0]].process.interrupt()
                self.shipment[
                    request[0]
                ].planning = self.env.event()  # Reset the planning signal
        else:
            self.shipment[
                request_list[0][0]
            ].planning = self.env.event()  # Reset the planning signal

        available_paths = self.FilterPath()
        unsolved_requests = []
        solved_requests = []

        # Select the best mode from solution pool (for K-Best solution)
        for req in request_list:
            # Eliminated disrupted itinerary from solution pool
            available_next_solution = []
            for path in self.shipment[req[0]].possible_itineraries:
                if disrupted_location not in path:
                    available_next_solution.append(path)
            if not available_next_solution:
                unsolved_requests.append(req)
            else:
                # Identify capacity constraint
                for path in available_next_solution:
                    path_capacity = []
                    for mode in path:
                        if "Truck" not in mode:
                            path_capacity.append(self.mode_schedule[mode].free_capacity)
                        else:
                            path_capacity.append(99999)
                    path_capacity = min(
                        path_capacity
                    )  # Determine path capacity from possible itineraries
                    if req[5] <= path_capacity:
                        solved_requests.append(req)
                        new_mode = path
                        req[6] = [req[6], new_mode]
                        break
                if req not in solved_requests:
                    unsolved_requests.append(req)

        # Trigger optimization model if there are unsolved requests
        if unsolved_requests:
            matching = self.OptimizationModule(unsolved_requests, available_paths)
        else:
            matching = {}

        # Assign the solved requests to the matching dictionary
        for req in solved_requests:
            new_mode = req[6][1]
            old_mode = req[6][0]
            matching[req[0]] = (old_mode, new_mode)

        # Assign the unmatched requests to truck
        for req in request_list:
            if not matching[req[0]][1]:
                origin = self.shipment[req[0]].origin
                destination = self.shipment[req[0]].destination
                old_mode = matching[req[0]][0]
                new_mode = old_mode
                for mode in self.truck_list:
                    if (
                        self.truck_schedule_dict[mode][1][0] == origin
                        and self.truck_schedule_dict[mode][1][1] == destination
                    ):
                        new_mode = [mode]
                        break
                matching[req[0]] = (old_mode, new_mode)

        # Triggers RL only if service disruption
        if (
            disrupted_location in self.node_list
            or disrupted_location in self.mode_list
            or disrupted_location in self.global_vars.truck_name_list
        ):
            # Start RL assginment
            self.global_vars.rl_triggers += 1
            rl_match = {}

            # Determine input for RL agent
            for request in request_list:
                if request[0] not in self.global_vars.shipment_to_rl:
                    self.global_vars.shipment_to_rl.append(request[0])

                # actions
                action_sets = matching[request[0]]

                # states
                current_location = self.shipment[request[0]].current_location
                if self.shipment[request[0]].status == "On board":
                    mode = action_sets[0]  # the current mode
                    current_location = self.mode_schedule[mode[0]].destination
                destination = self.shipment[request[0]].destination
                due_time = self.shipment[request[0]].due_time // 60
                volume = self.shipment[request[0]].num_containers
                d_profile = self.global_vars.d_profile_list[-1][0]
                current_time = self.env.now % (1440 * 7) // 60
                current_state = (
                    current_location,
                    destination,
                    due_time,
                    volume,
                    d_profile,
                    current_time,
                )
                print_event(
                    self.print_event_enabled,
                    f"State for RL: {current_state}, Actions: {action_sets}",
                )

                ## Notes
                # actions[0] = wait, actions[1] = reassign
                # reward will be delayed until the shipment arrives at the next terminal
                action_set_taken = self.rl_module.action_generator(
                    request, current_state, action_sets
                )  # select the best action

                if (
                    request[0] not in self.global_vars.rl_assignment
                ):  # For first disruption in a request
                    # Setup first assignment to RL
                    self.global_vars.reward_generator[request[0]] = []
                    if self.shipment[request[0]].status == "Waiting for arrival":
                        self.shipment[request[0]].rl_start_time = self.env.now
                        self.shipment[request[0]].assigned_to_rl = True
                    elif self.shipment[request[0]].status == "On board":
                        self.shipment[
                            request[0]
                        ].assigned_to_rl = (
                            False  # wait until arrrive in the next terminal
                        )
                    else:
                        self.shipment[request[0]].rl_start_time = max(
                            self.env.now, self.shipment[request[0]].release_time
                        )
                        self.shipment[request[0]].assigned_to_rl = True

                else:
                    # Update reward from previous actions
                    if self.shipment[request[0]].status == "Waiting for arrival":
                        storage_time_rl = max(
                            0, self.env.now - self.shipment[request[0]].rl_start_time
                        )
                        self.shipment[request[0]].reward += (
                            (storage_time_rl / 60)
                            * self.storage_cost
                            * self.shipment[request[0]].num_containers
                        ) * -1

                    # Interrupt the previous reward generation process for shipment with multiple disruptions
                    for process_ID in self.global_vars.reward_generator[request[0]]:
                        process = process_ID[0]
                        process.interrupt()

                    self.global_vars.reward_generator[request[0]] = []
                    self.shipment[request[0]].rl_start_time = self.env.now
                    self.shipment[request[0]].assigned_to_rl = True

                # Generate initial state for each action
                for i in range(len(action_set_taken)):
                    print_event(
                        self.print_event_enabled,
                        f"RL ASSIGNMENT: {request[0]} - {action_set_taken[i]}",
                    )
                    if i == 0:
                        state = current_state
                        future = False
                    else:  # for future actions
                        state = current_state
                        future = True
                    self.global_vars.rg_order += 1
                    if "Truck" in action_set_taken[i]:
                        action_set_taken[i] = identify_truck_line(action_set_taken[i])
                    reward_gen = self.env.process(
                        self.rl_module.reward_generator(
                            request,
                            state,
                            action_set_taken[i],
                            future,
                            self.global_vars.rg_order,
                        )
                    )
                    self.global_vars.reward_generator[request[0]].append(
                        [reward_gen, self.global_vars.rg_order]
                    )

                self.global_vars.rl_assignment.append(request[0])
                rl_match[request[0]] = (action_sets[0], action_set_taken)
            self.ModeAssignment(request_list, rl_match)
        else:
            self.ModeAssignment(request_list, matching)
        self.global_vars.requests_to_replan = []  # reset the requests to replan

    # Optimization Algorithm (Start)----------------------------------------------

    def FilterPath(self):
        disrupted_location = self.global_vars.disruption_location
        available_paths = self.global_vars.possible_paths[:]
        # available_paths.to_csv(f"{path}/available_paths.csv", index=False)
        for location in disrupted_location:
            if location != 0:
                if location in self.mode_list:  # For disruption in service line
                    available_paths = available_paths[
                        ~available_paths["service_ids"].str.contains(location)
                    ]
                else:  # For disruption in destination
                    available_paths = available_paths[
                        ~available_paths["Transshipment Terminal(s)"].str.contains(
                            location
                        )
                    ]
        return available_paths

    def OptimizationModule(self, request_list, available_paths):
        self.global_vars.om_triggers += 1
        print_event(
            self.print_event_enabled,
            f"This is triggers number: {self.global_vars.om_triggers}",
        )

        # Convert input to df
        df_request_list = pd.DataFrame(
            request_list,
            columns=[
                "Demand_ID",
                "Origin",
                "Destination",
                "Release Time",
                "Due Time",
                "Volume",
                "Mode",
            ],
        )

        # Demand input preprocessing
        df_request_list["Release Time"] = df_request_list["Release Time"] / 60
        df_request_list["Due Time"] = df_request_list["Due Time"] / 60

        # Update service capacity
        capacitated_service = available_paths.copy()
        for service in self.fixed_list:
            free_capacity = self.mode_schedule[service].free_capacity
            capacitated_service.loc[
                capacitated_service["service_ids"].str.contains(service),
                "service_capacity",
            ] = capacitated_service.loc[
                capacitated_service["service_ids"].str.contains(service),
                "service_capacity",
            ].apply(lambda x: max(0, (min(x, free_capacity))))
            update_service_capacity(capacitated_service, service, free_capacity)
        capacitated_service["Loading Time"] = self.handling_time / 60

        # Filter eligible paths according to the request list od pairs
        unique_pairs = unique_origin_destination_pairs(df_request_list)
        filtered_paths = capacitated_service[
            capacitated_service[["origin", "destination"]]
            .apply(tuple, axis=1)
            .isin(map(tuple, unique_pairs))
        ]
        filtered_paths["Week"] = 1  # Error handling

        # Run optimization algorithm
        matching = om.run_optimization(
            df_request_list,
            filtered_paths,
            self.storage_cost,
            self.delay_penalty,
            self.penalty_per_unfulfilled_demand,
        )
        df_matching = pd.DataFrame(matching)

        if df_matching.empty:
            df_matching_combined = df_request_list
            df_matching_combined["Service_ID"] = 0
        else:
            df_matching_combined = df_request_list.merge(
                df_matching, on="Demand_ID", how="left"
            ).fillna(0)

        # Convert output to dictionary
        matching_result = dict(
            zip(
                df_matching_combined["Demand_ID"],
                df_matching_combined["Service_ID"],
                strict=False,
            )
        )

        # Fill the matching result with old and new itinerary assignment
        for key, values in matching_result.items():
            old_mode = df_request_list.loc[
                df_request_list["Demand_ID"] == key, "Mode"
            ].values[0]
            for i in range(len(old_mode)):
                if "Truck" in old_mode[i]:
                    old_mode[i] = identify_truck_line(old_mode[i])
            if values == 0:
                new_mode = []
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - {key} has no possible new mode assignment",
                )
            else:
                new_mode = [values]
                new_mode = [
                    item.strip() for sublist in new_mode for item in sublist.split(",")
                ]

            matching_result[key] = (old_mode, new_mode)

        return matching_result

    # Optimization Algorithm (End)----------------------------------------------

    # Function to assign the selected mode to the shipment with simpy object
    def ModeAssignment(self, request_list, match):
        for request in request_list:
            current_location = self.shipment[request[0]].current_location
            old_mode, new_mode = match[request[0]]
            assigned_mode = []
            if new_mode:
                if current_location in self.mode_list:
                    assigned_mode = [self.mode_schedule[current_location]]

                if old_mode == new_mode:
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {request[0]} from {request[1]} to {request[2]} will wait",
                    )
                    assigned_mode = []
                else:
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {request[0]} from {request[1]} to {request[2]} is assigned to {new_mode}",
                    )

                for mode in new_mode:
                    if "Truck" in mode:
                        if len(mode) < 8:
                            self.global_vars.truck_id += 1
                            name, schedule, capacity, speed, distance, costs = (
                                self.truck_schedule_dict[mode]
                            )
                            name = f"{mode}.{self.global_vars.truck_id}"
                            # Create an object for truck
                            self.mode_schedule[name] = Mode(
                                self.env,
                                name,
                                schedule,
                                capacity,
                                speed,
                                distance,
                                costs,
                                self.config,
                                self.model_input,
                                self.global_vars,
                            )
                            self.global_vars.truck_name_list.append(name)
                            self.env.process(self.mode_schedule[name].operate())
                        else:
                            name = mode
                        assigned_mode.append(self.mode_schedule[name])
                    else:
                        assigned_mode.append(self.mode_schedule[mode])
                self.shipment[request[0]].mode = assigned_mode
                for mode in assigned_mode:
                    mode.free_capacity -= self.shipment[request[0]].num_containers
                self.shipment[request[0]].planning.succeed()

            else:
                new_mode = old_mode
                if new_mode:
                    print_event(
                        self.print_event_enabled,
                        f"{time_format(self.env.now)} - {request[0]} from {request[1]} to {request[2]} will wait",
                    )
                    for mode in new_mode:
                        assigned_mode.append(self.mode_schedule[mode])
                    self.shipment[request[0]].mode = assigned_mode
                    for mode in assigned_mode:
                        mode.free_capacity -= self.shipment[request[0]].num_containers
                    self.shipment[request[0]].planning.succeed()


# Reinforcement Learning Module
class ReinforcementLearning:
    def __init__(
        self,
        env,
        shipment,
        mode_schedule,
        q_table,
        discount_factor,
        alpha,
        policy,
        config,
        model_input,
        global_vars,
    ):
        self.env = env
        self.shipment = shipment
        self.mode_schedule = mode_schedule
        self.q_table = q_table
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.generate_reward = self.env.event()
        self.queue = []
        self.policy = policy
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled
        self.loc_to_index = model_input.loc_to_index
        self.dest_to_index = model_input.dest_to_index
        self.d_profile_to_index = model_input.d_profile_to_index
        self.mode_ID = model_input.mode_ID

    # Function to select best action according to the policy
    def action_generator(self, request, state, action_sets):
        # Define actions based on the first mode in each itinerary
        wait = action_sets[0][0]
        if "Truck" in wait:
            wait = identify_truck_line(wait)  # Convert truck object to truck name
        immediate_action = action_sets[1][0]
        possible_action = [wait, immediate_action]

        # Convert state to vector
        current_location_vector = tuple(state_to_vector(state[0], self.loc_to_index))
        destination_vector = tuple(state_to_vector(state[1], self.dest_to_index))
        profile_vector = tuple(state_to_vector(state[4], self.d_profile_to_index))
        state_vector = (
            current_location_vector,
            destination_vector,
            state[2],
            state[3],
            profile_vector,
            state[5],
        )

        # Select action based on the policy
        action_probs = self.policy(state_vector, possible_action)
        vary_seed = np.random.default_rng(
            int(time.time() * 1000)
        )  # independent random seed for epsilon greedy
        action_id = vary_seed.choice(np.arange(len(action_probs)), p=action_probs)
        chosen_set = action_sets[action_id]

        if action_id == 0:
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - RL choose wait for {request[0]}",
            )
            self.global_vars.wait_actions[request[0]] += 1
        else:
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - RL choose reassign for {request[0]}",
            )
            self.global_vars.reassign_actions[request[0]] += 1

        return chosen_set

    def reward_generator(self, request, state, action, future, gen_order):
        self.function_stop = self.env.event()

        action_taken = self.mode_ID[action]
        try:
            if not future:
                updated_state = state
                wait = False
                if self.shipment[request[0]].status == "On board":
                    wait = True
                    yield self.function_stop

            else:
                wait = True
                yield self.shipment[request[0]].state_event[
                    action
                ]  # Wait until the previous action is completed
                wait = False

                # State for future actions
                current_location = self.shipment[request[0]].current_location
                destination = self.shipment[request[0]].destination
                due_time = self.shipment[request[0]].due_time // 60
                volume = self.shipment[request[0]].num_containers
                d_profile = "no disruption"
                current_time = self.env.now % (1440 * 7) // 60
                updated_state = (
                    current_location,
                    destination,
                    due_time,
                    volume,
                    d_profile,
                    current_time,
                )

            # Wait until the action is completed
            yield self.shipment[request[0]].reward_event
            yield self.shipment[request[0]].action_event[action]

            # Determine next state
            current_location, destination, due_time, volume, d_profile, current_time = (
                updated_state
            )
            current_location = self.shipment[request[0]].current_location
            d_profile = "no disruption"
            current_time = self.env.now % (1440 * 7) // 60
            next_state = (
                current_location,
                destination,
                due_time,
                volume,
                d_profile,
                current_time,
            )

        except sim.Interrupt:
            if wait:
                # Terminal reward for future action that has not been executed
                yield self.function_stop
            if self.shipment[request[0]].status == "On board":
                # wait until it arrives at the next terminal
                yield self.shipment[request[0]].reward_event
                yield self.shipment[request[0]].action_event[action]
            # Determine next state
            current_location, destination, due_time, volume, d_profile, current_time = (
                updated_state
            )
            current_location = self.shipment[request[0]].current_location
            if "Truck" in current_location:
                current_location = identify_truck_line(current_location)
            d_profile = (
                self.global_vars.d_profile_list[-1][0]
                if self.global_vars.d_profile_list
                else "no disruption"
            )
            current_time = self.env.now % (1440 * 7) // 60
            next_state = (
                current_location,
                destination,
                due_time,
                volume,
                d_profile,
                current_time,
            )

        # Get reward
        reward = self.shipment[request[0]].reward
        self.global_vars.total_reward += reward
        self.shipment[request[0]].reward = 0
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - {request[0]} got reward: {reward} for action: {action}",
        )

        # Determine next action
        yield self.env.timeout(1)
        if (
            self.shipment[request[0]].status == "Delivered"
            or self.shipment[request[0]].status == "Undelivered"
        ):
            next_action = 0  # no action after a terminal state

        else:
            mode = self.shipment[request[0]].mode[0].name
            if "Truck" in mode:
                name = identify_truck_line(mode)
                next_action = self.mode_ID[name]
            else:
                next_action = self.mode_ID[mode]

        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - {request[0]} take next action: {next_action}",
        )
        self.shipment[request[0]].action_event[action] = self.env.event()
        while self.queue:
            yield self.env.timeout(1)

        # Convert current state to vector (could be useful for upgrading using deep RL)
        current_location_vector = tuple(
            state_to_vector(updated_state[0], self.loc_to_index)
        )
        destination_vector = tuple(
            state_to_vector(updated_state[1], self.dest_to_index)
        )
        profile_vector = tuple(
            state_to_vector(updated_state[4], self.d_profile_to_index)
        )
        updated_state_vector = (
            current_location_vector,
            destination_vector,
            updated_state[2],
            updated_state[3],
            profile_vector,
            updated_state[5],
        )

        # Convert next state to vector (could be useful for upgrading using deep RL)
        current_location_vector = tuple(
            state_to_vector(next_state[0], self.loc_to_index)
        )
        destination_vector = tuple(state_to_vector(next_state[1], self.dest_to_index))
        profile_vector = tuple(state_to_vector(next_state[4], self.d_profile_to_index))
        next_state_vector = (
            current_location_vector,
            destination_vector,
            next_state[2],
            next_state[3],
            profile_vector,
            next_state[5],
        )

        updated_state_tuple = tuple(updated_state_vector)
        next_state_tuple = tuple(next_state_vector)

        # Add the request to the queue for q table updating
        self.queue.append(
            [
                updated_state_tuple,
                action_taken,
                reward,
                next_state_tuple,
                next_action,
                request[0],
            ]
        )
        self.update_q_table(self.q_table, self.discount_factor, self.alpha)
        for reward_gen in self.global_vars.reward_generator[request[0]]:
            if reward_gen[1] == gen_order:
                self.global_vars.reward_generator[request[0]].remove(reward_gen)

    def update_q_table(self, Q, discount_factor, alpha):
        state, action, reward, next_state, next_action, request = self.queue[0]

        # Prevent error if the state/action is not in the Q table
        if state not in Q:
            Q[state] = {}
        if action not in Q[state]:
            Q[state][action] = 0

        if next_state not in Q:
            Q[next_state] = {}
        if next_action not in Q[next_state]:
            Q[next_state][next_action] = 0

        # Identify possible actions for the next state (for Q-learning)
        possible_action = {}
        for a in self.mode_ID.values():
            if get_q_value(Q, next_state, a) != 0:
                possible_action[a] = [Q[next_state][a]]
        if not possible_action:
            best_next_action = 0
        else:
            best_next_action = max(possible_action, key=possible_action.get)

        # Apply Q-learning equation
        td_target = reward + discount_factor * get_q_value(
            Q, next_state, best_next_action
        )
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - Q[s,a] before update: {Q[state][action]}",
        )
        td_delta = td_target - get_q_value(Q, state, action)
        Q[state][action] += alpha * td_delta
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - Q[s,a] after update: {Q[state][action]}",
        )
        self.queue.pop(0)


# Define service disruption
class ServiceDisruption:
    def __init__(self, env, mode_schedule, profile, config, model_input, global_vars):
        self.env = env
        self.disruption_signal = self.env.event()
        self.mode_schedule = mode_schedule
        self.profile = profile
        self.disruption_sequence = [0]
        self.start_times = []
        self.mode_list = model_input.mode_list
        self.node_list = model_input.node_list
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled
        self.start_operation = config.start_operation

    # Function to start a generator for each profile
    def produce(self):
        for profile in self.profile:
            self.env.process(self.generate_s_disruption(profile))
            yield self.env.timeout(1)  # Wait to generate the first disruption

    def generate_s_disruption(self, profile):
        name, type, lbd, ubd, lbc, ubc, possible_location, lambda_rate = profile

        while True:
            # Generate IAT for disruption start time
            IAT = int(np.random.exponential(scale=1 / lambda_rate))
            location = 0
            if IAT != 0:
                start_time = self.env.now + IAT
                if start_time in self.start_times:
                    IAT += 1  # avoiding conflict for event signal
                    start_time = self.env.now + IAT
                self.start_times.append(
                    start_time
                )  # avoiding conflict for event signal
                duration = np.random.randint(lbd * 60, ubd * 60)
                capacity_reduction = np.random.uniform(lbc, ubc)
                loc_candidate = np.random.choice(possible_location)
                count = 0
                yield self.env.timeout(IAT)
                while (
                    location in self.global_vars.disruption_location
                ):  # avoiding the same location
                    if loc_candidate == "Terminal":
                        location = np.random.choice(self.node_list)
                    else:
                        mode_candidate_ref = [
                            item for item in self.mode_list if loc_candidate in item
                        ]
                        mode_candidate = []
                        for mode in mode_candidate_ref:
                            operating = (
                                self.mode_schedule[mode].departure_time
                                - self.start_operation
                            )
                            depart = self.mode_schedule[mode].departure_time
                            if "Truck" in mode or operating <= self.env.now <= depart:
                                mode_candidate.append(mode)
                        if mode_candidate:
                            location = np.random.choice(mode_candidate)
                        else:
                            location = np.random.choice(mode_candidate_ref)
                            location = (
                                0  # if no service line as candidate to be disrupted
                            )
                            break
                        if "Truck" in location:
                            if self.global_vars.truck_name_list:
                                location = np.random.choice(
                                    self.global_vars.truck_name_list
                                )
                    count += 1
                    # Prevent infinite looping
                    if count > 10:
                        location = 0
                        break
                if location != 0:
                    self.disruption_sequence.append(location)
                    original_capacity = (
                        self.mode_schedule[location].capacity
                        if type == "Capacity reduction"
                        else 0
                    )
                    location_new = location
                    self.global_vars.disruption_location.append(
                        location_new
                    )  # Add location to a list for ongoing disruptions
                    if type == "Capacity reduction":
                        self.mode_schedule[location].capacity = math.ceil(
                            (1 - capacity_reduction)
                            * self.mode_schedule[location].capacity
                        )
                    self.global_vars.d_profile_list.append(
                        (name, location)
                    )  # list for identifying profile for RL state
                    self.env.process(
                        self.execute_disruption(
                            location, duration, name, type, original_capacity
                        )
                    )
                    yield self.env.timeout(1)
            else:
                yield self.env.timeout(1)

    # Function to execute the disruption
    def execute_disruption(self, location, duration, name, type, original_capacity):
        # Simulate service disruption
        self.global_vars.s_disruption_event[location] = self.env.event()
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - Service disruption type {type} at/on {location} starts",
        )
        self.global_vars.s_disruption_triggers += 1
        if type == "Capacity reduction":
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.mode_schedule[location].name} capacity is reduced to {self.mode_schedule[location].capacity}",
            )
            location_new = location
            self.global_vars.s_disruption_event[location_new].succeed()
        self.disruption_signal.succeed()

        yield self.env.timeout(duration)
        if type == "Capacity reduction":
            self.mode_schedule[location].capacity = original_capacity
            print_event(
                self.print_event_enabled,
                f"{time_format(self.env.now)} - {self.mode_schedule[location].name} capacity is restored to {original_capacity}",
            )
        else:
            location_new = location
            self.global_vars.s_disruption_event[location_new].succeed()
        print_event(
            self.print_event_enabled,
            f"{time_format(self.env.now)} - Service disruption type {type} at/on {location} ends",
        )
        self.disruption_sequence.remove(location)
        location_new = location

        # Update disruption list
        self.global_vars.disruption_location.remove(location_new)
        self.global_vars.d_profile_list.remove((name, location_new))


# Define demand disruption
class DemandDisruption:
    def __init__(self, env, shipment, profile, config, global_vars):
        self.env = env
        self.shipment = shipment
        self.profile = profile
        self.global_vars = global_vars
        self.print_event_enabled = config.print_event_enabled

    def produce(self):
        for profile in self.profile:
            self.env.process(self.generate_d_disruption(profile))
            yield self.env.timeout(1)  # Wait to generate the first disruption

    def generate_d_disruption(self, profile):
        # global self.global_vars.d_profile_list
        name, type, lbt, ubt, lbv, ubv, lambda_rate = profile
        # disruption_type = ('Release Time', 'Volume')
        while True:
            # Randomize disruptions
            lambda_rate = lambda_rate
            start_time_d = int(np.random.exponential(scale=1 / lambda_rate))
            yield self.env.timeout(start_time_d)
            disruption = type
            if self.global_vars.announced_requests:
                disrupted_shipment = np.random.choice(
                    self.global_vars.announced_requests
                )
                self.global_vars.d_profile_list.append((name, disrupted_shipment))
                print_event(
                    self.print_event_enabled,
                    f"{time_format(self.env.now)} - Disrupting {disrupted_shipment} with {disruption}",
                )
                if (
                    self.shipment[disrupted_shipment].status == "Announced"
                    or self.shipment[disrupted_shipment].status == "Assigned"
                ):
                    if disruption == "Release Time":
                        # Simulate disruption for change in the release time
                        late_release = np.random.randint(lbt * 60, ubt * 60)
                        self.shipment[disrupted_shipment].release_time += late_release
                        self.shipment[disrupted_shipment].status = "New Release Time"
                        self.shipment[disrupted_shipment].process.interrupt()
                    else:
                        # Simulate disruption for change in the shipment volume
                        volume_multiplier = np.random.uniform(1 + lbv, 1 + ubv)
                        # Update free capacity for the assigned mode
                        if self.shipment[disrupted_shipment].mode:
                            if not isinstance(
                                self.shipment[disrupted_shipment].mode[0], str
                            ):
                                modes = self.shipment[disrupted_shipment].mode
                                for mode in modes:
                                    mode.free_capacity += self.shipment[
                                        disrupted_shipment
                                    ].num_containers
                        self.shipment[disrupted_shipment].num_containers = math.ceil(
                            self.shipment[disrupted_shipment].num_containers
                            * volume_multiplier
                        )
                        self.shipment[disrupted_shipment].status = "New Volume"
                        self.shipment[disrupted_shipment].process.interrupt()
                yield self.env.timeout(1)

                # Update disruption list
                self.global_vars.d_profile_list.remove((name, disrupted_shipment))


# Update cost for undelivered shipments (at the end of the simulation)
def update_undelivered_shipments(
    env, shipment_dict, simulation_duration, penalty, config, global_vars
):
    yield env.timeout(simulation_duration - env.now - 1)
    print_event(config.print_event_enabled, "\nUPDATE UNDELIVERED SHIPMENTS")
    for _key, value in shipment_dict.items():
        if value.status != "Delivered":
            if value.status == "Waiting for arrival":
                storage_time = max(0, env.now - value.release_time)
                shipment_storage_cost = (
                    (storage_time / 60) * config.storage_cost * value.num_containers
                )
                global_vars.total_storage_time += storage_time
                value.tot_shipment_storage_cost += shipment_storage_cost
                global_vars.total_storage_cost += shipment_storage_cost
                global_vars.total_cost += value.tot_shipment_storage_cost

                if value.name in global_vars.rl_assignment:
                    storage_time_rl = max(0, env.now - value.rl_start_time)
                    value.reward += (
                        (storage_time_rl / 60)
                        * config.storage_cost
                        * value.num_containers
                    ) * -1
                    # penalty for undelivered shipments
                    value.reward += penalty * -1
                    value.tot_shipment_reward += value.reward

            if value.status == "On board":
                travel_cost1 = (
                    value.mode[0].travel_cost1
                    * (env.now - value.mode[0].actual_departure)
                    / 60
                    * value.num_containers
                )
                travel_cost2 = (
                    value.mode[0].travel_cost2
                    * value.mode[0].distance
                    * value.num_containers
                )
                travel_cost = travel_cost1 + travel_cost2
                value.tot_shipment_travel_cost += travel_cost
                global_vars.total_travel_cost += travel_cost
                global_vars.total_cost += value.tot_shipment_travel_cost

                if value.name in global_vars.rl_assignment:
                    value.reward += travel_cost * -1
                    # penalty for undelivered shipments
                    value.reward += penalty * -1
                    value.tot_shipment_reward += value.reward

            value.status = "Undelivered"
            if value.name in global_vars.rl_assignment:
                value.assigned_to_rl = True
                for process_ID in global_vars.reward_generator[value.name]:
                    process = process_ID[0]
                    process.interrupt()

                if value.name in global_vars.rl_assignment:
                    value.reward_event.succeed()
                    if "Truck" in value.mode[0].name:
                        name = identify_truck_line(value.mode[0].name)
                        value.action_event[name].succeed()
                    else:
                        value.action_event[value.mode[0].name].succeed()
