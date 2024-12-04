import pandas as pd

class AggregateStatistics:
    def __init__(self):
        # Lists for observation throughout the multiple simulations
        self.total_storage_cost_plot = []
        self.total_travel_cost_plot = []
        self.total_handling_cost_plot = []
        self.total_shipment_delay_plot = []
        self.total_late_plot = []
        self.total_number_late_plot = []
        self.total_rl_triggers = []
        self.total_assigned_rl = []
        self.undelivered_requests = []
        self.total_reassign_plot = []
        self.total_wait_plot = []
        self.x = []

    def dataframe(self, total_cost_plot, total_reward_plot):
        return pd.DataFrame(
            {
                "Episode": self.x,
                "Total Storage Cost": self.total_storage_cost_plot,
                "Total Travel Cost": self.total_travel_cost_plot,
                "Total Handling Cost": self.total_handling_cost_plot,
                "Total Delay Penalty": self.total_shipment_delay_plot,
                "Total Cost": total_cost_plot,
                "Total Reward": total_reward_plot,
                "Total Late Departure": self.total_late_plot,
                "Number of Late Departure": self.total_number_late_plot,
                "RL Triggers": self.total_rl_triggers,
                "Shipment to RL": self.total_assigned_rl,
                "Undelivered Requests": self.undelivered_requests,
                "Wait Actions": self.total_wait_plot,
                "Reassign Actions": self.total_reassign_plot,
            }
        )


class SimulationVars:
    def __init__(self, possible_paths_ref):
        # Set Simulation Variables
        self.possible_paths = possible_paths_ref.copy()
        self.announced_requests = []
        self.active_requests = []
        self.unassigned_requests = []
        self.requests_to_replan = []
        self.affected_requests = {}
        self.disruption_location = [0]
        self.d_profile_list = []
        self.s_disruption_event = {}
        self.total_storage_time = 0
        self.total_storage_cost = 0
        self.total_handling_cost = 0
        self.total_travel_cost = 0
        self.total_shipment_delay = 0
        self.total_delay_penalty = 0
        self.storage_time_list = []
        self.delivered_shipments = []
        self.actual_carried_shipments = {}
        self.actual_itinerary = {}
        self.wait_actions = {}
        self.reassign_actions = {}
        self.rl_assignment = []
        self.reward_generator = {}
        self.total_reward = 0
        self.total_cost = 0
        self.rg_order = 0
        self.om_triggers = 0
        self.rl_triggers = 0
        self.s_disruption_triggers = 0
        self.truck_id = 0
        self.truck_name_list = []
        self.shipment_to_rl = []
        self.nr_late_departure = 0
        self.total_late_departure = 0
        self.late_logs = []
        self.late_dict = {}

        self.reset(possible_paths_ref)

    def reset(self, possible_paths_ref):
        # Reset Simulation Variables
        self.possible_paths = possible_paths_ref.copy()
        self.announced_requests = []
        self.active_requests = []
        self.unassigned_requests = []
        self.requests_to_replan = []
        self.affected_requests = {}
        self.disruption_location = [0]
        self.d_profile_list = []
        self.s_disruption_event = {}
        self.total_storage_time = 0
        self.total_storage_cost = 0
        self.total_handling_cost = 0
        self.total_travel_cost = 0
        self.total_shipment_delay = 0
        self.total_delay_penalty = 0
        self.storage_time_list = []
        self.delivered_shipments = []
        self.actual_carried_shipments = {}
        self.actual_itinerary = {}
        self.wait_actions = {}
        self.reassign_actions = {}
        self.rl_assignment = []
        self.reward_generator = {}
        self.total_reward = 0
        self.total_cost = 0
        self.rg_order = 0
        self.om_triggers = 0
        self.rl_triggers = 0
        self.s_disruption_triggers = 0
        self.truck_id = 0
        self.truck_name_list = []
        self.shipment_to_rl = []
        self.nr_late_departure = 0
        self.total_late_departure = 0
        self.late_logs = []
        self.late_dict = {}
