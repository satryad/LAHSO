from lahso.model_input import possible_paths_ref

# Lists for observation throughout the multiple simulation
total_storage_cost_plot = []
total_travel_cost_plot = []
total_handling_cost_plot = []
total_shipment_delay_plot = []
total_late_plot = []
total_number_late_plot = []
total_rl_triggers = []
total_assigned_rl = []
undelivered_requests = []
total_reassign_plot = []
total_wait_plot = []
x = []


class GlobalVars:
    def __init__(self):
        # Set Global Variables
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

        self.reset()

    def reset(self):
        # Reset Global Variables
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


global_vars = GlobalVars()
