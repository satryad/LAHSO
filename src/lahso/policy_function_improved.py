import numpy as np

from lahso.helper_functions import print_event


# Define function to access Q-values safely
def get_q_value(Q, state, action, default_value=0):
    if state not in Q:
        Q[state] = {}
    if action not in Q[state]:
        Q[state][action] = default_value
    return Q[state][action]


# Initialize Q Table
def make_epsilon_greedy_policy(config, Q, epsilon, npA, mode_ID, policy_name):
    def policy_fn(observation, possible_actions):
        npA = len(possible_actions)
        obs_tuple = tuple(observation)
        # wait_ID = mode_ID[possible_action[0]]
        # reassigned_ID = mode_ID[possible_action[1]]
        action_ids = [mode_ID[action] for action in possible_actions]

        A = np.ones(npA, dtype=float) * epsilon / npA
        print_event(
            config.print_event_enabled,
            f"Q[s,a] all actions: {[get_q_value(Q, obs_tuple, action_id) for action_id in action_ids]}",
        )
        best_action = np.argmax(
            [get_q_value(Q, obs_tuple, action_id) for action_id in action_ids]
        )
        # worse_action = 1 - best_action  # for greedy policy

        # Epsilon-greedy policy
        if policy_name == "eg":
            A[best_action] += 1.0 - epsilon

        # Greedy policy
        elif policy_name == "gp":
            A = np.zeros(npA, dtype=float)
            A[best_action] = 1.0

        # Always reassign policy
        elif policy_name == "ar":
            A[0] = 0.0

        # Always wait policy
        elif policy_name == "aw":
            A = np.zeros(npA, dtype=float)
            A[1] = 1.0

        return A

    return policy_fn
