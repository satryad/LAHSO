import numpy as np
from helper_functions import print_event


# Define function to access Q-values safely
def get_q_value(Q, state, action, default_value=0):
    if state not in Q:
        Q[state] = {}
    if action not in Q[state]:
        Q[state][action] = default_value
    return Q[state][action]


# Initialize Q Table
def make_epsilon_greedy_policy(Q, epsilon, npA, mode_ID, policy_name):
    def policy_fn(observation, possible_action):
        obs_tuple = tuple(observation)
        wait_ID = mode_ID[possible_action[0]]
        reassigned_ID = mode_ID[possible_action[1]]

        A = np.ones(npA, dtype=float) * epsilon / npA
        print_event(
        f"Q[s,a] wait: {get_q_value(Q, obs_tuple, wait_ID)}, Q[s,a] reassign: {get_q_value(Q, obs_tuple, reassigned_ID)}")
        best_action = np.argmax([
            get_q_value(Q, obs_tuple, wait_ID),
            get_q_value(Q, obs_tuple, reassigned_ID)
        ])
        worse_action = 1 - best_action  # for greedy policy

        # Epsilon-greedy policy
        if policy_name == "eg":
            A[best_action] += (1.0 - epsilon)

        # Greedy policy
        elif policy_name == "gp":
            A[best_action] = 1
            A[worse_action] = 0

        # Always reassign policy
        elif policy_name == "ar":

            A = [0, 1]

        # Always wait policy
        elif policy_name == "aw":
            A = [1, 0]

        return A

    return policy_fn