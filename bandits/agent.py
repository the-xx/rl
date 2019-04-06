import numpy as np


class Agent:
    def __init__(self, k, policy, action_value_estimates=None):
        self.k = k
        self.policy = policy
        self.t = 0

        if action_value_estimates is None:
            action_value_estimates = np.zeros(k)
        self.action_value_estimates = action_value_estimates
        self.action_count_history = np.zeros(k)
        self.last_action = None

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def update(self, reward):
        self.t += 1
        self.action_count_history[self.last_action] += 1

        N = self.action_count_history[self.last_action]
        Q = self.action_value_estimates[self.last_action]

        self.action_value_estimates[self.last_action] = Q + 1/N * (reward - Q)
