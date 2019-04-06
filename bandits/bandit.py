import numpy as np


class Bandit:

    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)

    def reset(self):
        self.action_values = np.zeros(self.k)

    def pull(self, action):
        return self.action_values[action]


class GaussianBandit(Bandit):
    def __init__(self, k):
        super(GaussianBandit, self).__init__(k)
        self.action_values = np.random.normal(0, 1, self.k)
        print("True Action Values", self.action_values)

    def pull(self, action):
        return np.random.normal(self.action_values[action])
