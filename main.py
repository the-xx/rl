from bandits import *

k = 5
steps = 1000
policy = EpsilonGreedyPolicy(0.01)
bandit = GaussianBandit(k)
agent = Agent(k, policy)
agent.choose()


for i in range(steps):
    action = agent.choose()
    reward = bandit.pull(action)

    print("Action:", action, "Reward: ", reward)
    agent.update(reward)
print("True Action Values ", bandit.action_values)
print("Estimate Action Values ", agent.action_value_estimates)
