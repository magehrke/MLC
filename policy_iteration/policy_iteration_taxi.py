import gym
import numpy as np
import sys
import copy
import random
import time
sys.path.append("..")
from utils import policy_testing

env = gym.make("Taxi-v3")
env.reset()

n_states = 500
n_actions = 6
gamma = 0.9
theta = 1

# POLICY (stochastic!)
# access with [state][action]
# we start with equilibrium policy
pi = np.full((n_states, n_actions), 1/n_actions)


def policy_evaluation():
    V = np.zeros(n_states)
    delta = np.inf
    while delta > theta:
        delta = 0
        for s in range(n_states):
            v = V[s]

            action_sum = 0
            for a in range(n_actions):
                # env is deterministic (only one successor state)
                # that is why we can use the env to get successor
                env.env.s = s  # set state
                next_state, reward, _, _ = env.step(a)

                action_sum += pi[s][a] * (reward + gamma * V[next_state])

            V[s] = action_sum
            delta = max(delta, np.linalg.norm(v - V[s]))
    return V


def policy_improvement(V):
    policy_stable = True
    for s in range(n_states):
        old_action = copy.deepcopy(pi[s])

        test_actions = np.zeros((6,))
        for a in range(n_actions):
            # env is deterministic (only one successor state)
            # get successor state
            env.env.s = s  # set state
            next_state, reward, _, _ = env.step(a)

            test_actions[a] = reward + gamma * V[next_state]
        a_max = test_actions.max()
        indices = [i for i, e in enumerate(test_actions) if e == a_max]
        n_indices = len(indices)
        new_action = np.zeros((6,))
        for i in indices:
            new_action[i] = 1/n_indices

        pi[s] = new_action

        if not np.array_equal(new_action, old_action):
            policy_stable = False

    return policy_stable

policy_stable = False

while not policy_stable:
    V = policy_evaluation()
    policy_stable = policy_improvement(V)

print(f'--------------------')
print(V)
print(f'--------------------')
print(pi)

policy_testing.test_policy(env=env, pi=pi)