import gym
from gym_minigrid.wrappers import FullyObsWrapper
import time
import numpy as np
import random
import copy
import sys

from utils.gridworld_wrapper import MyWrapper


env = MyWrapper(FullyObsWrapper(gym.make('MiniGrid-Empty-Random-6x6-v0')))
state = env.reset()

n_states = env.observation_space.n
n_actions = env.action_space.n
print(f'# states: {n_states}')
print(f'# actions: {n_actions}')
gamma = 0.9
theta = 0.001

V = np.zeros(n_states)
pi = np.zeros((n_states,), dtype=int)

def value_iteration():
    delta = np.inf
    iter_count = 0
    while delta > theta:
        delta = 0
        for s in range(n_states):
            v = copy.deepcopy(V[s])

            test_actions = np.zeros((n_actions,))
            for a in range(n_actions):
                next_state, reward, _, _ = env.next_state_reward(s, a)
                test_actions[a] = reward + gamma * V[next_state]
            a_max = test_actions.max()
            indices = [i for i, e in enumerate(test_actions) if e == a_max]
            action = random.choice(indices)

            V[s] = test_actions[action]

            delta = max(delta, np.linalg.norm(v - V[s]))

        iter_count = iter_count + 1

        sys.stdout.write('\r')
        sys.stdout.flush()
        sys.stdout.write(f'Iteration: {iter_count}')
        sys.stdout.flush()

    print('\n')

    # ------------ OUTPUT DETERMINISTIC POLICY ------------------------------ #

    for s in range(n_states):
        test_actions = np.zeros((n_actions,))
        for a in range(n_actions):
            next_state, reward, _, _ = env.next_state_reward(s, a)
            test_actions[a] = reward + gamma * V[next_state]
        a_max = test_actions.max()
        indices = [i for i, e in enumerate(test_actions) if e == a_max]
        action = random.choice(indices)

        pi[s] = action

    return pi


pi = value_iteration()

print(f'# --------- VALUES ----------- #')
print(V)
print(f'# --------- POLICY ---------- #')
print(pi)

# ------------------------ TESTING ------------------------------------------ #

state = env.reset()
env.render()

for c in range(10):
    done = False
    total_reward = 0
    while not done:
        action = pi[state]
        print(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(1)
    print(f'Total reward: {total_reward}\n')
    print(f'------------------------------')
    print(f'          NEW ROUND           ')
    print(f'------------------------------\n')

    state = env.reset()
    env.render()
