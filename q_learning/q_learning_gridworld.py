import gym
from gym_minigrid.wrappers import FullyObsWrapper
import time
import numpy as np
import random
from tqdm import tqdm
from utils.gridworld_wrapper import MyWrapper

"""
    Important:
    We need gamme to be != 1. Undiscounted would allow
    the agent to run against a wall.
    
    Some actions may never be taken and the agent
    therefore does not chose it. It's value will still
    be 0 (if we initialized with 0) and the path will
    be suboptimal. However, an optimal strategy from
    starting point to finishing point will always be 
    found.
"""

env = MyWrapper(FullyObsWrapper(gym.make('MiniGrid-Empty-5x5-v0')))
state = env.reset()
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f'# states: {n_states}')
print(f'# actions: {n_actions}')
alpha = 0.9
gamma = 0.5

q = np.zeros((n_states, n_actions))

print(q)


def q_learning():

    # Number of episodes
    for e in tqdm(range(10000)):
        state = env.reset()

        done = False

        while not done:

            print(f'state: {state}')

            # eps-greedy policy
            prob = random.uniform(0, 1)
            if prob > 1/20:
                qs = q[state]
                max_q = np.max(qs)
                indices = [i for i, e in enumerate(qs) if e == max_q]
                action = random.choice(indices)
            else:
                action = random.choice(range(n_actions))

            print(f'action: {action}')

            next_state, reward, done, _ = env.step(action)

            print(f'ns: {next_state}\n\n')

            # get next max action
            qs = q[next_state]
            max_q = np.max(qs)
            indices = [i for i, e in enumerate(qs) if e == max_q]
            next_action = random.choice(indices)

            q[state][action] = q[state][action] + alpha * (
                reward + gamma * q[next_state][next_action] - q[state][action]
            )
            state = next_state


q_learning()

# ----------------- BUILD DETERMINISTIC POLICY ------------------------------ #

pi = np.zeros((n_states,), dtype=int)

for s in range(n_states):
    qs = q[s]
    q_max = qs.max()
    indices = [i for i, e in enumerate(qs) if e == q_max]
    action = random.choice(indices)

    pi[s] = action

print(q)
print(pi)

# ------------------------ TESTING ------------------------------------------ #

state = env.reset()
env.render()

overall_reward = 0

for c in range(10):
    done = False
    total_reward = 0
    while not done:
        action = pi[state]
        print(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(2)
    print(f'Total reward: {total_reward}\n')
    overall_reward += total_reward
    time.sleep(3)
    print(f'------------------------------')
    print(f'          NEW ROUND           ')
    print(f'------------------------------\n')

    state = env.reset()
    env.render()

print(f'Average reward: {overall_reward/10}\n')







