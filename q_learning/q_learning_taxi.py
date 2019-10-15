import gym
import numpy as np
import sys
import random
import time
from tqdm import tqdm

env = gym.make(f'Taxi-v3')

n_states = 500
n_actions = 6
alpha = 0.9
gamma = 0.9

q = np.zeros((n_states, n_actions))


def q_learning():

    # Number of episodes
    for e in tqdm(range(10000)):
        state = env.reset()
        done = False

        while not done:

            # eps-greedy policy
            prob = random.uniform(0, 1)
            if prob > 1/(e+1):
                qs = q[state]
                max_q = np.max(qs)
                indices = [i for i, e in enumerate(qs) if e == max_q]
                action = random.choice(indices)
            else:
                action = random.choice(range(n_actions))

            next_state, reward, done, _ = env.step(action)

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

for c in range(10):
    done = False
    total_reward = 0
    while not done:
        action = pi[state]
        print(action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(3)
    print(f'Total reward: {total_reward}\n')
    time.sleep(5)
    print(f'------------------------------')
    print(f'          NEW ROUND           ')
    print(f'------------------------------\n')

    state = env.reset()
    env.render()



