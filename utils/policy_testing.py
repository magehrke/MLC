import random
import time


def test_policy(env, pi, n_iter=10):
    overall_reward = 0

    for t in range(n_iter):
        done = False
        total_reward = 0

        state = env.reset()
        env.render()

        while not done:
            action_array = pi[state]
            indices = [i for i, e in enumerate(action_array) if e != 0]
            action = random.choice(indices)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(3)
        print(f'Total reward: {total_reward}\n')
        overall_reward += total_reward
        print(f'------------------------------')
        print(f'          NEW ROUND           ')
        print(f'------------------------------\n')

    print(f'Average reward: {overall_reward / n_iter}\n')

