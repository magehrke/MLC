import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper
import time
import numpy as np
import random
from tqdm import tqdm
import copy
import sys


class MyWrapper():
    """
        ROTATION:
        0 == right
        1 == down
        2 == left
        3 == up

        Fields:
         0 3 6
         1 4 7
         2 5 8
    """

    def __init__(self, env):
        self.env = env
        self.orientation = 0
        self.field_width = env.observation_space.shape[0]
        self.field_height = env.observation_space.shape[1]
        self.observation_space = gym.spaces.discrete.Discrete(
            (self.field_height - 2) * (self.field_width - 2))
        self.action_space = gym.spaces.discrete.Discrete(4)

        # Because rendering is delayed
        self.first_render = True

    def step(self, action):
        if action not in range(0, 4):
            raise ValueError("Please specify an action from 0 to 3.")

        """
            0 == left, 1 == right, 2 == forward
            Turn right until we are in position to move forward
        """
        for k in range((action - self.orientation) % 4):
            self.env.step(1)

        next_state, reward, done, info = self.env.step(2)
        return self.transform_obs(next_state), reward, done, info

    def reset(self):
        return self.transform_obs(self.env.reset())

    def render(self, mode='human'):
        render_output = self.env.render(mode)
        if self.first_render:
            self.first_render = False
            time.sleep(1)
            self.env.step(0)
            time.sleep(1)
            self.env.step(1)
        return render_output

    def transform_obs(self, orig_obs):
        obs = 0
        for i in range(1, self.field_width - 1):
            for j in range(1, self.field_height - 1):
                if orig_obs[i][j][0] == 10:
                    self.orientation = orig_obs[i][j][2]
                    return obs
                else:
                    obs += 1

        raise RuntimeError("Agent is not on the field.")

    # DO I NEED THIS?
    def create_orig_obs(self, our_obs):
        agent_counter = 0
        obs = np.zeros((self.field_width, self.field_height, 3), dtype=int)
        for i in range(self.field_width):
            for j in range(self.field_height):
                if i == 0 or j == 0 \
                        or i == self.field_height - 1 or j == self.field_width - 1:
                    obs[i][j] = np.array([2, 5, 0])
                else:
                    if agent_counter == our_obs:
                        obs[i][j] = np.array([10, 0, self.orientation])
                    elif i == self.field_height - 2 \
                            and j == self.field_width - 2:
                        obs[i][j] = np.array([8, 1, 0])
                    else:
                        obs[i][j] = np.array([1, 0, 0])
                    agent_counter += 1
        return obs

    def next_state_reward(self, state, action):
        next_state = state
        reward = 0
        done = False

        width = self.field_width-2
        height = self.field_height-2
        fields = width * height

        if action == 1:
            if not state % width == width - 1:
                next_state = state + 1
        elif action == 0:
            if state not in range(fields - width, fields):
                next_state = state + width
        elif action == 3:
            if not state % width == 0:
                next_state = state - 1
        elif action == 2:
            if state not in range(0, width):
                next_state = state - width
        else:
            raise ValueError(f'Please provide an action between 0 and 3.')

        if next_state == fields - 1:
            reward = 1
            done = True

        return next_state, reward, done, ""

