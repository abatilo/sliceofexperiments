import os

import gym
from gym.spaces import Discrete


class Environment:
    UPPER_BOUND = 5

    def __init__(self, *args, **kwargs):
        self.seeker, self.goal = (0, 0), (self.UPPER_BOUND - 1, self.UPPER_BOUND - 1)
        self.info = {"seeker": self.seeker, "goal": self.goal}
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.UPPER_BOUND**2)

    def reset(self):
        """Reset seeker position and return observations."""
        self.seeker = (0, 0)
        return self.get_observation()

    def get_observation(self):
        """Encode the seeker position as integer"""
        row, col = self.seeker
        return self.UPPER_BOUND * row + col

    def get_reward(self):
        """Reward finding the goal"""

        if self.seeker == self.goal:
            return 1

        # Punish being against a wall
        if self.seeker != (0, 0):
            if self.seeker[0] == 0:
                return -0.1
            if self.seeker[1] == 0:
                return -0.1
            if self.seeker[0] == self.UPPER_BOUND - 1:
                return -0.1
            if self.seeker[1] == self.UPPER_BOUND - 1:
                return -0.1

        return 0

    def is_done(self):
        """We're done if we found the goal"""
        return self.seeker == self.goal

    def step(self, action):
        """Take a step in a direction and return all available information"""
        if action == 0:  # move down
            self.seeker = (
                min(self.seeker[0] + 1, self.UPPER_BOUND - 1),
                self.seeker[1],
            )
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (
                self.seeker[0],
                min(self.seeker[1] + 1, self.UPPER_BOUND - 1),
            )
        else:
            raise ValueError("Invalid action")

        obs = self.get_observation()
        rew = self.get_reward()
        done = self.is_done()

        return obs, rew, done, self.info

    def render(self, *args, **kwargs):
        """Render the environment, e.g., by printing its representation."""
        os.system("clear")

        grid = [
            ["| " for _ in range(self.UPPER_BOUND)] + ["|\n"]
            for _ in range(self.UPPER_BOUND)
        ]
        grid[self.goal[0]][self.goal[1]] = "|G"
        grid[self.seeker[0]][self.seeker[1]] = "|S"
        print("".join(["".join(row) for row in grid]))


class GymEnvironment(Environment, gym.Env):
    def __init__(self, *args, **kwargs):
        """Make our original Environment a gym `Env`"""
        super().__init__(*args, **kwargs)


gym_env = GymEnvironment()
