import os
import random
import time
from pprint import pprint

import numpy as np
import ray

ray.init()


class Discrete:
    def __init__(self, num_actions: int):
        """
        Discrete action space for num_actions.
        Discrete(4) can be used as encoding moving in one of the cardinal directions.
        """
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n - 1)


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
            self.seeker = (min(self.seeker[0] + 1, 4), self.seeker[1])
        elif action == 1:  # move left
            self.seeker = (self.seeker[0], max(self.seeker[1] - 1, 0))
        elif action == 2:  # move up
            self.seeker = (max(self.seeker[0] - 1, 0), self.seeker[1])
        elif action == 3:  # move right
            self.seeker = (self.seeker[0], min(self.seeker[1] + 1, 4))
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


class Policy:
    def __init__(self, env):
        """
        A Policy suggests actions based on the current state.
        We do this by tracking the value of each state-action pair.
        """
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)]
            for _ in range(env.observation_space.n)
        ]
        self.action_space = env.action_space

    def get_action(self, state, explore=True, epsilon=0.1):
        """Explore randomly or exploit the best value currently available."""
        if explore and random.uniform(0, 1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])


class Simulation:
    def __init__(self, env):
        """Simulates rollouts of an environment, given a policy to follow."""
        self.env = env

    def rollout(self, policy, render=False, explore=True, epsilon=0.1, sleep=0.1):
        """Return experiences for a policy rollout."""
        experiences = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experiences.append([state, action, reward, next_state])
            state = next_state
            if render:
                time.sleep(sleep)
                self.env.render()

        return experiences


def update_policy(policy, experiences, weight=0.1, discount_factor=0.9):
    """Updates a given policy with a list of (state, action, reward, next_state) experiences."""
    for state, action, reward, next_state in experiences:
        next_max = np.max(policy.state_action_table[next_state])
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + weight * (
            reward + discount_factor * next_max
        )
        policy.state_action_table[state][action] = new_value


def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9):
    """Training a policy by updating it with rollout experiences."""
    policy = Policy(env)
    sim = Simulation(env)

    for i in range(num_episodes):
        experiences = sim.rollout(policy, render=False, explore=True, epsilon=0.1)
        update_policy(policy, experiences, weight, discount_factor)

    return policy


def evaluate_policy(env, policy, num_episodes=10):
    simulation = Simulation(env)
    steps = 0

    for _ in range(num_episodes):
        experiences = simulation.rollout(policy, render=True, explore=False)
        steps += len(experiences)

    print(
        f"{steps / num_episodes} steps on average for a total of {num_episodes} episodes."
    )

    return steps / num_episodes


@ray.remote
class SimulationActor(Simulation):
    """Ray actor for Simulation."""

    def __init__(self):
        env = Environment()
        super().__init__(env)


def train_policy_parallel(env, num_episodes=1000, num_simulations=4):
    """Parallel policy training function."""
    policy = Policy(env)
    simulations = [SimulationActor.remote() for _ in range(num_simulations)]
    policy_ref = ray.put(policy)
    for _ in range(num_episodes):
        experiences = [sim.rollout.remote(policy_ref) for sim in simulations]

        while len(experiences) > 0:
            finished, experiences = ray.wait(experiences)
            for xp in ray.get(finished):
                update_policy(policy, xp)

    return policy


environment = Environment()
# untrained_policy = Policy(environment)
start_time = time.time()
p = train_policy(environment)
# p = train_policy_parallel(environment)
end_time = time.time()

sim = Simulation(environment)
exp = sim.rollout(p, render=True, sleep=0.2)

print(f"Training took {end_time - start_time} seconds.")
evaluate_policy(environment, p)
