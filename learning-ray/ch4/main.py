import time

import gymnasium as gym
from maze_gym_env import GymEnvironment
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

config = (
    PPOConfig()
    .environment("BipedalWalker-v3")
    .rollouts(num_rollout_workers=16, create_env_on_local_worker=True)
)

algo = config.build()

for i in range(10000):
    result = algo.train()
    print("Iteration:", i)
    print("Episode reward max:", result["episode_reward_max"])
    print("Episode reward min:", result["episode_reward_min"])
    print("Episode reward mean:", result["episode_reward_mean"])
    print()
    # print(pretty_print(result))

checkpoint = algo.save()

# algo = Algorithm.from_checkpoint(
#     "/home/aaron/ray_results/DQN_GymEnvironment_2023-03-19_18-18-55tkp8v2jf/checkpoint_001000"
# )

env = gym.make("BipedalWalker-v3", render_mode="human")
terminated = truncated = False
observations, info = env.reset()

while True:
    env.render()
    action = algo.compute_single_action(observations)
    observations, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observations, info = env.reset()

env.close()
print(checkpoint)
