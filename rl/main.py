import sys

import gymnasium as gym
import ray
from ray import air, tune
from ray.air.config import RunConfig, ScalingConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig

# PPOConfig().rol

tuner = tune.Tuner(
    "DQN",
    param_space={
        "env": "LunarLander-v2",
        "num_rollout_workers": 4,
        "num_cpus_per_worker": 3,
        "num_envs_per_worker": 10,
    },
    run_config=RunConfig(
        # stop={"timesteps_total": 200000},
        stop={"episode_reward_mean": 300},
    ),
)

results = tuner.fit()
best_result = results.get_best_result("episode_reward_mean", "max")

if best_result.checkpoint:
    best_checkpoint = best_result.checkpoint
    print("Best checkpoint: ", best_checkpoint)
    algo = Algorithm.from_checkpoint(best_checkpoint)
