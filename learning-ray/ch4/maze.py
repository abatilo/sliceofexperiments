from ray.rllib.algorithms.dqn import DQNConfig

config = (
    DQNConfig()
    .environment("maze_gym_env.GymEnviornment")
    .rollouts(num_rollout_workers=2)
)
