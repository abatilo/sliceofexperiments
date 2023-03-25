import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="human")
algo = PPOConfig().environment(env_name).build()


terminated = truncated = False
obs, info = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    # action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    print(terminated, truncated)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
