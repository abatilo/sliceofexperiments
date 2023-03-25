import random

import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer

config = {
    "env": "LunarLander-v2",
    # Change the following line to `“framework”: “tf”` to use tensorflow
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [32],
        "fcnet_activation": "linear",
    },
}

stop = {"episode_reward_mean": 195}
ray.shutdown()

ray.init(
    num_cpus=12,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)
# execute training
analysis = ray.tune.run(
    "PPO",
    config=config,
    stop=stop,
    checkpoint_at_end=True,
)

trial = analysis.get_best_trial("episode_reward_mean", "max")
checkpoint = analysis.get_best_checkpoint(trial, "episode_reward_mean", "max")
trainer = PPOTrainer(config=config)
trainer.restore(checkpoint)

env = gym.make("LunarLander-v2", render_mode="human")
observation = env.reset()
done = False

while not done:
    env.render()
    action = trainer.compute_single_action(observation)
    observation, reward, done, info = env.step(action)

env.close()
