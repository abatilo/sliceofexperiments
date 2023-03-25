import os

import gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

env = gym.make("LunarLander-v2")
states = env.observation_space.shape[0]
actions = env.action_space.n
episodes = 10


def buildModel(statez, actiones):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1, statez)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actiones, activation="linear"))
    return model


def buildAgent(modell, actionz):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=modell,
        memory=memory,
        policy=policy,
        nb_actions=actionz,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )
    return dqn


model = buildModel(states, actions)
DQN = buildAgent(model, actions)

# if file dqn_weights.h5f exists, load it, otherwise train the model
if os.path.isfile("dqn_weights.h5f"):
    DQN.load_weights("dqn_weights.h5f")
else:
    DQN.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=["mae"])
    DQN.fit(env, nb_steps=50000, visualize=True, verbose=1)
    DQN.save_weights("dqn_weights.h5f", overwrite=True)

scores = DQN.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history["episode_reward"]))

_ = DQN.test(env, nb_episodes=15, visualize=True)
