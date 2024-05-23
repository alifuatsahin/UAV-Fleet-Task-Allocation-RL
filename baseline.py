import torch as th
import numpy as np
from matplotlib import pyplot as plt

from agent import Agent
from UAV_gym_env import UAVGymEnv
import gym

def baseline_strategy(state, uav_number):
    attr_length = 10
    state = np.stack(state, axis=-1)
    lowest_healths = []

    for i in range(uav_number):
        uav_health = state[attr_length*i:attr_length*(i+1)]
        lowest_healths.append(uav_health.min(axis=0))

    action = th.softmax(th.tensor(lowest_healths), dim=0)
    # action = th.tensor(np.ones(uav_number)/uav_number)
    return action

env = UAVGymEnv(uav_number=4, max_distance=100)

episode_reward = 0
episode_timesteps = 0
done = False
state, _ = env.reset()

while not done:
    action = baseline_strategy(state, 4)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    episode_reward += reward
    episode_timesteps += 1

print("Missions Completed: ", episode_timesteps, "Episode Reward: ", episode_reward)

plt.figure()
env.plot_lowest_degredations()
plt.figure()
env.plot_flown_distances(show_legend=False)
plt.show()