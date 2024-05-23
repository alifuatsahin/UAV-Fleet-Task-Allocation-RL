import torch as th
import numpy as np
from matplotlib import pyplot as plt

from agent import Agent
from UAV_gym_env import UAVGymEnv
import gym

th.manual_seed(2124)

env = UAVGymEnv(uav_number=4, max_distance=200)

agent = Agent(env=env,
                hidden_dim=[256, 256],
                batch_size=256,
                alpha=0.01,
                gamma=0.99,
                tau=0.005,
                lr=0.0001,
                update_interval=1,
                auto_entropy=True,
                policy="Dirichlet")

path = "logs/best_training/model.pt"

agent.load_checkpoint(path)

episode_reward = 0
episode_timesteps = 0
done = False
state, _ = env.reset()

while not done:
    action = agent.get_action(state)
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