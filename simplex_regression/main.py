import numpy as np
import torch as th
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from copy import deepcopy

from agent import Agent
from buffer import ReplayBuffer
from regression_gym import SimplexGymEnv

date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
ckpt_path = "logs/checkpoint_regression_gauss_{}".format(date_time)

env = SimplexGymEnv(simplex_size=4)

th.manual_seed(1234)

# Agent
agent = Agent(env=env, 
            hidden_dim=[64, 64, 64],
            batch_size=256,
            alpha=0.2,
            gamma=0.99,
            tau=0.005,
            lr=0.1,
            update_interval=1,
            auto_entropy=False,
            policy="Gaussian",)

# Memory
memory = ReplayBuffer(capacity=1000000, seed=1234)

# Plotting
qf1_loss_arr = []
qf2_loss_arr = []
policy_loss_arr = []
alpha_loss_arr = []
alpha_tlogs_arr = []
rewards = []
moving_average = 10

# Training Loop
total_timesteps = 0
updates = 0
start_steps = 0
# max_episode_steps = 10000

try:
    for i in itertools.count(1):
        episode_reward = 0
        episode_timesteps = 0
        done = False
        state, _ = env.reset()

        while not done:
            if total_timesteps < start_steps:
                action = env.action_space.sample()
                action = action/np.sum(action)
            else:
                action = agent.get_action(state)

            if len(memory) > agent.batch_size:
                qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha_tlogs = agent.update_parameters(memory, agent.batch_size, updates)
                qf1_loss_arr.append(qf1_loss)
                qf2_loss_arr.append(qf2_loss)
                policy_loss_arr.append(policy_loss)
                alpha_loss_arr.append(alpha_loss)
                alpha_tlogs_arr.append(alpha_tlogs)
                updates += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1

            done = terminated or truncated

            memory.push(state, action, reward, next_state, terminated)

            state = next_state
            # print("Total Timesteps: {} Episode Timesteps: {} Reward: {}".format(total_timesteps, episode_timesteps, episode_reward))

        # example plotting
        #env.plot_one_metric(UAVStats.PUSHER_BEARING_HEALTH, uav_index=None)
        #env.plot_one_metric(UAVStats.HOVER_BEARING_HEALTH, uav_index=None, plot_strategy=Statistics.LOWEST)

        if total_timesteps > start_steps:
            rewards.append(episode_reward)

        current_env = deepcopy(env)

        print("Total Timesteps: {} Episode Num: {} Episode Timesteps: {} Reward: {}".format(total_timesteps, i, episode_timesteps, episode_reward))

except KeyboardInterrupt:

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    agent.save_checkpoint(env_name="UAV", suffix="episodes_{}".format(episode_timesteps), ckpt_path=ckpt_path)

    data = {
        'score': rewards,
        'episode number': range(len(rewards)),
    }

    loss = {
        'qf1_loss': qf1_loss_arr,
        'qf2_loss': qf2_loss_arr,
        'policy_loss': policy_loss_arr,
        'alpha_loss': alpha_loss_arr,
        'alpha_tlogs': alpha_tlogs_arr
    }

    df = pd.DataFrame(data)
    df.to_csv(ckpt_path + '/hdata.csv', index=False)

    fig, ax = plt.subplots()
    rewards = np.convolve(rewards, np.ones(moving_average)/moving_average, 'valid')
    ax.plot(range(len(rewards)), rewards)
    ax.set(xlabel='Episode', ylabel="Score",
        title="Score vs Iterations")

    fig.savefig(ckpt_path + "/score.png")

    for name, value in loss.items():
        fig, ax = plt.subplots()
        value = np.convolve(value, np.ones(moving_average)/moving_average, 'valid')
        ax.plot(range(len(value)), value)
        ax.set(xlabel='Iterations', ylabel="{}".format(name),
            title="{} vs Iterations".format(name))

        fig.savefig(ckpt_path + "/{}.png".format(name))
