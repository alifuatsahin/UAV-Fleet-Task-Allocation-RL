import gym
import numpy as np
import torch as th
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent
from buffer import ReplayBuffer
from UAV_gym_env import UAVGymEnv

env = gym.make("HalfCheetah-v4")
env.action_space.seed(1234)

# env = UAVGymEnv(uav_number=3)
# env.seed(1234)

th.manual_seed(1234)

# Agent
agent = Agent(env=env, 
            hidden_dim=256,
            batch_size=256,
            alpha=0.2,
            gamma=0.99,
            tau=0.005,
            lr=0.0003,
            update_interval=1,
            auto_entropy=True,
            policy="Gaussian")

# Memory
memory = ReplayBuffer(capacity=1000000, seed=1234)

# Plotting
qf1_loss_arr = []
qf2_loss_arr = []
policy_loss_arr = []
alpha_loss_arr = []
alpha_tlogs_arr = []
rewards = []

# Training Loop
total_timesteps = 0
updates = 0
start_steps = 10000
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
        rewards.append(episode_reward)

        print("Total Timesteps: {} Episode Num: {} Episode Timesteps: {} Reward: {}".format(total_timesteps, i, episode_timesteps, episode_reward))

except KeyboardInterrupt:
    data = {
        'score': rewards,
        'episode number': range(len(rewards))
    }

    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)

    fig, ax = plt.subplots()
    ax.plot(range(len(rewards)), rewards)
    ax.set(xlabel='Episode', ylabel='Rewards',
        title='Rewards vs Episode')

    fig.savefig("score.png")
