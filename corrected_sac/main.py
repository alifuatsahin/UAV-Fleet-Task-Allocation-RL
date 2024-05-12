import gym
import numpy as np
import torch as th
import itertools

from agent import Agent
from buffer import ReplayBuffer
from UAV_gym_env import UAVGymEnv

# env = gym.make("HalfCheetah-v4")
# env.action_space.seed(1234)

env = UAVGymEnv(uav_number=3)
env.seed(1234)

th.manual_seed(1234)

# Agent
agent = Agent(env=env, 
            hidden_dim=64,
            batch_size=512,
            alpha=0.8,
            gamma=0.99,
            tau=0.01,
            lr=0.0003,
            update_interval=1,
            auto_entropy=True,
            policy="Dirichlet")

# Memory
memory = ReplayBuffer(capacity=50000, seed=1234)

# Training Loop
total_timesteps = 0
updates = 0
start_steps = 10000

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
            agent.update_parameters(memory, agent.batch_size, updates)
            updates += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_timesteps += 1
        total_timesteps += 1

        # if episode_timesteps == max_episode_steps:
        #     truncated = True
        # else:
        #     truncated = False
        done = terminated or truncated

        memory.push(state, action, reward, next_state, terminated)

        state = next_state
        # print("Total Timesteps: {} Episode Timesteps: {} Reward: {}".format(total_timesteps, episode_timesteps, episode_reward))

    print("Total Timesteps: {} Episode Num: {} Episode Timesteps: {} Reward: {}".format(total_timesteps, i, episode_timesteps, episode_reward))