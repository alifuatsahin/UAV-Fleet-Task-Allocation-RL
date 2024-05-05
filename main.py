import numpy as np
from agent import Agent
from UAV_gym_env import UAVGymEnv
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = UAVGymEnv(uav_number=5, max_distance=100)
    agent = Agent(alpha=0.0003, beta=0.0003, 
                  input_dims=[env.observation_space.shape[0]], 
                  tau=0.005, scale=2, env=env, gamma=0.99, 
                  n_actions=env.action_space.shape[0], max_size=1000000, 
                  layer1_size=256, layer2_size=256, batch_size=256)
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []
    max_iter = 1000
    total_iter = 0
    episode = 0

    try:
        while True:
            observation = env.reset()
            done = False
            score = 0
            iter = 0
            while not done and iter < max_iter:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                score += reward
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
                iter += 1
                total_iter += 1
            avg_score = np.mean(score_history[-10:])
            score_history.append(avg_score)
            episode += 1
            if score > best_score:
                best_score = score

            print('episode ', episode, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'total_iter %d' % total_iter)

    except KeyboardInterrupt:
        data = {
            'score': score_history,
            'episode number': range(episode)
        }

        df = pd.DataFrame(data)
        df.to_csv('data.csv', index=False)

        fig, ax = plt.subplots()
        ax.plot(range(episode), score_history)
        ax.set(xlabel='Episode', ylabel='Score',
            title='Score vs Episode')

        fig.savefig("score.png")

