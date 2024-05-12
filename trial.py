import gym
import numpy as np
from agent import Agent
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = Agent(input_dims = env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        max_iter = 0
        while not done and max_iter < 1000:
            action = agent.choose_action(observation)
            observation_, reward, done, _ , info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.update()
            observation = observation_
            max_iter += 1
            print(score)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)