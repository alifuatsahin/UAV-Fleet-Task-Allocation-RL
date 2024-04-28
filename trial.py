import gym
import numpy as np
from SAC import SAC
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v4')
    agent = SAC(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)