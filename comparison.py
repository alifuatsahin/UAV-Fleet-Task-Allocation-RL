import torch as th
import numpy as np
from matplotlib import pyplot as plt

from agent import Agent
from UAV_gym_env import UAVGymEnv
import gym

env1 = UAVGymEnv(uav_number=4, max_distance=200)
env2 = UAVGymEnv(uav_number=4, max_distance=200)

agent = Agent(env=env2,
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

def baseline_strategy(state, uav_number):
    attr_length = 10
    state = np.stack(state, axis=-1)
    lowest_healths = []

    for i in range(uav_number):
        uav_health = state[attr_length*i:attr_length*(i+1)]
        lowest_healths.append(uav_health.min(axis=0))

    # action = th.softmax(th.tensor(lowest_healths), dim=0)
    action = th.tensor(np.ones(uav_number)/uav_number)
    return action



done1 = False
done2 = False
state1, _ = env1.reset()
state2, _ = env2.reset()

while not done1 and not done2:

    if not done1:
        action1 = baseline_strategy(state1, 4)
        next_state1, reward1, terminated1, truncated1, info1 = env1.step(action1)
        done1 = terminated1 or truncated1
        state1 = next_state1

    if not done2:
        action2 = agent.get_action(state2)
        next_state2, reward2, terminated2, truncated2, info2 = env2.step(action2)
        done2 = terminated2 or truncated2
        state2 = next_state2



plt.figure()
env1.plot_lowest_healths()
env2.plot_lowest_healths()
plt.figure()
env1.plot_flown_distances(show_legend=False)
env2.plot_flown_distances(show_legend=False)
plt.show()