from uav import UAV
from uav import UAVStats
from matplotlib import pyplot as plt
import numpy as np

def get_metric(degredations, metric):
    attr_lengths = [4, 4, 1, 1]

    start = sum(attr_lengths[:metric])
    return np.min(degredations[start:start+attr_lengths[metric]], 0)

uav = UAV(0, 123)
stats = UAVStats(uav)

uav.startMission()
degradations = []

while uav.detectFailure() == False:
    uav.degrade(0.5, 0.5)
    degredation = uav.getStats()
    degradations.append(degredation)


plot_strategy = 2

degradations = np.stack(degradations, axis=-1)

metric_vals = [get_metric(degradations, metric) for metric in stats.METRICS]
step_count = degradations.shape[1]
x = np.arange(step_count)
for metric, vals in zip(stats.METRICS, metric_vals):
    label = stats.STAT_NAMES[metric] + " health"
    if metric == 0 or metric == 1:
        label %= "s lowest"
    plt.plot(x, vals, label=label)
plt.ylabel(f"Degradations of UAV {1}")
plt.xlabel("Number of missions")
plt.legend()
plt.show()