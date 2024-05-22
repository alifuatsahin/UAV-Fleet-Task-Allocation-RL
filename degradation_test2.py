from fleet import Fleet
from stats import Statistics
from matplotlib import pyplot as plt


fleet = Fleet(1, seed=1342)
stats = Statistics(fleet)
N = 1000

for _ in range(N):
    for uav in fleet: uav.startMission()
    while not fleet.detectFailure():
        for uav in fleet: uav.degrade(0.5, 0.5)
        stats.step()
    fleet.reset()


# stats.plot_all_metrics(uav_index=0, x_label="Time")
# plt.show()
stats.plot_failures(True)
plt.show()
