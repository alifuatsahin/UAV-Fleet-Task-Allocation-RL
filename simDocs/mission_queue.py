import numpy as np
import random

mission_queue = np.zeros((0, 4), dtype=int)  # mission id, mission type, number of uavs involved, mission length
mission_types = [0, 1]
mission_id = 1
# create missions
"""Delivery missions only use a single UAV, Reconnessaince missions require multiple uav"""
for i in range(20000): #20000 für über Nacht Simulation
    mission_type = random.choices(mission_types, [80, 20])[0]
    print(mission_id, mission_type)
    if mission_type == 0:
        no_uav = 1
        mission_len = int(np.random.randint(40, 50))
        mission_id = mission_id #int(mission_queue[-1, 0] + 1)
        new_mission = np.empty((1, 4), dtype=int)
        new_mission[0, :] = [mission_id, mission_type, no_uav, mission_len]
        mission_queue = np.append(mission_queue, new_mission, axis=0)

    if mission_type == 1:
        no_uav = np.random.randint(3, 6)
        mission_len = np.random.randint(7, 15)
        new_mission = np.empty((no_uav, 4), dtype=int)
        mission_id = mission_id #int(mission_queue[-1, 0] + 1)
        for j in range(no_uav):
            new_mission[j, :] = [mission_id, mission_type, no_uav, mission_len]
        mission_queue = np.append(mission_queue, new_mission, axis=0)

    mission_id += 1

print("Number of delivery missions: ", np.count_nonzero(mission_queue[:, 1] == 0))
print("Number of SAR missions: ", np.count_nonzero(mission_queue[:, 1] == 1))
print("Number of total missions: ", len(mission_queue))

np.savetxt("mission_queue.csv", mission_queue, delimiter=",")
