from fleet import Fleet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main():
    fleet = Fleet(1)
    health_data = np.array([fleet.getStats()])
    while fleet.hasFailed() == False:
        fleet.executeMission(50, [1])
        health_data = np.concatenate((health_data, [fleet.getStats()]), axis=0)

    df = pd.DataFrame(health_data)
    
    file_path = os.path.join(os.path.dirname(__file__), f'health_data.csv')
    df.to_csv(file_path)

    n = health_data.shape[1]

    for i in range(n):
        fig, ax = plt.subplots()
        if i in [0, 1, 2, 3]:
            name = f"hover_bearing_health_{i+1}"
        elif i in [4, 5, 6, 7]:
            name = f"hover_coil_health_{i-3}"
        elif i == 8:
            name = "pusher_bearing_health"
        elif i == 9:
            name = "pusher_coil_health"

        ax.plot(health_data[:, i], label=f'health stat {i+1}')
        ax.legend()
        file_path = os.path.join(os.path.dirname(__file__), f'{name}.png')
        plt.savefig(file_path)
        plt.close(fig)

    plt.show()


if __name__ == '__main__':
    main()
    