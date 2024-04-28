import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date

def real_t_qual():
    boxplot_data = []
    today = date.today()

    for i in range(10):
        plt.figure(figsize=(8, 6), dpi=80)
        print(i)
        con = psycopg2.connect(
            host="localhost",
            database="uavdatabase",
            user="postgres",
            password="kurabiyeci54"
        )

        cur = con.cursor()
        cur.execute("select entry_time, no_missions from uav{}".format(i+1))
        uav_timestamp = np.asarray(cur.fetchall())
        cur.close()
        con.close()

        no_missions = uav_timestamp[-1, -1]
        timesteps = []
        for j in range(no_missions):
            single_section = uav_timestamp[np.where(uav_timestamp[:, 1] == j+1)][:, 0]
            diff_time = np.diff(single_section)
            diff_time_array = np.zeros((len(single_section)))
            for j in range(len(diff_time)):
                diff_time_array[j] = diff_time[j].total_seconds()
            timesteps.extend(diff_time_array)

        timesteps = np.asarray(timesteps)
        # timesteps = timesteps[timesteps != 0]

        ts_mean = np.round(timesteps.mean(), 4)
        ts_std = np.round(timesteps.std(), 4)

        outliers = [np.nan if (ts_mean-ts_std) < k < (ts_mean+ts_std) else k for k in timesteps]
        outliers = pd.DataFrame(outliers)
        outliers.dropna(inplace=True)
        outlier_idx = outliers.index
        print(outlier_idx)

        insides = pd.DataFrame(timesteps)
        insides.drop(outlier_idx, inplace=True)

        plt.plot(insides, 'C0.', alpha=0.1)
        plt.plot(outliers, 'C1.', alpha=0.1)
        plt.hlines(0.06, xmin=0, xmax=len(timesteps), colors='k')
        plt.ylim(0, 0.1)
        plt.xlabel("Entries into database")
        plt.ylabel("Timedelta of entries")
        plt.title("Real Time Quality of Simulation - UAV {}\nMEAN: {} - STD: {}\n{}% of Entries outside of STD".format(i+1, ts_mean, ts_std, np.round(100*(len(outliers)-1)/len(timesteps), 2)))
        plt.savefig("plots/{}/0{}_rt_qual_{}".format(today, 6+i, i+1))
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        boxplot_data.append(timesteps)

    plt.boxplot(boxplot_data, showfliers=False)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['UAV 1', 'UAV 2', 'UAV 3', 'UAV 4', 'UAV 5',
                                                 'UAV 6', 'UAV 7', 'UAV 8', 'UAV 9', 'UAV 10', ])
    plt.savefig("plots/{}/16_rt_qual_box".format(today))
    plt.show(block=False)
    plt.pause(1)
    plt.close()

# real_t_qual()
