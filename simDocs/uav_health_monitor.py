import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import os


def health_plot():
    # create folder to save plot
    today = date.today()
    if not os.path.exists("plots/{}".format(today)):
        os.makedirs("plots/{}".format(today))
    fig, axs = plt.subplots(10, figsize=(20, 15), sharex=True)

    executed_missions = []
    for i in range(10):
        con = psycopg2.connect(
            host="localhost",
            database="uavdatabase",
            user="postgres",
            password="kurabiyeci54"
        )


        cur = con.cursor()

        cur.execute('select * from uav{}'.format(i+1))
        uav_complete = pd.DataFrame(cur.fetchall())
        # uav_complete.to_csv('data/sim_complete_health_uav{}.csv'.format(i+1), sep=",", header=False)

        cur.execute("select entry_time from uav{}".format(i+1))
        uav_timestamp = np.asarray(cur.fetchall())
        cur.execute("select health_index from uav{}".format(i+1))
        uav_health_index = np.asarray(cur.fetchall())
        cur.execute("select hb1, hb2, hb3, hb4, hc1, hc2, hc3, hc4, psb, psc from uav{}".format(i+1))
        uav_components_health = np.asarray(cur.fetchall())
        cur.execute("select bat_h from uav{}".format(i+1))
        uav_bat_h = np.asarray(cur.fetchall())
        cur.execute("select no_missions from uav{}".format(i+1))
        no_missions = int(np.asarray(cur.fetchall())[-1])
        executed_missions.append(no_missions)

        cur.close()
        con.close()

        #axs[i].plot(uav_timestamp, uav_bat_h[0:len(uav_timestamp)], ".", alpha=0.2)
        axs[i].plot(uav_timestamp, uav_components_health[0:len(uav_timestamp)], alpha=0.2)
        axs[i].plot(uav_timestamp, uav_health_index[0:len(uav_timestamp)], 'k')
        axs[i].set_ylabel("UAV {}".format(i+1))
        axs[i].set_ylim([0, 1])
        axs[i].title.set_text("UAV {} - {} Missions executed".format(i+1, no_missions))
    print("Total Missions executed: ", sum(executed_missions))

    axs[9].set_xlabel("Time in min")
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Individual UAV Health Progressions - CBM", fontsize=30)
    plt.ylim((0, 1))
    # safe plot as png in created folder
    plt.savefig("plots/{}/00_uav_health_progression".format(today))
    plt.show()

# health_plot()


