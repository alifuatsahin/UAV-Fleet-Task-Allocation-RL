import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from operator import add
from datetime import date
import os


def usage_plot():
    # create folder to save plot
    today = date.today()
    if not os.path.exists("plots/{}".format(today)):
        os.makedirs("plots/{}".format(today))

    labels = ['UAV 1', 'UAV 2', 'UAV 3', 'UAV 4', 'UAV 5', 'UAV 6', 'UAV 7', 'UAV 8', 'UAV 9', 'UAV 10']
    X = np.arange(len(labels))
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(9)
    ax.set_ylabel('Dwell Time\nin Relation to absolute Time')
    ax.set_xticks(X+0.2)
    ax.set_xticklabels(labels, rotation=45)

    del_times_list = []
    sar_times_list = []
    idle_times_list = []
    prep_times_list = []
    mx_times_list = []

    for i in range(10):
        print("UAV {}".format(i+1))
        con = psycopg2.connect(
            host="localhost",
            database="uavdatabase",
            user="postgres",
            password="kurabiyeci54"
        )


        cur = con.cursor()
        cur.execute("select entry_time, mission_mode from uav{}".format(i+1))
        uav_util = np.asarray(cur.fetchall())
        cur.close()
        con.close()

        tot_dt = (uav_util[-1, 0]-uav_util[0, 0]).total_seconds()

        prep_start_times = []
        prep_end_times = []
        for j in range(len(uav_util)-1):
            if uav_util[j+1, 1] == 3 and uav_util[j, 1] == 2:
                prep_start = uav_util[j+1, 0]
                prep_start_times.append(prep_start)
            if uav_util[j, 1] == 3 and (uav_util[j+1, 1] == 0 or uav_util[j+1, 1] == 1):
                prep_end = uav_util[j+1, 0]
                prep_end_times.append(prep_end)

        prep_times = []
        for j in range(len(prep_start_times)):
            prep_time = (prep_end_times[j]-prep_start_times[j]).total_seconds()
            prep_times.append(prep_time)
        # print("Prep Time: ", sum(prep_times))

        del_start_times = []
        del_end_times = []
        for j in range(len(uav_util) - 1):
            if uav_util[j + 1, 1] == 0 and uav_util[j, 1] == 3:
                del_start = uav_util[j + 1, 0]
                del_start_times.append(del_start)
            if uav_util[j, 1] == 0 and uav_util[j + 1, 1] == 5:
                del_end = uav_util[j+1, 0]
                del_end_times.append(del_end)
        del_times = []
        for j in range(len(del_start_times)):
            del_time = (del_end_times[j] - del_start_times[j]).total_seconds()
            del_times.append(del_time)
        # print("Del Time: ", sum(del_times))

        sar_start_times = []
        sar_end_times = []
        for j in range(len(uav_util) - 1):
            if uav_util[j + 1, 1] == 1 and uav_util[j, 1] == 3:
                sar_start = uav_util[j + 1, 0]
                sar_start_times.append(sar_start)
            if (uav_util[j, 1] == 1 or uav_util[j, 1] == 4) and uav_util[j + 1, 1] == 5:
                sar_end = uav_util[j+1, 0]
                sar_end_times.append(sar_end)
        sar_times = []
        for j in range(len(sar_start_times)):
            sar_time = (sar_end_times[j] - sar_start_times[j]).total_seconds()
            sar_times.append(sar_time)
        # print("SAR Time: ", sum(sar_times))

        mx_start_times = []
        mx_end_times = []
        for j in range(len(uav_util) - 1):
            if (uav_util[j, 1] == 1 or uav_util[j, 1] == 4 or uav_util[j, 1] == 0) and uav_util[j+1, 1] == 5:
                mx_start = uav_util[j + 1, 0]
                mx_start_times.append(mx_start)
            if  uav_util[j, 1] == 5 and uav_util[j + 1, 1] == 2:
                mx_end = uav_util[j+1, 0]
                mx_end_times.append(mx_end)
        mx_times = []
        for j in range(len(mx_start_times)):
            mx_time = (mx_end_times[j] - mx_start_times[j]).total_seconds()
            mx_times.append(mx_time)
        # print("MX Time: ", sum(mx_times))

        idle_start_times = []
        idle_end_times = []
        for j in range(len(uav_util) - 1):
            if uav_util[j, 1] == 5 and uav_util[j + 1, 1] == 2:
                idle_start = uav_util[j, 0]
                idle_start_times.append(idle_start)
            if uav_util[j, 1] == 2 and uav_util[j + 1, 1] == 3:
                idle_end = uav_util[j+1, 0]
                idle_end_times.append(idle_end)
        idle_times = []
        idle_start_times = idle_start_times[:-1]
        idle_end_times = idle_end_times[1:]
        for j in range(len(idle_start_times)):
            idle_time = (idle_end_times[j] - idle_start_times[j]).total_seconds()
            idle_times.append(idle_time)
        # print("Idle Time: ", sum(idle_times))

        print("Total Time: ",
              sum(prep_times)/tot_dt + sum(del_times)/tot_dt + sum(sar_times)/tot_dt + sum(mx_times)/tot_dt + sum(idle_times)/tot_dt)

        prep_perc = sum(prep_times)/tot_dt
        prep_times_list.append(prep_perc)

        del_perc = sum(del_times)/tot_dt
        del_times_list.append(del_perc)

        sar_perc = sum(sar_times)/tot_dt
        sar_times_list.append(sar_perc)

        mx_perc = sum(mx_times)/tot_dt
        mx_times_list.append(mx_perc)

        idle_perc = sum(idle_times)/tot_dt
        idle_times_list.append(idle_perc)

    dply_ratio = (sum(sar_times_list) + sum(del_times_list))/sum(idle_times_list)
    mx_ratio = (sum(sar_times_list) + sum(del_times_list))/sum(mx_times_list)
    total_mission_t = list(map(add, del_times_list, sar_times_list))

    ax.set_title('UAV Usage - CBM\n\nDeployment Time / Idle Time: {} \nDeployment Time / Maintenance Time: {}'.format(dply_ratio, mx_ratio))
    ax.bar(X+0.05, total_mission_t, width=0.2, color="#DCDCDC", label="Total Mission Time", alpha=0.5)
    ax.bar(X+0.0, del_times_list, width=0.1, color="#005AA9", label="Delivery Mission Time")
    ax.bar(X+0.1, sar_times_list, width=0.1, color="#009D81", label="SAR Mission Time")
    ax.bar(X+0.2, idle_times_list, width=0.1, color="#F5A300", label="Idle Time")
    ax.bar(X+0.3, prep_times_list, width=0.1, color="#E6001A", label="Preparation Time")
    ax.bar(X+0.4, mx_times_list, width=0.1, color="#721085", label="Maintenance Time")

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.55))
    fig.subplots_adjust(right=0.8)
    # safe fig as png in created folder
    plt.savefig("plots/{}/01_uav_usage".format(today))
    fig.show()

# usage_plot()


