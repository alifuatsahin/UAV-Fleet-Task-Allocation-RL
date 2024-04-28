import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import psycopg2
from datetime import date
import os

def deg_plot():
    # create folder to save plot
    today = date.today()
    if not os.path.exists("plots/{}".format(today)):
        os.makedirs("plots/{}".format(today))


    def degradation_isolation(in_df):
        ag_deg = []
        for j in range(len(in_df.columns)):
            intervals = []
            for i in range(len(in_df)-1):
                if in_df.iloc[i+1, j] > in_df.iloc[i, j]:
                    # print("Condition_true")
                    intervals.append(i+1)
            # print(intervals)

            for i in range(len(intervals)-1):
                ag_deg.append(np.asarray(in_df.iloc[intervals[i]:intervals[i+1], j]))
            return ag_deg


    def plot_degradtaions(deg_list, title, plt_id):
        deg_len = []
        fig, ax1 = plt.subplots(figsize=(20, 10))
        ax2 = ax1.twinx()
        for i in range(len(deg_list)):
            ax1.plot(deg_list[i], 'lightgrey')
            deg_len.append(len(deg_list[i]))
        hist, bins = np.histogram(deg_len, bins=10)
        kde = stats.gaussian_kde(deg_len)
        xx = np.linspace(min(bins), max(bins), 1000)
        loc = np.asarray(deg_len).mean()
        scale = np.asarray(deg_len).std()
        pdf = stats.norm.pdf(deg_len, loc=loc, scale=scale)
        pdf_array = np.empty([len(deg_len), 2])
        pdf_array[:, 0] = deg_len
        pdf_array[:, 1] = pdf
        sorted_pdf_array = pdf_array[pdf_array[:, 0].argsort()]

        ax2.hist(x=deg_len, density=True, color='#0504aa', alpha=0.3, rwidth=0.9)
        ax2.plot(sorted_pdf_array[:, 0], sorted_pdf_array[:, 1])
        ax2.plot(xx, kde(xx), 'r')
        ax1.set_xlabel("\nTime in Minutes")
        ax1.set_ylabel("Failure Degradation\n")
        ax2.set_ylabel("\nEoL Probability")
        fig.suptitle("Degradation Characteristics and Probability of Occurrence - CBM\n\n" + title + " - appeared {} times".format(len(deg_list)))
        # safe fig as png in created folder
        plt.savefig("plots/{}/{}_{}".format(today, plt_id, title))
        plt.show()


    isolated_pusher_bearing = []
    lens_pb = []
    isolated_pusher_coil = []
    lens_pc = []
    isolated_hover_bearing = []
    lens_hb = []
    isolated_hover_coil = []
    lens_hc = []


    con = psycopg2.connect(
        host="localhost",
        database="uavdatabase",
        user="postgres",
        password="kurabiyeci54")

    for k in range(10):
        print("UAV Number ", k+1)
        # raw_df = pd.read_csv("data/complete_health_uav{}.csv".format(k+1), header=None)

        # hover_bearing = raw_df.iloc[:, 5:9]
        # print(hover_bearing)
        hover_bearing_df = pd.read_sql_query('select hb1, hb2, hb3, hb4 from "uav{}"'.format(k+1), con=con)
        hover_bearing_df.drop_duplicates(inplace=True, ignore_index=True)

        # hover_coil = raw_df.iloc[:, 13:17]
        hover_coil_df = pd.read_sql_query('select hc1, hc2, hc3, hc4 from "uav{}"'.format(k+1), con=con)
        hover_coil_df.drop_duplicates(inplace=True, ignore_index=True)

        # pusher_health = raw_df.iloc[:, 21:24]
        pusher_bearing_df = pd.read_sql_query('select psb from "uav{}"'.format(k+1), con=con)
        pusher_bearing_df.drop_duplicates(inplace=True, ignore_index=True)
        pusher_coil_df = pd.read_sql_query('select psc from "uav{}"'.format(k+1), con=con)
        pusher_coil_df.drop_duplicates(inplace=True, ignore_index=True)

        single_uav_hover_bearing = degradation_isolation(hover_bearing_df)
        lens_hb.append(len(single_uav_hover_bearing))
        isolated_hover_bearing.extend(single_uav_hover_bearing)

        single_uav_hover_coil = degradation_isolation(hover_coil_df)
        lens_hc.append(len(single_uav_hover_coil))
        isolated_hover_coil.extend(single_uav_hover_coil)

        single_uav_pusher_bearing = degradation_isolation(pusher_bearing_df)
        lens_pb.append(len(single_uav_pusher_bearing))
        isolated_pusher_bearing.extend(single_uav_pusher_bearing)

        single_uav_pusher_coil = degradation_isolation(pusher_coil_df)
        lens_pc.append(len(single_uav_pusher_coil))
        isolated_pusher_coil.extend(single_uav_pusher_coil)

    print("Hover Bearing mean: ", np.mean(lens_hb))
    print("Hover Coil mean: ", np.mean(lens_hb))
    print("Pusher Bearing mean: ", np.mean(lens_hb))
    print("Pusher Coil mean: ", np.mean(lens_hb))

    failure_modes = ["Hover Bearing Failure", "Hover Coil Failure", "Pusher Bearing Failure", "Pusher Coil Failure"]

    plot_degradtaions(isolated_hover_bearing, failure_modes[0], "02")
    plot_degradtaions(isolated_hover_coil, failure_modes[1], "03")
    plot_degradtaions(isolated_pusher_bearing, failure_modes[2], "04")
    plot_degradtaions(isolated_pusher_coil, failure_modes[3], "05")


# deg_plot()


