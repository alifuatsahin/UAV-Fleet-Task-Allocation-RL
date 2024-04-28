import psycopg2.extensions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler


timewindow = 300
step = 20
# create connection to DB
con = psycopg2.connect(
    host="localhost",
    database="set_database_name",
    user="postgres",
    password="set_password"
)
# cursor
cur = con.cursor()

#
failures = ["hb", "hc", "pb", "pc"]
for k in failures:

    # extract samples from DB to train LSTM
    ts = pd.read_sql_query('select ts1, ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, ts11 , ts12, ts13, ts14, ts15, ts16, ts17, ts18, ts19, ts20 from "{}_ts"'.format(k), con=con)
    for i in range(ts.shape[1]):
        ts_array = np.asarray(ts.iloc[:, i])
        ts_array = ts_array[ts_array != 0]
        no_samples = int(len(ts_array) / timewindow)
        discard_len = len(ts_array) - no_samples * timewindow  # discard first timesteps to get samples from time close to failure
        ts_array = ts_array[discard_len:]

        healthy_std = np.std(ts_array[0:step])
        for j in range(int(len(ts_array)/step)):
            cur_std = np.std(ts_array[j*step:(j+1)*step])

            if abs(cur_std-healthy_std) > healthy_std+healthy_std*0.01:
                fault_time = step*j
                break

        label = np.empty((len(ts_array)))
        label[:fault_time] = 1
        label[fault_time:] = 0
        plt.plot(np.arange(0, fault_time, 1), ts_array[:fault_time], 'g')
        plt.plot(np.arange(fault_time, len(ts_array), 1), ts_array[fault_time:], 'r')
        plt.plot(label)
        plt.show()
