import psycopg2.extensions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create connection to DB
con = psycopg2.connect(
    host="localhost",
    database="uavdatabase",
    user="postgres",
    password="kurabiyeci54"
)
# cursor
cur = con.cursor()

#
failures = ["hb", "hc", "pb", "pc"] #"hb", "hc", "pb",22
for k in failures:

    # extract samples from DB to train LSTM
    ts = pd.read_sql_query('select * from "{}_ts"'.format(k), con=con)
    ts_lens = []
    for i in np.arange(1, ts.shape[1], 1):
        single_ts = ts.iloc[:, i]
        if single_ts.iloc[0] >= 0.95:
            single_ts = single_ts.loc[~(single_ts == 0)]
            plt.plot(single_ts)
            ts_lens.append(len(single_ts))
    mx_interval = int(min(ts_lens) - min(ts_lens)*0.01)
    print(mx_interval)
    plt.vlines(mx_interval, 0, 1)
    plt.hlines(0.95, 0, 100)
    plt.show()