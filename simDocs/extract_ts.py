import psycopg2
from psycopg2 import sql
import psycopg2.extensions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
load_ts = True


def ts_extraction(table_name, failure_collection):
    drop_query = sql.SQL("drop table if exists {table_name}").format(table_name=sql.Identifier(table_name))
    cur.execute(drop_query)
    cur.execute("create table if not exists {} (index serial primary key)".format(table_name))

    ts_lens = []
    for i in range(len(failure_collection)):
        ts_lens.append(len(failure_collection[i]))

    iso_pc_ar = np.empty((max(ts_lens), len(failure_collection)))
    for i in range(len(failure_collection)):
        iso_pc_ar[:len(failure_collection[i]), i] = failure_collection[i]

    for i in range(len(failure_collection)):
        cur.execute("ALTER TABLE {} ADD COLUMN ts{} float8;".format(table_name, i + 1))
        con.commit()

    for i in range(iso_pc_ar.shape[0]):
        values = list(iso_pc_ar[i, :])

        table_col_names = ['ts{}'.format(j + 1) for j in range(iso_pc_ar.shape[1])]

        col_names = sql.SQL(', ').join(sql.Identifier(n) for n in table_col_names)
        place_holders = sql.SQL(', ').join(sql.Placeholder() * len(table_col_names))
        query_base = sql.SQL("insert into {table_name} ({col_names}) values ({values})").format(
            table_name=sql.Identifier(table_name),
            col_names=col_names,
            values=place_holders)

        cur.execute(query_base, values)
    print(table_name, " created")

def degradation_isolation(in_df):
    ag_deg = []
    for j in range(len(in_df.columns)):
        intervals = []
        for i in range(len(in_df) - 1):
            if in_df.iloc[i + 1, j] > in_df.iloc[i, j]:
                # print("Condition_true")
                intervals.append(i + 1)
        # print(intervals)

        for i in range(len(intervals) - 1):
            ag_deg.append(np.asarray(in_df.iloc[intervals[i]:intervals[i + 1], j]))
        return ag_deg

isolated_pusher_bearing = []
isolated_pusher_coil = []
isolated_hover_bearing = []
isolated_hover_coil = []

# create table to store timeseries
# connect to postgres
con = psycopg2.connect(
    host="localhost",
    database="uavdatabase",
    user="postgres",
    password="kurabiyeci54"
)
# cursor
cur = con.cursor()


if load_ts == True:
    for k in range(10):
        print("Collection Degradation Data of UAV Number ", k + 1)
        # raw_df = pd.read_csv("data/complete_health_uav{}.csv".format(k+1), header=None)

        # hover_bearing = raw_df.iloc[:, 5:9]
        # print(hover_bearing)
        hover_bearing_df = pd.read_sql_query('select hb1, hb2, hb3, hb4 from "uav{}"'.format(k + 1), con=con)
        hover_bearing_df.drop_duplicates(inplace=True, ignore_index=True)

        # hover_coil = raw_df.iloc[:, 13:17]
        hover_coil_df = pd.read_sql_query('select hc1, hc2, hc3, hc4 from "uav{}"'.format(k + 1), con=con)
        hover_coil_df.drop_duplicates(inplace=True, ignore_index=True)

        # pusher_health = raw_df.iloc[:, 21:24]
        pusher_bearing_df = pd.read_sql_query('select psb from "uav{}"'.format(k + 1), con=con)
        pusher_bearing_df.drop_duplicates(inplace=True, ignore_index=True)
        pusher_coil_df = pd.read_sql_query('select psc from "uav{}"'.format(k + 1), con=con)
        pusher_coil_df.drop_duplicates(inplace=True, ignore_index=True)

        single_uav_hover_bearing = degradation_isolation(hover_bearing_df)
        isolated_hover_bearing.extend(single_uav_hover_bearing)

        single_uav_hover_coil = degradation_isolation(hover_coil_df)
        isolated_hover_coil.extend(single_uav_hover_coil)

        single_uav_pusher_bearing = degradation_isolation(pusher_bearing_df)
        isolated_pusher_bearing.extend(single_uav_pusher_bearing)

        single_uav_pusher_coil = degradation_isolation(pusher_coil_df)
        isolated_pusher_coil.extend(single_uav_pusher_coil)


ts_extraction(table_name="hb_ts", failure_collection=isolated_hover_bearing)
ts_extraction(table_name="hc_ts", failure_collection=isolated_hover_coil)
ts_extraction(table_name="pb_ts", failure_collection=isolated_pusher_bearing)
ts_extraction(table_name="pc_ts", failure_collection=isolated_pusher_coil)

con.commit()
cur.close()