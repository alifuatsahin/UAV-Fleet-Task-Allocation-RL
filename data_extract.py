import pandas as pd
import os

directory = "data/2023_BLDC_motor_degradation_simulation/a_raw_data_example"
directory = "data/2023_BLDC_motor_degradation_simulation/b_aggregated_feature_data/voltage_noise_level_1"
df_collection = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    if os.path.isfile(f):
        print("File: ", f)
        df_collection.append(pd.read_parquet(f))

df_collection[0].to_csv('data.csv', index=False)