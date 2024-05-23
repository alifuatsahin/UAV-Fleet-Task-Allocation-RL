import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
df1 = pd.read_csv('logs/checkpoint_regression_dirichlet_001_2024-05-23_01-03-06/hdata.csv')
df2 = pd.read_csv('logs/checkpoint_regression_gauss_001_2024-05-23_02-21-19/hdata.csv')

# Select the first 50,000 rows
df1_subset = df1.head(50000)
df2_subset = df2.head(50000)

# Transform the data
df1_subset['transformed_score'] = 1 - df1_subset['score'] / 1000
df2_subset['transformed_score'] = 1 - df2_subset['score'] / 1000

# Calculate the rolling mean and rolling standard deviation for 1000 steps
df1_subset['rolling_mean'] = df1_subset['transformed_score'].rolling(window=1000).mean()
df1_subset['rolling_std'] = df1_subset['transformed_score'].rolling(window=1000).std()

df2_subset['rolling_mean'] = df2_subset['transformed_score'].rolling(window=1000).mean()
df2_subset['rolling_std'] = df2_subset['transformed_score'].rolling(window=1000).std()

# Print the final rolling mean and variance
print(f"File 1 - Rolling Mean: {df1_subset['rolling_mean'].iloc[-1]}, Rolling Variance: {df1_subset['rolling_std'].iloc[-1]**2}")
print(f"File 2 - Rolling Mean: {df2_subset['rolling_mean'].iloc[-1]}, Rolling Variance: {df2_subset['rolling_std'].iloc[-1]**2}")

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot rolling mean and variance as transparent area for File 1
plt.plot(df1_subset['episode number'], df1_subset['rolling_mean'], label='Dirichlet Policy')
plt.fill_between(df1_subset['episode number'],
                 df1_subset['rolling_mean'] - df1_subset['rolling_std'],
                 df1_subset['rolling_mean'] + df1_subset['rolling_std'],
                 color='blue', alpha=0.1)

# Plot rolling mean and variance as transparent area for File 2
plt.plot(df2_subset['episode number'], df2_subset['rolling_mean'], label='Gaussian Policy')
plt.fill_between(df2_subset['episode number'],
                 df2_subset['rolling_mean'] - df2_subset['rolling_std'],
                 df2_subset['rolling_mean'] + df2_subset['rolling_std'],
                 color='orange', alpha=0.1)

plt.xlabel('Episode Number')
plt.ylabel('MSE')
plt.title('MSE vs Episode Number')
plt.legend()

plt.show()
