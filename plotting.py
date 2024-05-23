import pandas as pd
import matplotlib.pyplot as plt

def plot_rolling_mean_variance(paths, labels, transform=False):
    plt.figure(figsize=(10, 6))
    for path, label in zip(paths, labels):
        # Load the CSV file
        df = pd.read_csv(path)

        # Select the first 50,000 rows
        df_subset = df.head(50000)

        # Transform the data
        if transform:
            df_subset['transformed_score'] = 1 - df_subset['score'] / 1000
        else:
            df_subset['transformed_score'] = df_subset['score']

        # Calculate the rolling mean and rolling standard deviation for 1000 steps
        df_subset['rolling_mean'] = df_subset['transformed_score'].rolling(window=1000).mean()
        df_subset['rolling_std'] = df_subset['transformed_score'].rolling(window=1000).std()

        # Print the final rolling mean and variance
        print(f"File {path} - Rolling Mean: {df_subset['rolling_mean'].iloc[-1]}, Rolling Variance: {df_subset['rolling_std'].iloc[-1]**2}")

        # Plotting the data
        

        # Plot rolling mean and variance as transparent area
        plt.plot(df_subset['episode number'], df_subset['rolling_mean'], label=label)
        plt.fill_between(df_subset['episode number'],
                         df_subset['rolling_mean'] - df_subset['rolling_std'],
                         df_subset['rolling_mean'] + df_subset['rolling_std'],
                         alpha=0.1)

        plt.xlabel('Episode Number')
        plt.ylabel('MSE')
        plt.title('MSE vs Episode Number')
        plt.legend()

    plt.show()

# Example usage:
paths = ['logs/checkpoint_regression_dirichlet_001_2024-05-23_01-03-06/hdata.csv',
         'logs/checkpoint_regression_gauss_001_2024-05-23_02-21-19/hdata.csv']
labels = ['Dirichlet', 'Gaussian']

plot_rolling_mean_variance(paths, labels, transform = True)
