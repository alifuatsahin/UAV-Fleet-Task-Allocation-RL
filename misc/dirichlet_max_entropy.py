import torch
from torch.distributions import Dirichlet, Normal
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def compute_entropy(alpha_value, max_action_space_size):
    action_space_sizes = list(range(2, max_action_space_size + 1))
    entropies = []

    for size in action_space_sizes:
        alpha = torch.tensor([alpha_value] * size)
        dist = Dirichlet(alpha)
        entropy = dist.entropy().item()
        entropies.append(entropy)

    return np.array(action_space_sizes), np.array(entropies)


def compute_entropy_gaussian(mean_value, sigma_value, max_action_space_size):
    action_space_sizes = list(range(2, max_action_space_size + 1))
    entropies = []

    for size in action_space_sizes:
        mean = torch.tensor([mean_value] * size)
        sigma = torch.tensor([sigma_value] * size)
        dist = Normal(mean, sigma)
        entropy = dist.entropy().sum().item()
        entropies.append(entropy)

    return np.array(action_space_sizes), np.array(entropies)


def plot_entropy_vs_action_space_size(alpha_value, max_action_space_size):
    action_space_sizes, entropies = compute_entropy(alpha_value, max_action_space_size)

    # Plotting the entropies
    plt.figure(figsize=(10, 6))
    plt.plot(action_space_sizes, entropies, marker='o', linestyle='-')
    plt.xlabel('Action Space Size')
    plt.ylabel('Entropy')
    plt.title(f'Entropy of Dirichlet Policy vs. Action Space Size (alpha={alpha_value})')
    plt.grid(True)

    # Fit a curve (polynomial of degree 2) to the data
    coefficients = np.polyfit(action_space_sizes, entropies, 2)
    polynomial = np.poly1d(coefficients)
    fit_values = polynomial(action_space_sizes)
    plt.plot(action_space_sizes, fit_values, linestyle='--', color='red', label='Fit: $ax^2 + bx + c$')
    print(f'Fit: {coefficients[0]:.2f}x^2 + {coefficients[1]:.2f}x + {coefficients[2]:.2f}')
    plt.legend()
    plt.show()




def plot_entropy_vs_action_space_size_gauss(mean, sigma, max_action_space_size):
    action_space_sizes, entropies = compute_entropy_gaussian(mean, sigma, max_action_space_size)

    # Plotting the entropies
    plt.figure(figsize=(10, 6))
    plt.plot(action_space_sizes, entropies, marker='o', linestyle='-')
    plt.xlabel('Action Space Size')
    plt.ylabel('Entropy')
    plt.title(f'Entropy of Dirichlet Policy vs. Action Space Size (alpha={mean})')
    plt.grid(True)

    # Define the form of the function we want to fit
    def func(x, A, B):
        return -A * np.exp(B * x)

    # Fit the function to the data
    popt, pcov = curve_fit(func, action_space_sizes, entropies)

    # Plot the fitted function
    plt.plot(action_space_sizes, func(action_space_sizes, *popt), 'r--', label='fit: A=%5.3f, B=%5.3f' % tuple(popt))
    plt.legend()
    plt.show()

def plot_entropy_vs_action_space_size_log(alpha_value, max_action_space_size, extra):
    action_space_sizes, entropies = compute_entropy(alpha_value, max_action_space_size)

    # Plotting the entropies
    plt.figure(figsize=(10, 6))
    plt.plot(action_space_sizes, entropies, marker='o', linestyle='-')
    plt.xlabel('Action Space Size')
    plt.ylabel('Entropy')
    plt.title(f'Entropy of Dirichlet Policy vs. Action Space Size (alpha={alpha_value})')
    plt.grid(True)

    # Define the form of the function we want to fit
    def func(x, A, B,C):
        return -A * np.exp(B * x) + C

    # Fit the function to the data
    popt, pcov = curve_fit(func, action_space_sizes, entropies)
    print(func(action_space_sizes, *popt))
    # Plot the fitted function
    plt.plot(action_space_sizes, func(action_space_sizes, *popt), 'r--', label='fit: A=%5.3f, B=%5.3f, C=%5.3f' % tuple(popt))
    popt[1] = popt[1] + extra
    plt.plot(action_space_sizes, func(action_space_sizes, *popt), 'r--', label='fit: A=%5.3f, B=%5.3f, C=%5.3f' % tuple(popt))
    plt.legend()
    plt.show()


# plot_entropy_vs_action_space_size(1, 10)
# Example usage
plot_entropy_vs_action_space_size_log(alpha_value=5, max_action_space_size=10, extra=0.04)

# plot_entropy_vs_action_space_size_gauss(mean = 3, sigma = 0.5, max_action_space_size=10)