import numpy as np
from scipy import stats

# Define the parameters of the factorial design
num_simulations = 1000
b_levels = 2
w_levels = 3

# Function to simulate data and perform ANOVA
def simulate_data(n, b_levels, w_levels):
    # Generate random data
    data = np.random.normal(loc=0, scale=1, size=(n, b_levels, b_levels, w_levels, w_levels))
    
    # Flatten data for ANOVA
    flattened_data = data.reshape((n, -1))
    
    # Perform ANOVA
    anova_results = stats.f_oneway(*[flattened_data[:, i] for i in range(flattened_data.shape[1])])
    
    return anova_results.pvalue < 0.05

# Simulate multiple experiments and calculate power
significant_results = 0
for _ in range(num_simulations):
    significant_results += simulate_data(10000, b_levels, w_levels)

power = significant_results / num_simulations
print("Statistical Power:", power)
