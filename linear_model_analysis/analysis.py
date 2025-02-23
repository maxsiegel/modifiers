#%%
import numpy as np
import pandas as pd

# Memory-efficient loading
# Option 1: Load with memory mapping
results = np.load('UK_finer_feb2.npy', mmap_mode='r', allow_pickle=True)

# Define parameter ranges (assuming these are your parameters)
theta_to_test = np.linspace(0, 20, 10)  # 10 values from 0 to 20
possible_soc_terms = np.linspace(0, 1, 5)  # 5 values from 0 to 1
possible_inf_terms = np.linspace(0, 1, 5)  # 5 values from 0 to 1   
costs = np.linspace(0, 1, 5)  # 5 values from 0 to 1

# Reshape the array
results = results.reshape(len(theta_to_test), len(theta_to_test), len(theta_to_test),
                        len(possible_soc_terms), len(possible_inf_terms), len(costs),
                        len(theta_to_test), len(theta_to_test), len(theta_to_test)).transpose(0,1,2,8,7,6,5,4,3)

# Create a grid of all parameter combinations
param_names = ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 
               'costs', 'inf_terms', 'soc_terms']
param_ranges = [theta_to_test, theta_to_test, theta_to_test, 
               theta_to_test, theta_to_test, theta_to_test,
               costs, possible_inf_terms, possible_soc_terms]


# Create indices for each dimension
indices = [range(len(r)) for r in param_ranges]

# Create all combinations of indices
grid_indices = np.meshgrid(*indices, indexing='ij')
flat_indices = [g.ravel() for g in grid_indices]
grid_indices
# %%
#%%

# Create DataFrame
df = pd.DataFrame({
    name: param_ranges[i][idx] 
    for i, (name, idx) in enumerate(zip(param_names, flat_indices))
})

# Add the results column
df['result'] = results.ravel()

# Optional: Save to CSV to avoid memory issues in future
df.to_csv('results_table.csv', index=False)