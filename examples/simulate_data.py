import numpy as np
import gridlib
import gridlib.io

# Set the argument values
k = np.array([0.005, 0.03, 0.2, 1.2, 5.9])
s = np.array([0.02, 0.05, 0.12, 0.26, 0.55])
kb = 0.2
t_int = 0.05
t_tl_all = [0.05, 0.2, 1.0, 5.0]
N = 10000

# Simulate the survival functions
data_simulated = gridlib.tl_simulation(k, s, kb, t_int, t_tl_all, N)

print(data_simulated.keys())
# print(data_simulated["5s"]["time"])

# Save the simulated data in csv file
gridlib.io.write_data_survival_function(
    "examples\data\example_simulation.csv", data_simulated
)
