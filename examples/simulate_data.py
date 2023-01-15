# Import the required libraries
import numpy as np
import gridlib
import gridlib.io

if __name__ == "__main__":  # not required, but recommended

    # Set the parameters for the simulation
    k = np.array([0.005, 0.03, 0.2, 1.2, 5.9])  # decay rates
    s = np.array([0.02, 0.05, 0.12, 0.26, 0.55])  # amplitudes
    kb = 0.3  # photobleaching rate
    t_int = 0.05  # integration time
    t_tl_all = [0.05, 0.2, 1.0, 5.0]  # all the time-lapse time for which to simulate
    N = 5000  # number of data points for every time-lapse time, can also be a sequence

    # Simulate the survival functions
    data_simulated = gridlib.tl_simulation(k, s, kb, t_int, t_tl_all, N)

    # Save the simulated data in csv file
    gridlib.io.write_data_survival_function(
        "examples\data\example_simulation.csv", data_simulated
    )
