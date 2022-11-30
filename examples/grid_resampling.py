import numpy as np
import matplotlib.pyplot as plt

import gridlib
import gridlib.io
import gridlib.plot

# if __name__ == "__main__" is required for the multiprocessing in the
# resampling_grid function to work.
if __name__ == "__main__":
    # Load the data
    data = gridlib.io.read_data_survival_function("examples/data/example1.csv")

    # Set the parameters
    parameters = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    # Perform the resampling
    fit_result_full, fit_results_resampled = gridlib.resampling_grid(
        parameters, data, n=10, perc=0.8
    )

    print(len(fit_result_full), len(fit_results_resampled))

    # Save the the resampled data
    path_save = "examples/data/example1_resampling.mat"
    gridlib.io.write_data_grid_resampling(
        path_save, fit_result_full, fit_results_resampled
    )

    # Load the resampled data (this is just for a sanity check)
    fit_result_full_2, fit_results_resampled_2 = gridlib.io.read_data_grid_resampling(
        path_save
    )

    print(len(fit_result_full_2), len(fit_results_resampled_2))

    # Plot the resampled data
    gridlib.plot.event_spectrum_heatmap(fit_result_full, fit_results_resampled)
    gridlib.plot.state_spectrum_heatmap(fit_result_full, fit_results_resampled)

    plt.show()
