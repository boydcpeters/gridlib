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

    # Set the parameters for the GRID fitting
    parameters = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    # Perform the resampling, the number of resamplings is set to 200, the percentage
    # of data to use per resampling is set to 80% and the fitting mode is set to the
    # GRID fitting procedure.
    fit_result_full, fit_results_resampled = gridlib.resample_and_fit(
        parameters,
        data,
        n=200,
        perc=0.8,
        fit_mode="grid",
        multiprocess_flag=True,
    )

    # Uncomment the next lines and change the path str to the preferred path to save the
    # fit results
    # gridlib.io.write_data_grid_resampling(
    #     "path/to/file_resampling_data.mat", fit_result_full, fit_results_resampled
    # )

    # Plot the resampled data
    fig1, ax1 = gridlib.plot.event_spectrum_heatmap(
        fit_result_full, fit_results_resampled
    )
    fig2, ax2 = gridlib.plot.state_spectrum_heatmap(
        fit_result_full, fit_results_resampled
    )

    # Set the titles
    ax1.set_title("Resampling event spectrum")
    ax2.set_title("Resampling state spectrum")

    plt.show()
