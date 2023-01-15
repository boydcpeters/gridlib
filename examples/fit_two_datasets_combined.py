"""
File with an example workflow if you want to fit two datasets first in a combined way
and then fit them them seperately with the optimal combined decay rates to check
for differences in the event and state spectra. Here the data is simulated but
you can use this for real life data as well. For example, if you have imaged binding
molecules inside a cell and you want to combine molecules in different regions, eg.
inside and outside of the nucleus.

When using real data, you should load in the the data with
gridlib.io.read_data_survival_function() instead of performing the simulations.
"""


if __name__ == "__main__":

    # Import the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import gridlib
    import gridlib.io
    import gridlib.plot

    # Define the values required for the simulations
    k = np.array([0.005, 0.03, 0.2, 1.2, 5.9])
    kb = 0.3
    t_int = 0.05
    t_tl_all = [0.05, 0.2, 1.0, 5.0]
    N = 5000

    # Event amplitudes for simulated data 1
    s1 = np.array([0.02, 0.05, 0.12, 0.1, 0.71])

    # Event amplitudes for simulated data 2
    s2 = np.array([0.08, 0.15, 0.20, 0.17, 0.4])

    # Simulate the survival functions for the different event amplitudes
    data1 = gridlib.tl_simulation(k, s1, kb, t_int, t_tl_all, N)
    data2 = gridlib.tl_simulation(k, s2, kb, t_int, t_tl_all, N)

    # Combine the data into one dataset for the combined fitting
    data_both = gridlib.combine_data(data1, data2)

    # Threshold the survival function to remove the
    data_both = gridlib.threshold_survival_function(data_both, prob_min=10 ** (-2))

    # Plot the thresholded data
    fig1, ax1 = gridlib.plot.data_sf(data_both)

    # Define the GRID fitting parameters
    parameters_grid = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

    # Perform the GRID fitting
    fit_results_both = gridlib.fit_grid(parameters_grid, data_both, disp=True)

    # Find the peaks in the resulting GRID spectrum.
    k_peaks, s_peaks = gridlib.find_peaks(fit_results_both)
    # print(f"Decay rates: {k_peaks}, amplitudes: {s_peaks}") # uncomment to print the results

    # Plot the data agianst the GRID fit
    fig2, ax2 = gridlib.plot.data_sf_vs_grid(data_both, fit_results_both)

    # Define the GRID fitting parameters, but now with the decay rates fixed at the
    # peak positions
    parameters_peaks = {
        "k": k_peaks,
        "reg_weight": 0.01,
        "fit_a": False,
        "a_fixed": fit_results_both["grid"]["a"],
    }

    # Perform the GRID fitting, but now on the two different datasets
    fit_results1 = gridlib.fit_grid(parameters_peaks, data1, disp=True)
    fit_results2 = gridlib.fit_grid(parameters_peaks, data2, disp=True)

    # Plot the data and results for data1, and plot the event and state spectrum
    fig3, ax3 = gridlib.plot.data_sf_vs_grid(data1, fit_results1)

    fig4, ax4 = gridlib.plot.event_spectrum(
        fit_results1,
        threshold=10e-6,
        xlim=(10 ** (-3.1), 10**1.1),
        ylim=(-0.05, 0.75),
    )
    fig5, ax5 = gridlib.plot.state_spectrum(
        fit_results1,
        threshold=10e-6,
        xlim=(10 ** (-3.1), 10**1.1),
        ylim=(-0.05, 0.75),
    )

    # Set the titles
    ax3.set_title("data1 vs GRID fit")
    ax4.set_title("data1 event spectrum")
    ax5.set_title("data1 state spectrum")

    # Plot the data and results for data1, and plot the event and state spectrum
    fig6, ax6 = gridlib.plot.data_sf_vs_grid(data2, fit_results2)

    fig7, ax7 = gridlib.plot.event_spectrum(
        fit_results2,
        threshold=10e-6,
        xlim=(10 ** (-3.1), 10**1.1),
        ylim=(-0.05, 0.8),
    )
    fig8, ax8 = gridlib.plot.state_spectrum(
        fit_results2,
        threshold=10e-6,
        xlim=(10 ** (-3.1), 10**1.1),
        ylim=(-0.05, 0.8),
    )

    # Set the titles
    ax6.set_title("data2 vs GRID fit")
    ax7.set_title("data2 event spectrum")
    ax8.set_title("data2 state spectrum")

    # Show all the plots
    plt.show()

    # Uncomment the two following lines and change the path str to save the fit results
    # gridlib.io.write_fit_results("path/to/file1.mat", fit_results1)
    # gridlib.io.write_fit_results("path/to/file2.mat", fit_results2)
