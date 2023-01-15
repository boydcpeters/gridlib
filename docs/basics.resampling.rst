.. _basics.resampling:

Resampling
==========
To estimate the accuracy and precision of a GRID results, you can analyze a set of
randomly chosen values from the measured survival time distributions and repeat this
process a certain number of times. This can be done with
:py:func:`~gridlib.resample_and_fit`.

This function works very similar to the other fitting functions, but you need to provide
some extra information. To perform the resampling you need to define the parameters as
described :ref:`here <basics.parameters>`. You will also need to define the number of
times you want to perform the resampling (``n``), the percentage of data you want to use
to create a random set (``perc``) and the fitting mode (``fit_mode``). Furthermore, you
can perform the resampling in a multiprocessed way or in a sequential way. To perform
the resampling in a multiprocessed way (which is a faster) you need to set
``multiprocess_flag`` to ``True``, and you can then also set the maximum number of
workers, which is limited and defaults to the number of logical cores on your pc - 1.

.. warning::
    If you want to perform the resampling in a multiprocessed manner then you need
    to encapsulate :py:func:`~gridlib.resample_and_fit` in a specific type of
    if-statement, namely the following:

    .. code-block:: python
        
        import gridlib

        if __name__ == "__main__":
            fit_result_full, fit_results_resampled = gridlib.resample_and_fit(...)


Here is an example, where we perform 200 resamples with sets of 80% randomly chosen
data points and perform GRID fitting on this. This resampling and fitting is done in a
multiprocessed way. See the example:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import gridlib
    import gridlib.io
    import gridlib.plot

    # if __name__ == "__main__" is required for the multiprocessing in the
    # resampling_grid function to work.
    if __name__ == "__main__":  # required for multiprocessing, not required for sequential

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