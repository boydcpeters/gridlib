Fitting
=======

Here it is explained how the GRID fitting and multi-exponential fitting can be performed
with GRIDLib. However, to access all the required routines, the correct packages need
to be imported.

Import the required packages as follows:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import gridlib
    import gridlib.io
    import gridlib.plot

We see that we shorten ``numpy`` to ``np`` and ``matplotlib.pyplot`` to ``plt`` since
this is convention. However, :py:mod:`gridlib`, :py:mod:`gridlib.io`, and
:py:mod:`gridlib.plot` are not shortened.

Once the packages are imported, the data can be loaded in:

.. code-block:: python

    # Load the data
    data = gridlib.io.read_data_survival_function("/path/to/file.csv")

The survival time distributions are then stored in the ``data`` variable. For the rest
of the examples, we will use the example data provided on GitHub at
``gridlib/examples/data/example1.csv``, which has four simulated survival time
distributions. Let's say you want to perform the GRID fitting procedure and perform
single-, double- and triple-exponential fitting then you need to set the appropriate
parameters. For example:

.. code-block:: python
    
    # Set the fitting parameters
    parameters = {
        "k_min": 10**(-3),
        "k_max": 10**2,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
        "n_exp": [1, 2, 3],  # fit a 1-, 2- and 3- exponential
    }

You then need to fit, which can be done as follows:

.. code-block:: python
    
    # Perform the fitting procedures and display logging messages
    fit_results = gridlib.fit_all(parameters, data, disp=True)

We set ``disp=True``, because we want to logging messages to be displayed, set it to
``False`` to turn this off.

.. note::
    If you only want to perform the GRID fitting procedure or multi-exponential fitting,
    adjust the parameter values accordingly as explained :ref:`here <basics.parameters>`
    and use the functions ``gridlib.fit_grid()`` and ``gridlib.fit_multi_exp()``
    respectively.

Let's save the resulting fit_results in a file:

.. code-block:: python

    # Write the fit results to file of interest
    gridlib.io.write_fit_results("/path/to/file.mat", fit_results)

Now the results are safely written to the set file, it is a good idea to plot the
results and see how the resulting spectrum looks like. Let's plot both the event and the
state spectrum, and the original data vs the fitted GRID spectrum:

.. code-block:: python

    # Plot the results
    fig1, ax1 = gridlib.plot.event_spectrum(fit_results)
    fig2, ax2 = gridlib.plot.state_spectrum(fit_results)
    fig3, ax3 = gridlib.plot.data_vs_grid(data, fit_results)

    plt.show()

Full example
------------

Full example:

.. code-block:: python

    # Import the required libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import gridlib
    import gridlib.io
    import gridlib.plot

    # Load the data
    data = gridlib.io.read_data_survival_function("/path/to/file.csv")

    # Set the fitting parameters
    parameters = {
        "k_min": 10**(-3),
        "k_max": 10**2,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
        "n_exp": [1, 2, 3],  # fit a 1-, 2- and 3- exponential
    }

    # Perform the fitting procedures and display logging messages
    fit_results = gridlib.fit_all(parameters, data, disp=True)

    # Write the fit results to file of interest
    gridlib.io.write_fit_results("/path/to/file.mat", fit_results)

    # Plot the results
    fig1, ax1 = gridlib.plot.event_spectrum(fit_results)
    fig2, ax2 = gridlib.plot.state_spectrum(fit_results)
    fig3, ax3 = gridlib.plot.data_vs_grid(data, fit_results)

    plt.show()