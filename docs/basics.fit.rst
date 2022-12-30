.. _basics.fit:

Data fitting
============

Currently, it is possible to perform GRID fitting and multi-exponential fitting.


Import libraries
----------------

Before we can do anything, we need to import the required modules.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import gridlib
    import gridlib.io
    import gridlib.plot

We can see that we need the :py:mod:`gridlib`, :py:mod:`gridlib.io`, and the
:py:mod:`gridlib.plot` module.


Data loading
------------

Once the packages are imported, the data can be loaded in:

.. code-block:: python

    # Load the data
    data = gridlib.io.read_data_survival_function("/path/to/file.csv")

The survival time distributions are now stored in the ``data`` variable with the following
structure:

.. code-block:: python

    data = {
        "0.05s": {
            "time": array([0.05, 0.1, 0.15, 0.2, ...])
            "value": array([1.000e+04, 8.464e+03, 7.396e+03, 6.527e+03, ...])
        },
        "0.2s": {
            "time": array([0.2, 0.4, 0.6, 0.8, ...])
            "value": array([1.000e+04, 6.984e+03, 5.643e+03, 4.851e+03, ...])
        },
        "1s": {
            "time": array([1., 2., 3., 4., ...])
            "value": array([1.000e+04, 6.925e+03, 5.541e+03, 4.756e+03, ...])
        },
        "5s": {
            "time": array([5., 10., 15., 20., ...])
            "value": array([1.000e+04, 6.637e+03, 5.135e+03, 4.328e+03, ...])
        },
    }


Parameters
----------

Whether we want to perform GRID fitting or multi-exponential, we need to define the 
parameters used for the fitting procedure. The GRID fitting and multi-exponential
fitting require some different variables.

GRID parameters
^^^^^^^^^^^^^^^
Parameters need to be provided in a dictionary. There are two different ways we can
perform the GRID fitting. Either use a grid as described in the paper, most often used,
or we can fixate the decay-rates and only let the amplitudes of them vary.

For the GRID fitting procedure as described in the paper, the following values
need to be provided:

* ``"k_min"``: (float) minimum decay-rate
* ``"k_max"``: (float) maximum decay-rate
* ``"N"``: (int) number of decay-rates of which the grid should consist
* ``"scale"``: (str) scale of the created grid, two options:
  * ``"log"``: logarithmic scale
  * ``"linear"``: linear scale
* ``"reg_weight"``: (float) regularization weight, advised is **0.01**.
* ``"fit_a"``: (bool) indicates whether the photobleaching number should be fitted, if set to ``False`` than a photobleaching number needs to be provided.
* ``"a_fixed"``: (float) photobleaching number used during fitting if ``parameters["fit_a"] = False``

For example:

.. code-block:: python

    parameters = {
        "k_min": 10 ** (-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

However, if you want to fixate the decay-rates then the following parameter values are
required:

* ``"k"``: (np.ndarray) array with the decay-rates
* ``"reg_weight"``: (float) regularization weight, advised is **0.01**.
* ``"fit_a"``: (bool) indicates whether the photobleaching number should be fitted, if set to ``False`` than a photobleaching number needs to be provided.
* ``"a_fixed"``: (float) photobleaching number used during fitting if ``parameters["fit_a"] = False``

For example:

.. code-block:: python

    parameters = {
        "k": np.array(
            [
                0.005,
                0.03,
                0.25,
                1.4,
                6.1,
            ],
            dtype=np.float64,
        ),
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }


Multi-exponential parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pass



The complete example:
pass