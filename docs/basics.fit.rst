Fitting
=======

Here it is explained how the GRID fitting and multi-exponential fitting can be performed
with GRIDLib. However, to access all the required fitting routines the required packages
need to be imported.

Import libraries
----------------

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


Data loading
------------

Once the packages are imported, the data can be loaded in:

.. code-block:: python

    # Load the data
    >>> data = gridlib.io.read_data_survival_function("/path/to/file.csv")

The survival time distributions are now stored in the ``data`` variable. If we would use
the example data provided in ``gridlib/examples/data/example1.csv``, which had four
simulated survival time distributions than we get the following structure:

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

Now that the data is loaded in, you can decide whether you want to perform GRID fitting
and/or multi-exponential fitting. However, before you do this you need to specify the
fitting parameters.


Parameters
----------

Both the GRID fitting and the multi-exponential fitting require the user to specify
fitting parameters. Here we show the all the possible parameter options. Since GRID
fitting and multi-exponential fitting require some different variables, we will first
introduce the GRID parameters.

GRID parameters
^^^^^^^^^^^^^^^
The fitting parameters need to be provided in a dictionary. There are two possible GRID
fitting options:

1. Define a fixed grid between a minimum and maximum decay-rate and
perform the GRID fitting procedure *(original GRID paper method)*.

2. Provide a set of decay-rates and perform the GRID fitting procedure
*(newly added, not in original GRID paper)*.

For option 1, the GRID fitting procedure as described in the paper, the following
parameter values need to be provided:

* ``"k_min"``: (*float*) minimum decay-rate
* ``"k_max"``: (*float*) maximum decay-rate
* ``"N"``: (*int*) number of decay-rates of which the grid should consist
* ``"scale"``: (*str*) scale of the fixed grid, two options:

  * ``"log"``: logarithmic scale
  * ``"linear"``: linear scale

* ``"reg_weight"``: (*float*) regularization weight, advised value is **0.01** *(as in the original paper)*
* ``"fit_a"``: (*bool*) determines whether the :term:`photobleaching number` should be fitted:

  * ``True``: photobleaching number is varied during the fitting
  * ``False``: photobleaching number needs to be provided and is fixed during fitting

* ``"a_fixed"``: (*float*) :term:`photobleaching number` used during fitting if
  ``parameters["fit_a"] = False`` otherwise set to ``None``

For example, if we would want to create a grid of :math:`200` decay-rates with a minimum
decay-rate of :math:`10^{-3}\,\mathrm{s}^{-1}`, and a maximum decay-rate of
:math:`10\,\mathrm{s}^{-1}` at a logarithmic scale and we would want the photobleaching
number to be fitted as well then the parameter dictionary would look as follows:

.. code-block:: python

    parameters = {
        "k_min": 10**(-3),
        "k_max": 10**1,
        "N": 200,
        "scale": "log",
        "reg_weight": 0.01,
        "fit_a": True,
        "a_fixed": None,
    }

For option 2, when the user provides a set of decay-rates, the grid does not have to be
created anymore so the number of parameters is lower. For option 2, the following parameter
values need to be provided:

* ``"k"``: (*np.ndarray*) array with the decay-rates
* ``"reg_weight"``: (*float*) regularization weight, advised value is **0.01** *(as in the original paper)*
* ``"fit_a"``: (*bool*) determines whether the :term:`photobleaching number` should be fitted:

  * ``True``: photobleaching number is varied during the fitting
  * ``False``: photobleaching number needs to be provided and is fixed during fitting

* ``"a_fixed"``: (*float*) :term:`photobleaching number` used during fitting if
  ``parameters["fit_a"] = False`` otherwise set to ``None``


For example, if we would want to perform GRID fitting with the decay-rates:
:math:`0.005\,\mathrm{s}^{-1}`, :math:`0.03\,\mathrm{s}^{-1}`,
:math:`0.25\,\mathrm{s}^{-1}`, :math:`1.4\,\mathrm{s}^{-1}`, and
:math:`6.1\,\mathrm{s}^{-1}` and we would want the photobleaching number to be fitted
as well then the parameter dictionary would look as follows:

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