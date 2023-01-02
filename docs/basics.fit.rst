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
            "value": array([1.000e+04, 8.477e+03, 7.306e+03, 6.424e+03, ...])
        },
        "0.2s": {
            "time": array([0.2, 0.4, 0.6, 0.8, ...])
            "value": array([1.000e+04, 6.922e+03, 5.577e+03, 4.807e+03, ...])
        },
        "1s": {
            "time": array([1., 2., 3., 4., ...])
            "value": array([1.000e+04, 6.790e+03, 5.493e+03, 4.692e+03, ...])
        },
        "5s": {
            "time": array([5., 10., 15., 20., ...])
            "value": array([1.000e+04, 6.669e+03, 5.157e+03, 4.309e+03, ...])
        },
    }

Now that the data is loaded in, you can decide whether you want to perform GRID fitting
and/or multi-exponential fitting. However, before you do this you need to specify the
fitting parameters.