.. _basics.parameters:

Parameters
==========

Both the GRID fitting and the multi-exponential fitting require the user to specify
fitting parameters. Here we show the all the possible parameter options. Since GRID
fitting and multi-exponential fitting require some different variables, we will first
introduce the GRID parameters.

GRID
----
The fitting parameters need to be provided in a dictionary. There are two possible GRID
fitting options:

1. Define a fixed grid between a minimum and maximum decay-rate and
perform the GRID fitting procedure *(original GRID paper method)*.

2. Provide a set of decay rates and perform the GRID fitting procedure
*(newly added, not in original GRID paper)*.

For option 1, the GRID fitting procedure as described in the paper, the following
parameter values need to be provided:

* ``"k_min"``: (*float*) minimum decay-rate
* ``"k_max"``: (*float*) maximum decay-rate
* ``"N"``: (*int*) number of decay rates of which the grid should consist
* ``"scale"``: (*str*) scale of the fixed grid, two options:

  * ``"log"``: logarithmic scale
  * ``"linear"``: linear scale

* ``"reg_weight"``: (*float*) regularization weight, advised value is **0.01** *(as in the original paper)*
* ``"fit_a"``: (*bool*) determines whether the :term:`photobleaching number` should be fitted:

  * ``True``: photobleaching number is varied during the fitting
  * ``False``: photobleaching number needs to be provided and is fixed during fitting

* ``"a_fixed"``: (*float*) :term:`photobleaching number` used during fitting if
  ``parameters["fit_a"] = False`` otherwise set to ``None``

For example, if we would want to create a grid of :math:`200` decay rates with a minimum
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

For option 2, when the user provides a set of decay rates, the grid does not have to be
created anymore so the number of parameters is lower. For option 2, the following parameter
values need to be provided:

* ``"k"``: (*np.ndarray*) array with the decay rates
* ``"reg_weight"``: (*float*) regularization weight, advised value is **0.01** *(as in the original paper)*
* ``"fit_a"``: (*bool*) determines whether the :term:`photobleaching number` should be fitted:

  * ``True``: photobleaching number is varied during the fitting
  * ``False``: photobleaching number needs to be provided and is fixed during fitting

* ``"a_fixed"``: (*float*) :term:`photobleaching number` used during fitting if
  ``parameters["fit_a"] = False`` otherwise set to ``None``


For example, if we would want to perform GRID fitting with the decay rates:
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

The GRID fitting procedure can be performed with :py:func:`~gridlib.fit_grid`. For example:

.. code-block:: python
  
  import gridlib
  fit_results = gridlib.fit_grid(parameters, data)


Multi-exponential
-----------------

The fitting parameters need to be provided in a dictionary. There is only one
multi-exponential fitting option. The following parameter values need to be provided:

* ``"n_exp"``: (*int | List[int]*) number of exponentials to fit
* ``"k_min"``: (*float*) minimum decay-rate
* ``"k_max"``: (*float*) maximum decay-rate
* ``"N"``: (*int*) number of decay rates of which the grid should consist
* ``"fit_a"``: (*bool*) determines whether the :term:`photobleaching number` should be fitted:

  * ``True``: photobleaching number is varied during the fitting
  * ``False``: photobleaching number needs to be provided and is fixed during fitting

* ``"a_fixed"``: (*float*) :term:`photobleaching number` used during fitting if
  ``parameters["fit_a"] = False`` otherwise set to ``None``

For example, if you would want to fit a double-exponential (two decay rates) to the
survival time distributions with a minimum decay-rate of
:math:`10^{-3}\,\mathrm{s}^{-1}`, and a maximum decay-rate of
:math:`10\,\mathrm{s}^{-1}` and if you would want the photobleaching number to be fitted
as well then the parameter dictionary would look as follows:

.. code-block:: python
    
    parameters = {
        "n_exp": 2
        "k_min": 10**(-3),
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
    }

.. note::
    Note that the ``"n_exp"`` value is now an integer value, since we are only fitting
    a double-exponential function.

However, if you would want to fit a single-, double-, and triple-exponential function
then the parameters dictionary would look as follows:

.. code-block:: python

    parameters = {
        "n_exp": [1, 2, 3],  # fit a 1-, 2- and 3- exponential
        "k_min": 10**(-3),
        "k_max": 10**1,
        "fit_a": True,
        "a_fixed": None,
    }

.. note::
    Note that the ``"n_exp"`` value is now a list with integer values indicating the
    number of exponentials to fit, namely a single-, double- and triple-exponential.
  

The multi-exponential fitting procedure can be performed with
:py:func:`~gridlib.fit_multi_exp`. For example:

.. code-block:: python
  
  import gridlib
  fit_results = gridlib.fit_multi_exp(parameters, data)


Both
----

Option 1 of the GRID fitting procedure and the multi-exponential fitting procedure can
be combined into one function call. In this case, the parameters dictionary needs all the
required parameters for both, for example:

.. code-block:: python

  parameters = {
        "k_min": 10**(-3),      # required for: GRID and multi-exp
        "k_max": 10**1,         # required for: GRID and multi-exp
        "N": 200,               # required for: GRID
        "scale": "log",         # required for: GRID
        "reg_weight": 0.01,     # required for: GRID
        "fit_a": True,          # required for: GRID and multi-exp
        "a_fixed": None,        # required for: GRID and multi-exp
        "n_exp": [1, 2, 3],     # required for: multi-exp
    }
  
The combination of the GRID fitting procedure and multi-exponential fitting procedure
can be performed with :py:func:`~gridlib.fit_all`. For example:

.. code-block:: python
  
  import gridlib
  fit_results = gridlib.fit_all(parameters, data)