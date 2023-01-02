.. _dataformat:

Data format
===========

Survival time distributions
---------------------------

The data analysed during either GRID analysis or multi-exponential analysis are
:term:`survival time distributions<Survival time distribution>`, which should be stored
in a specific way for this package to be able to read in this data. The data should be
stored into a ``.csv`` file with two columns for every survival time distribution. The
first of these two columns should contain the time points and the column next to it
should contain the number of molecules still surviving at that point. So the structure
should be as follows in a ``.csv`` file.:

.. code-block:: text

    Δt_1,f(Δt_1),Δt_2,f(Δt_2),...
    2Δt_1,f(2Δt_1),2Δt_2,f(2Δt_2),...
    3Δt_1,f(3Δt_1),3Δt_2,f(3Δt_2),...
    4Δt_1,f(4Δt_1),4Δt_2,f(4Δt_2),...
    ...

If the survival time distribution value is zero at a certain time point than the
position for the time and the value should be empty. An example of how a ``.csv``file
would look like for four simulated survival time distributions for the time-lapse
conditions 50 ms, 200 ms, 1 s, and 5 s, all starting with 10000 molecules:

.. code-block:: text

    0.05,10000.0,0.2,10000.0,1.0,10000.0,5.0,10000.0
    0.1,8477.0,0.4,6922.0,2.0,6790.0,10.0,6669.0
    0.15,7306.0,0.6,5577.0,3.0,5493.0,15.0,5157.0
    0.2,6424.0,0.8,4807.0,4.0,4692.0,20.0,4309.0
    0.25,5701.0,1.0,4240.0,5.0,4161.0,25.0,3749.0
    0.3,5193.0,1.2,3837.0,6.0,3718.0,30.0,3338.0
    0.35,4716.0,1.4,3500.0,7.0,3384.0,35.0,3033.0
    0.4,4335.0,1.6,3218.0,8.0,3090.0,40.0,2790.0
    0.45,4037.0,1.8,3001.0,9.0,2859.0,45.0,2535.0
    ...
    19.3,12.0,77.2,10.0,386.0,1.0,,
    19.35,12.0,77.4,10.0,387.0,1.0,,
    19.4,12.0,77.6,10.0,388.0,1.0,,
    19.45,12.0,77.8,10.0,,,,
    19.5,11.0,78.0,10.0,,,,
    19.55,11.0,78.2,9.0,,,,
    19.6,11.0,78.4,9.0,,,,
    ...
    34.45,1.0,,,,,,
    34.5,1.0,,,,,,
    34.55,1.0,,,,,,
    34.6,1.0,,,,,,
    34.65,1.0,,,,,,
    34.7,1.0,,,,,,


So here it can be seen that for the first survival time distribution only 8512 molecules
are still surviving after 0.1 seconds.

An example dataset is provided at ``gridlib/examples/data/example1.csv`` on the GitHub
repository, `go to example dataset
<https://github.com/boydcpeters/gridlib/blob/master/examples/data/example1.csv>`_.

These ``.csv`` files can be read in as follows:

.. code-block:: python

    import gridlib.io
    data = gridlib.io.read_data_survival_function("/path/to/file.csv")

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

So dictionary maps the time-lapse condition to the time and value arrays.


Fit and resampling results
--------------------------

Fit results and resampling results can be stored in ``.mat`` files. GRIDLib provides
methods to read and write fit results and resampling results. Furthermore, GRIDLib
provides methods to read fit and resampling results stored by GRID toolbox (MATLAB)
created by the orginal authors. However, it is currently not possible to write the fit
results and resampling results to a file that can be opened with methods in the GRID
toolbox. The methods to read and write results can be found :ref:`here <routines.io>`.

WRITE HERE SOMETHING ABOUT THE FIT_RESULTS STRUCTURE.