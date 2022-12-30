.. _dataformat:

Data format
===========
All the fitting procedures require the survival time distributions. This package
provides functions to be able to read in this data. The data should be formatted in the
following way without the header in a `.csv` file.:

.. code-block:: text

    Δt_1,value,Δt_2,value,... (header)
    Δt_1,f(Δt_1),Δt_2,f(Δt_2),...
    2Δt_1,f(2Δt_1),2Δt_2,f(2Δt_2),...
    3Δt_1,f(3Δt_1),3Δt_2,f(3Δt_2),...
    4Δt_1,f(4Δt_1),4Δt_2,f(4Δt_2),...
    ...

If the survival time distribution value is zero at a certain time point than the
position should be empty. An example of how such a .csv file should like:

.. code-block:: text

    0.05,10000.0,0.2,10000.0,1.0,10000.0,5.0,10000.0
    0.1,8512.0,0.4,6984.0,2.0,6882.0,10.0,6637.0
    0.15,7337.0,0.6,5643.0,3.0,5519.0,15.0,5135.0
    0.2,6445.0,0.8,4851.0,4.0,4737.0,20.0,4328.0
    0.25,5761.0,1.0,4307.0,5.0,4215.0,25.0,3783.0
    0.3,5196.0,1.2,3896.0,6.0,3774.0,30.0,3392.0
    0.35,4751.0,1.4,3545.0,7.0,3401.0,35.0,3011.0
    0.4,4354.0,1.6,3298.0,8.0,3127.0,40.0,2767.0
    ...
    26.4,1.0,105.6,1.0,,,,
    26.45,1.0,105.8,1.0,,,,
    ,,106.0,1.0,,,,
    ,,106.2,1.0,,,,
    ,,106.4,1.0,,,,
    ,,106.6,1.0,,,,
    ,,106.8,1.0,,,,
    ,,107.0,1.0,,,,
    ,,107.2,1.0,,,,
    ,,107.4,1.0,,,,
    ,,107.6,1.0,,,,
    ,,107.8,1.0,,,,

An example dataset is also provided in the `examples` folder on the GitHub repository,
see `examples/data/example1.csv`.